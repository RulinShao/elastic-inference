#!/usr/bin/env python3
"""
Elastic Serving Worker
========================

Runs on each SLURM-allocated GPU node (8 H200 GPUs exclusively).
Supports DP + TP parallelism:
  - TP (tensor parallelism): each vLLM/SGLang instance uses `tp_size` GPUs.
  - DP (data parallelism):  runs `gpus_per_node / tp_size` instances per node,
    each on a disjoint GPU subset, each registered as a separate worker.

Lifecycle:
1. Registers DP worker instances with the scheduler (status=LOADING).
2. Starts N = gpus_per_node / tp_size vLLM/SGLang servers, each on its own
   GPU slice (CUDA_VISIBLE_DEVICES) and port.
3. Waits until each server is healthy, then reports READY.
4. Sends periodic heartbeats for each instance.
5. On preemption / signal, kills all servers and deregisters.

Usage:
    python -m elastic_serving.worker \\
        --scheduler-url http://HEAD_NODE:8780 \\
        --engine vllm --model meta-llama/Llama-3-8B-Instruct \\
        --tensor-parallel-size 4  # DP=2 on 8-GPU node
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests as req_lib

from elastic_serving.config import WorkerStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("worker")


@dataclass
class ServerInstance:
    """A single vLLM/SGLang server instance (one DP shard)."""
    worker_id: str
    port: int
    gpu_ids: List[int]          # e.g., [0,1,2,3]
    process: Optional[subprocess.Popen] = None
    log_file: Optional[str] = None
    status: str = "LOADING"


class WorkerDaemon:
    """
    Worker daemon: manages N data-parallel vLLM/SGLang server instances
    on a single exclusive node.
    """

    def __init__(
        self,
        scheduler_url: str,
        engine: str = "vllm",
        model: str = "",
        base_port: int = 8001,
        tensor_parallel_size: int = 1,
        gpus_per_node: int = 8,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        served_model_name: Optional[str] = None,
        engine_extra_args: str = "",
        heartbeat_interval: float = 30.0,
    ):
        self.scheduler_url = scheduler_url.rstrip("/")
        self.engine = engine
        self.model = model
        self.base_port = base_port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpus_per_node = gpus_per_node
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.served_model_name = served_model_name
        self.engine_extra_args = engine_extra_args
        self.heartbeat_interval = heartbeat_interval

        self.hostname = socket.gethostname()
        self.ip_address = self._get_ip()
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID")

        # Detect actual GPUs available
        self.all_gpu_ids = self._detect_gpu_ids()
        if len(self.all_gpu_ids) < self.gpus_per_node:
            logger.warning(
                f"Detected {len(self.all_gpu_ids)} GPUs but expected {self.gpus_per_node}. "
                f"Using detected count."
            )
            self.gpus_per_node = len(self.all_gpu_ids)

        # Compute DP degree
        if self.gpus_per_node % self.tensor_parallel_size != 0:
            raise ValueError(
                f"gpus_per_node={self.gpus_per_node} not divisible by "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )
        self.dp_size = self.gpus_per_node // self.tensor_parallel_size

        # Create server instances
        self.instances: List[ServerInstance] = []
        short_id = str(uuid.uuid4())[:6]
        for dp_rank in range(self.dp_size):
            start_gpu = dp_rank * self.tensor_parallel_size
            gpu_ids = self.all_gpu_ids[start_gpu:start_gpu + self.tensor_parallel_size]
            worker_id = f"worker-{self.hostname}-dp{dp_rank}-{short_id}"
            port = self.base_port + dp_rank
            self.instances.append(ServerInstance(
                worker_id=worker_id,
                port=port,
                gpu_ids=gpu_ids,
            ))

        self.running = True
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _get_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _detect_gpu_ids(self) -> List[int]:
        """Detect available GPU IDs."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                n = len(result.stdout.strip().split("\n"))
                return list(range(n))
        except Exception:
            pass
        return list(range(8))

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down all instances...")
        self.running = False

    # ---- Scheduler Communication ----

    def _api_call(self, method: str, endpoint: str, data=None, timeout: float = 30.0):
        url = f"{self.scheduler_url}{endpoint}"
        try:
            if method == "GET":
                resp = req_lib.get(url, timeout=timeout)
            elif method == "POST":
                resp = req_lib.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unknown method: {method}")
            resp.raise_for_status()
            return resp.json()
        except req_lib.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to scheduler at {self.scheduler_url}")
            return None
        except Exception as e:
            logger.warning(f"Error calling scheduler: {e}")
            return None

    def _register_instance(self, inst: ServerInstance) -> bool:
        resp = self._api_call("POST", "/register_worker", {
            "worker_id": inst.worker_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "port": inst.port,
            "engine": self.engine,
            "model_name": self.model,
            "n_gpus": self.tensor_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "slurm_job_id": self.slurm_job_id,
        })
        if resp and resp.get("status") == "ok":
            logger.info(
                f"Registered {inst.worker_id} "
                f"(port={inst.port}, GPUs={inst.gpu_ids})"
            )
            return True
        logger.error(f"Failed to register {inst.worker_id}")
        return False

    def _deregister_instance(self, inst: ServerInstance):
        self._api_call("POST", f"/deregister_worker/{inst.worker_id}")

    def _send_heartbeat(self, inst: ServerInstance, status: str):
        self._api_call("POST", "/heartbeat", {
            "worker_id": inst.worker_id,
            "status": status,
        })

    # ---- Server Management ----

    def _build_server_cmd(self, inst: ServerInstance) -> list:
        """Build command to start vLLM or SGLang server."""
        if self.engine == "vllm":
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model,
                "--port", str(inst.port),
                "--tensor-parallel-size", str(self.tensor_parallel_size),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--trust-remote-code",
                "--disable-log-requests",
            ]
            if self.max_model_len is not None:
                cmd.extend(["--max-model-len", str(self.max_model_len)])
            if self.served_model_name:
                cmd.extend(["--served-model-name", self.served_model_name])
        elif self.engine == "sglang":
            cmd = [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", self.model,
                "--port", str(inst.port),
                "--tp", str(self.tensor_parallel_size),
                "--mem-fraction-static", str(self.gpu_memory_utilization),
                "--trust-remote-code",
                "--disable-radix-cache",
            ]
            if self.max_model_len is not None:
                cmd.extend(["--context-length", str(self.max_model_len)])
            if self.served_model_name:
                cmd.extend(["--served-model-name", self.served_model_name])
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

        # Append extra args
        if self.engine_extra_args:
            import shlex
            cmd.extend(shlex.split(self.engine_extra_args))

        return cmd

    def _start_instance(self, inst: ServerInstance):
        """Start a single server instance on its GPU slice."""
        cmd = self._build_server_cmd(inst)

        # Set CUDA_VISIBLE_DEVICES for this DP shard
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inst.gpu_ids)

        log_path = f"/tmp/elastic_{inst.worker_id}.log"
        inst.log_file = log_path
        log_file = open(log_path, "w")

        logger.info(
            f"Starting {self.engine} instance {inst.worker_id}: "
            f"GPUs={inst.gpu_ids}, port={inst.port}"
        )
        logger.info(f"  Command: {' '.join(cmd)}")

        inst.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logger.info(f"  PID={inst.process.pid}, log={log_path}")

    def _stop_instance(self, inst: ServerInstance):
        if inst.process:
            logger.info(f"Stopping {inst.worker_id} (PID={inst.process.pid})...")
            inst.process.terminate()
            try:
                inst.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                inst.process.kill()
                inst.process.wait(timeout=10)
            inst.process = None
            inst.status = "OFFLINE"

    def _wait_for_ready(self, inst: ServerInstance, timeout: float = 900.0, poll: float = 5.0) -> bool:
        """Poll server health endpoint until ready."""
        url = f"http://localhost:{inst.port}/health"
        start = time.time()
        while time.time() - start < timeout:
            if not self.running:
                return False
            if inst.process and inst.process.poll() is not None:
                logger.error(
                    f"{inst.worker_id} process died with code {inst.process.returncode}"
                )
                return False
            try:
                resp = req_lib.get(url, timeout=5)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    logger.info(f"{inst.worker_id} is READY (took {elapsed:.1f}s)")
                    return True
            except Exception:
                pass
            # Heartbeat while loading
            self._send_heartbeat(inst, WorkerStatus.LOADING.value)
            time.sleep(poll)
        logger.error(f"{inst.worker_id} did not become ready within {timeout}s")
        return False

    # ---- Heartbeat loop ----

    def _heartbeat_loop(self):
        while self.running:
            for inst in self.instances:
                if inst.status == "READY":
                    alive = inst.process and inst.process.poll() is None
                    status = WorkerStatus.READY.value if alive else WorkerStatus.OFFLINE.value
                    self._send_heartbeat(inst, status)
            time.sleep(self.heartbeat_interval)

    # ---- Main ----

    def run(self):
        logger.info("=" * 60)
        logger.info("Elastic Serving Worker (DP + TP)")
        logger.info(f"  Hostname:    {self.hostname}")
        logger.info(f"  IP:          {self.ip_address}")
        logger.info(f"  Engine:      {self.engine}")
        logger.info(f"  Model:       {self.model}")
        logger.info(f"  GPUs/node:   {self.gpus_per_node}")
        logger.info(f"  TP size:     {self.tensor_parallel_size}")
        logger.info(f"  DP size:     {self.dp_size}")
        logger.info(f"  GPU IDs:     {self.all_gpu_ids}")
        logger.info(f"  SLURM Job:   {self.slurm_job_id}")
        logger.info(f"  Scheduler:   {self.scheduler_url}")
        for inst in self.instances:
            logger.info(f"  Instance:    {inst.worker_id} port={inst.port} GPUs={inst.gpu_ids}")
        logger.info("=" * 60)

        # Register all instances with scheduler
        for inst in self.instances:
            for attempt in range(10):
                if self._register_instance(inst):
                    break
                wait = min(2 ** attempt, 60)
                logger.warning(
                    f"Registration of {inst.worker_id} failed "
                    f"(attempt {attempt+1}/10), retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Failed to register {inst.worker_id} after 10 attempts.")
                # Continue with other instances

        # Start server instances with staggered startup to avoid race
        # conditions (e.g. both vLLM instances loading Harmony encoding
        # simultaneously can cause panics).
        # Strategy: start dp0, wait for it to be ready, then start dp1, etc.
        for idx, inst in enumerate(self.instances):
            self._start_instance(inst)

            if idx < len(self.instances) - 1:
                # Wait for this instance to be ready before starting the next
                logger.info(
                    f"Waiting for {inst.worker_id} to be ready before "
                    f"starting next DP instance..."
                )
                ok = self._wait_for_ready(inst)
                if ok:
                    inst.status = "READY"
                    self._send_heartbeat(inst, WorkerStatus.READY.value)
                    logger.info(f"{inst.worker_id} is READY")
                else:
                    inst.status = "FAILED"
                    logger.error(f"{inst.worker_id} FAILED to start")
                    self._stop_instance(inst)
                    self._deregister_instance(inst)
                    # Continue starting remaining instances anyway

        # Wait for the last instance
        last = self.instances[-1]
        if last.status != "READY" and last.status != "FAILED":
            ok = self._wait_for_ready(last)
            if ok:
                last.status = "READY"
                self._send_heartbeat(last, WorkerStatus.READY.value)
                logger.info(f"{last.worker_id} is READY")
            else:
                last.status = "FAILED"
                logger.error(f"{last.worker_id} FAILED to start")
                self._stop_instance(last)
                self._deregister_instance(last)

        ready_count = sum(1 for inst in self.instances if inst.status == "READY")
        if ready_count == 0:
            logger.error("No instances started successfully. Exiting.")
            sys.exit(1)

        logger.info(f"{ready_count}/{self.dp_size} DP instances READY")

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        heartbeat_thread.start()

        # Wait for any server process to exit
        try:
            while self.running:
                any_alive = False
                for inst in self.instances:
                    if inst.process and inst.process.poll() is None:
                        any_alive = True
                    elif inst.status == "READY":
                        logger.warning(
                            f"{inst.worker_id} process exited "
                            f"(code={inst.process.returncode if inst.process else '?'})"
                        )
                        inst.status = "OFFLINE"
                        self._send_heartbeat(inst, WorkerStatus.OFFLINE.value)
                if not any_alive:
                    logger.info("All server processes have exited")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

        # Cleanup
        for inst in self.instances:
            self._stop_instance(inst)
            self._deregister_instance(inst)
        logger.info("Worker daemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="Elastic Serving Worker (DP + TP)")
    parser.add_argument("--scheduler-url", required=True,
                        help="URL of the elastic scheduler")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--base-port", type=int, default=8001,
                        help="Base port for vLLM/SGLang servers (DP instances use base+0, base+1, ...)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="TP size per instance")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="Total GPUs on this node")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--engine-extra-args", type=str, default="")
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    args = parser.parse_args()

    daemon = WorkerDaemon(
        scheduler_url=args.scheduler_url,
        engine=args.engine,
        model=args.model,
        base_port=args.base_port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        served_model_name=args.served_model_name,
        engine_extra_args=args.engine_extra_args,
        heartbeat_interval=args.heartbeat_interval,
    )
    daemon.run()


if __name__ == "__main__":
    main()
