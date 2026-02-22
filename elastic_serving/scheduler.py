#!/usr/bin/env python3
"""
Elastic Serving Scheduler
============================

A lightweight FastAPI server (no GPU) that:
1. Greedily acquires SLURM nodes on h200_lowest QoS up to max_nodes.
2. Each node starts multiple vLLM/SGLang instances (DP + TP parallelism)
   and each instance registers back as a separate worker.
3. Exposes an OpenAI-compatible /v1/... proxy that load-balances across
   all ready workers (round-robin over all DP×nodes instances).
4. Monitors worker health via heartbeats; replaces dead/preempted nodes.

Usage:
    python -m elastic_serving.scheduler --model <model> [options]
"""

import argparse
import itertools
import json as _json
import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from elastic_serving.config import (
    NodeRequest,
    NodeRequestStatus,
    SchedulerConfig,
    WorkerInfo,
    WorkerStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scheduler")


# =============================================================================
# Request / Response models
# =============================================================================

class RegisterWorkerRequest(BaseModel):
    worker_id: str
    hostname: str
    ip_address: str
    port: int
    engine: str = "vllm"
    model_name: str = ""
    n_gpus: int = 8
    tensor_parallel_size: int = 1
    slurm_job_id: Optional[str] = None


class HeartbeatRequest(BaseModel):
    worker_id: str
    status: Optional[str] = None
    requests_served: Optional[int] = None


class WorkerStatusResponse(BaseModel):
    worker_id: str
    hostname: str
    ip_address: str
    port: int
    base_url: str
    status: str
    engine: str
    model_name: str
    n_gpus: int
    tensor_parallel_size: int
    slurm_job_id: Optional[str] = None
    registered_at: str
    last_heartbeat: str
    requests_served: int


class ClusterStatusResponse(BaseModel):
    model: str
    engine: str
    max_nodes: int
    tensor_parallel_size: int
    dp_per_node: int
    total_nodes_active: int
    total_workers: int
    ready_workers: int
    loading_workers: int
    offline_workers: int
    pending_slurm_jobs: int
    workers: List[WorkerStatusResponse]


# =============================================================================
# Scheduler Core
# =============================================================================


class AdaptiveScheduler:
    """
    Core scheduler logic:
    - Manages worker registry, heartbeats, health checks.
    - Greedily submits SLURM jobs to fill up to max_nodes (8 GPUs each).
    - Each node runs gpus_per_node/tp_size DP instances.
    - Round-robin load balancing across all READY worker instances.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.lock = threading.Lock()
        self.workers: Dict[str, WorkerInfo] = {}
        self.node_requests: Dict[str, NodeRequest] = {}  # slurm_job_id -> NodeRequest
        self._rr_cycle = itertools.cycle([])  # round-robin iterator
        self._rr_workers: List[str] = []

    # ---- Worker management ----

    def register_worker(self, req: RegisterWorkerRequest) -> WorkerInfo:
        with self.lock:
            now = datetime.now().isoformat()
            w = WorkerInfo(
                worker_id=req.worker_id,
                hostname=req.hostname,
                ip_address=req.ip_address,
                port=req.port,
                engine=req.engine,
                model_name=req.model_name,
                n_gpus=req.n_gpus,
                tensor_parallel_size=req.tensor_parallel_size,
                slurm_job_id=req.slurm_job_id,
                status=WorkerStatus.LOADING,
                registered_at=now,
                last_heartbeat=now,
            )
            self.workers[req.worker_id] = w

            # Link to node request
            if req.slurm_job_id and req.slurm_job_id in self.node_requests:
                nr = self.node_requests[req.slurm_job_id]
                nr.status = NodeRequestStatus.RUNNING
                nr.started_at = nr.started_at or now
                nr.worker_ids.append(req.worker_id)

            logger.info(
                f"Worker {req.worker_id} registered "
                f"({req.hostname}:{req.port}, {req.engine}, "
                f"TP={req.tensor_parallel_size}, {req.n_gpus} GPUs, "
                f"SLURM {req.slurm_job_id})"
            )
            return w

    def heartbeat(self, worker_id: str, status: Optional[str] = None,
                  requests_served: Optional[int] = None) -> bool:
        with self.lock:
            w = self.workers.get(worker_id)
            if not w:
                return False
            w.last_heartbeat = datetime.now().isoformat()
            if status:
                old_status = w.status
                w.status = WorkerStatus(status)
                if old_status != w.status:
                    logger.info(f"Worker {worker_id}: {old_status.value} -> {w.status.value}")
                    if w.status == WorkerStatus.READY:
                        self._rebuild_rr()
            if requests_served is not None:
                w.requests_served = requests_served
            return True

    def deregister_worker(self, worker_id: str) -> bool:
        with self.lock:
            if worker_id in self.workers:
                w = self.workers.pop(worker_id)
                logger.info(f"Worker {worker_id} deregistered ({w.hostname})")
                self._rebuild_rr()
                return True
            return False

    def _rebuild_rr(self):
        """Rebuild round-robin cycle for READY workers. Must hold self.lock."""
        self._rr_workers = [
            wid for wid, w in self.workers.items()
            if w.status == WorkerStatus.READY
        ]
        self._rr_cycle = itertools.cycle(self._rr_workers) if self._rr_workers else itertools.cycle([])

    def pick_worker(self) -> Optional[WorkerInfo]:
        """Pick next READY worker via round-robin."""
        with self.lock:
            if not self._rr_workers:
                return None
            for _ in range(len(self._rr_workers)):
                wid = next(self._rr_cycle)
                w = self.workers.get(wid)
                if w and w.status == WorkerStatus.READY:
                    return w
            return None

    def get_ready_workers(self) -> List[WorkerInfo]:
        with self.lock:
            return [w for w in self.workers.values() if w.status == WorkerStatus.READY]

    # ---- Health checking ----

    def check_worker_health(self):
        """Mark workers as OFFLINE if heartbeat timeout exceeded."""
        with self.lock:
            now = datetime.now()
            changed = False
            for w in self.workers.values():
                if w.status == WorkerStatus.OFFLINE:
                    continue
                try:
                    last = datetime.fromisoformat(w.last_heartbeat)
                    delta = (now - last).total_seconds()
                    if delta > self.config.heartbeat_timeout:
                        logger.warning(
                            f"Worker {w.worker_id} timed out "
                            f"({delta:.0f}s since heartbeat), marking OFFLINE"
                        )
                        w.status = WorkerStatus.OFFLINE
                        changed = True
                except Exception as e:
                    logger.warning(f"Error checking worker {w.worker_id}: {e}")
            if changed:
                self._rebuild_rr()

    def cleanup_offline_workers(self):
        """Remove offline workers. If ALL workers for a node are offline,
        mark the node request as FAILED so a replacement can be acquired."""
        with self.lock:
            offline_ids = [
                wid for wid, w in self.workers.items()
                if w.status == WorkerStatus.OFFLINE
            ]
            for wid in offline_ids:
                w = self.workers.pop(wid)
                logger.info(f"Cleaned up offline worker {wid}")

            # Check which node requests have zero active workers left
            for nr in self.node_requests.values():
                if nr.status in (NodeRequestStatus.RUNNING, NodeRequestStatus.PENDING):
                    alive = any(
                        wid in self.workers and self.workers[wid].status != WorkerStatus.OFFLINE
                        for wid in nr.worker_ids
                    )
                    if nr.worker_ids and not alive:
                        nr.status = NodeRequestStatus.FAILED
                        logger.info(
                            f"Node request {nr.slurm_job_id}: all workers offline, "
                            f"marking FAILED for replacement"
                        )

            if offline_ids:
                self._rebuild_rr()

    # ---- SLURM node acquisition ----

    def _active_node_count(self) -> int:
        """Count active nodes = non-failed node requests + pending requests.
        We count by unique SLURM jobs, not individual DP workers."""
        count = 0
        for nr in self.node_requests.values():
            if nr.status in (NodeRequestStatus.PENDING, NodeRequestStatus.RUNNING):
                count += 1
        return count

    def should_acquire_node(self) -> bool:
        """Return True if we should submit another SLURM job."""
        with self.lock:
            return self._active_node_count() < self.config.max_nodes

    def submit_slurm_job(self) -> Optional[str]:
        """Submit a SLURM job to acquire a new worker node. Returns job ID."""
        cfg = self.config

        # Build sbatch script
        script_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=elastic-worker",
            "#SBATCH --nodes=1",
            f"#SBATCH --gpus-per-node={cfg.gpus_per_node}",
            f"#SBATCH --time={cfg.time_limit}",
            f"#SBATCH --output={cfg.log_dir}/worker_%j.out",
            f"#SBATCH --error={cfg.log_dir}/worker_%j.err",
        ]

        # Exclusive node access for full utilization
        if cfg.exclusive:
            script_lines.append("#SBATCH --exclusive")

        if cfg.qos:
            script_lines.append(f"#SBATCH --qos={cfg.qos}")
        if cfg.partition:
            script_lines.append(f"#SBATCH --partition={cfg.partition}")
        if cfg.account:
            script_lines.append(f"#SBATCH --account={cfg.account}")
        if cfg.constraint:
            script_lines.append(f"#SBATCH --constraint={cfg.constraint}")
        if cfg.slurm_extra:
            for flag in cfg.slurm_extra.split(";"):
                flag = flag.strip()
                if flag:
                    script_lines.append(f"#SBATCH {flag}")

        # Discover scheduler URL
        scheduler_host = self._get_hostname()
        scheduler_url = f"http://{scheduler_host}:{cfg.port}"

        script_lines.extend([
            "",
            "set -e",
            "",
        ])

        # Conda activation — try common conda install locations
        if cfg.conda_env:
            script_lines.extend([
                "# Activate conda",
                'for conda_sh in /opt/conda/etc/profile.d/conda.sh "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do',
                '    if [ -f "$conda_sh" ]; then',
                '        source "$conda_sh"',
                '        break',
                '    fi',
                'done',
                f'conda activate {cfg.conda_env}',
                "",
            ])

        if cfg.project_root:
            script_lines.append(f'export PYTHONPATH="{cfg.project_root}:${{PYTHONPATH}}"')

        script_lines.append("")
        script_lines.append(f'echo "Node $(hostname) starting worker with {cfg.dp_per_node} DP instances (TP={cfg.tensor_parallel_size})"')
        script_lines.append('nvidia-smi || true')
        script_lines.append("")

        # Build worker command — note --base-port and --gpus-per-node for DP
        worker_cmd = (
            f"python -m elastic_serving.worker "
            f"--scheduler-url {scheduler_url} "
            f"--engine {cfg.engine} "
            f"--model {cfg.model} "
            f"--base-port {cfg.worker_base_port} "
            f"--tensor-parallel-size {cfg.tensor_parallel_size} "
            f"--gpus-per-node {cfg.gpus_per_node} "
            f"--gpu-memory-utilization {cfg.gpu_memory_utilization}"
        )
        if cfg.max_model_len is not None:
            worker_cmd += f" --max-model-len {cfg.max_model_len}"
        if cfg.served_model_name:
            worker_cmd += f" --served-model-name {cfg.served_model_name}"
        if cfg.engine_extra_args:
            worker_cmd += f" --engine-extra-args '{cfg.engine_extra_args}'"

        script_lines.append(worker_cmd)

        # Write script to temp file and sbatch
        os.makedirs(cfg.log_dir, exist_ok=True)
        script_content = "\n".join(script_lines) + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", dir=cfg.log_dir, delete=False, prefix="sbatch_"
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            result = subprocess.run(
                ["sbatch", "--parsable", script_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"sbatch failed: {result.stderr.strip()}")
                return None

            slurm_job_id = result.stdout.strip().split(";")[0]

            with self.lock:
                self.node_requests[slurm_job_id] = NodeRequest(
                    slurm_job_id=slurm_job_id,
                    status=NodeRequestStatus.PENDING,
                )
            logger.info(
                f"Submitted SLURM job {slurm_job_id} "
                f"(QoS={cfg.qos}, partition={cfg.partition}, "
                f"DP={cfg.dp_per_node}×TP={cfg.tensor_parallel_size})"
            )
            return slurm_job_id

        except subprocess.TimeoutExpired:
            logger.error("sbatch timed out")
            return None
        except FileNotFoundError:
            logger.error("sbatch not found — is SLURM installed?")
            return None

    def check_pending_slurm_jobs(self):
        """Check if pending SLURM jobs are still in queue."""
        with self.lock:
            pending = [
                nr for nr in self.node_requests.values()
                if nr.status == NodeRequestStatus.PENDING
            ]

        for nr in pending:
            try:
                result = subprocess.run(
                    ["squeue", "--job", nr.slurm_job_id, "--noheader", "-o", "%T"],
                    capture_output=True, text=True, timeout=10,
                )
                state = result.stdout.strip().upper()
                if not state:
                    with self.lock:
                        if nr.status == NodeRequestStatus.PENDING:
                            if nr.worker_ids:
                                nr.status = NodeRequestStatus.RUNNING
                            else:
                                nr.status = NodeRequestStatus.FAILED
                                logger.warning(
                                    f"SLURM job {nr.slurm_job_id} disappeared from queue"
                                )
                elif state == "RUNNING":
                    with self.lock:
                        if nr.status == NodeRequestStatus.PENDING:
                            nr.status = NodeRequestStatus.RUNNING
                            nr.started_at = datetime.now().isoformat()
            except Exception as e:
                logger.warning(f"Error checking SLURM job {nr.slurm_job_id}: {e}")

    def _get_hostname(self) -> str:
        """Get IP address reachable by worker nodes (FQDN may not resolve)."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return socket.getfqdn()

    # ---- Status ----

    def get_cluster_status(self) -> Dict[str, Any]:
        with self.lock:
            workers_list = []
            for w in self.workers.values():
                workers_list.append(WorkerStatusResponse(
                    worker_id=w.worker_id,
                    hostname=w.hostname,
                    ip_address=w.ip_address,
                    port=w.port,
                    base_url=w.base_url,
                    status=w.status.value,
                    engine=w.engine,
                    model_name=w.model_name,
                    n_gpus=w.n_gpus,
                    tensor_parallel_size=w.tensor_parallel_size,
                    slurm_job_id=w.slurm_job_id,
                    registered_at=w.registered_at,
                    last_heartbeat=w.last_heartbeat,
                    requests_served=w.requests_served,
                ))
            pending = sum(
                1 for nr in self.node_requests.values()
                if nr.status == NodeRequestStatus.PENDING
            )
            active_nodes = sum(
                1 for nr in self.node_requests.values()
                if nr.status in (NodeRequestStatus.PENDING, NodeRequestStatus.RUNNING)
            )
            return ClusterStatusResponse(
                model=self.config.model,
                engine=self.config.engine,
                max_nodes=self.config.max_nodes,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dp_per_node=self.config.dp_per_node,
                total_nodes_active=active_nodes,
                total_workers=len(self.workers),
                ready_workers=sum(1 for w in self.workers.values() if w.status == WorkerStatus.READY),
                loading_workers=sum(1 for w in self.workers.values() if w.status == WorkerStatus.LOADING),
                offline_workers=sum(1 for w in self.workers.values() if w.status == WorkerStatus.OFFLINE),
                pending_slurm_jobs=pending,
                workers=workers_list,
            ).model_dump()


# =============================================================================
# FastAPI Application
# =============================================================================


def create_app(scheduler: AdaptiveScheduler) -> FastAPI:
    app = FastAPI(
        title="Elastic Serving Scheduler",
        description="Auto-scaling vLLM/SGLang on SLURM low-priority H200 resources",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    # ---- Internal endpoints ----

    @app.get("/")
    async def root():
        ready = scheduler.get_ready_workers()
        return {
            "service": "Elastic Serving Scheduler",
            "version": "0.1.0",
            "model": scheduler.config.model,
            "engine": scheduler.config.engine,
            "ready_workers": len(ready),
            "dp_per_node": scheduler.config.dp_per_node,
            "tp_size": scheduler.config.tensor_parallel_size,
        }

    @app.get("/health")
    async def health():
        ready = scheduler.get_ready_workers()
        return {"status": "ok", "ready_workers": len(ready)}

    @app.post("/register_worker")
    async def register_worker(req: RegisterWorkerRequest):
        w = scheduler.register_worker(req)
        return {"status": "ok", "worker_id": w.worker_id}

    @app.post("/heartbeat")
    async def heartbeat(req: HeartbeatRequest):
        ok = scheduler.heartbeat(req.worker_id, req.status, req.requests_served)
        if not ok:
            raise HTTPException(status_code=404, detail="Worker not found")
        return {"status": "ok"}

    @app.post("/deregister_worker/{worker_id}")
    async def deregister_worker(worker_id: str):
        ok = scheduler.deregister_worker(worker_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Worker not found")
        return {"status": "ok", "message": f"Worker {worker_id} deregistered"}

    @app.get("/cluster_status")
    async def cluster_status():
        return scheduler.get_cluster_status()

    @app.get("/workers")
    async def list_workers():
        with scheduler.lock:
            return {
                "workers": [
                    {
                        "worker_id": w.worker_id,
                        "base_url": w.base_url,
                        "status": w.status.value,
                        "hostname": w.hostname,
                        "requests_served": w.requests_served,
                    }
                    for w in scheduler.workers.values()
                ]
            }

    # ---- OpenAI-compatible proxy endpoints ----

    @app.get("/v1/models")
    async def list_models():
        worker = scheduler.pick_worker()
        if not worker:
            return {"object": "list", "data": []}
        try:
            resp = await http_client.get(f"{worker.base_url}/models")
            return resp.json()
        except Exception as e:
            logger.warning(f"Error proxying /v1/models to {worker.worker_id}: {e}")
            return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_post(request, "/chat/completions", scheduler, http_client)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_post(request, "/completions", scheduler, http_client)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/embeddings", scheduler, http_client)

    return app


async def _proxy_post(
    request: Request,
    path: str,
    scheduler: AdaptiveScheduler,
    http_client: httpx.AsyncClient,
):
    """Proxy a POST request to a ready worker with streaming support."""
    worker = scheduler.pick_worker()
    if not worker:
        raise HTTPException(
            status_code=503,
            detail="No ready workers available. Workers may still be starting.",
        )

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    target_url = f"{worker.base_url}{path}"

    try:
        body_json = _json.loads(body)
        is_stream = body_json.get("stream", False)
    except Exception:
        is_stream = False

    if is_stream:
        async def stream_generator():
            try:
                async with http_client.stream(
                    "POST", target_url, content=body, headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except Exception as e:
                logger.error(f"Streaming error from {worker.worker_id}: {e}")
                yield f"data: {_json.dumps({'error': str(e)})}\n\n".encode()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    else:
        try:
            resp = await http_client.post(target_url, content=body, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.ConnectError:
            logger.warning(f"Connection failed to {worker.worker_id}, marking offline")
            with scheduler.lock:
                w = scheduler.workers.get(worker.worker_id)
                if w:
                    w.status = WorkerStatus.OFFLINE
                    scheduler._rebuild_rr()

            worker2 = scheduler.pick_worker()
            if not worker2:
                raise HTTPException(status_code=503, detail="No ready workers available")
            target_url2 = f"{worker2.base_url}{path}"
            resp = await http_client.post(target_url2, content=body, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )


# =============================================================================
# Background loops
# =============================================================================


def start_background_loops(scheduler: AdaptiveScheduler):
    def _health_loop():
        while True:
            try:
                scheduler.check_worker_health()
                scheduler.cleanup_offline_workers()
                scheduler.check_pending_slurm_jobs()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            time.sleep(scheduler.config.health_check_interval)

    def _acquire_loop():
        """Greedy node acquisition: always try to fill up to max_nodes."""
        time.sleep(5)
        while True:
            try:
                if scheduler.should_acquire_node():
                    with scheduler.lock:
                        active = scheduler._active_node_count()
                    logger.info(
                        f"Acquiring new node "
                        f"({active}/{scheduler.config.max_nodes} active, "
                        f"DP={scheduler.config.dp_per_node}×TP={scheduler.config.tensor_parallel_size})"
                    )
                    scheduler.submit_slurm_job()
                    time.sleep(3)
                else:
                    time.sleep(scheduler.config.node_acquire_interval)
            except Exception as e:
                logger.error(f"Node acquisition error: {e}")
                time.sleep(scheduler.config.node_acquire_interval)

    t1 = threading.Thread(target=_health_loop, daemon=True, name="health-checker")
    t2 = threading.Thread(target=_acquire_loop, daemon=True, name="node-acquirer")
    t1.start()
    t2.start()
    logger.info("Background loops started (health checker + node acquirer)")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Elastic Serving Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve Llama on up to 16 H200 nodes, TP=4 (DP=2 per node)
  python -m elastic_serving.scheduler \\
      --model meta-llama/Llama-3-70B-Instruct \\
      --tensor-parallel-size 4 --max-nodes 16

  # Serve smaller model with full DP (TP=1, DP=8 per node)
  python -m elastic_serving.scheduler \\
      --model meta-llama/Llama-3-8B-Instruct \\
      --tensor-parallel-size 1 --max-nodes 16

  # Use SGLang engine
  python -m elastic_serving.scheduler \\
      --model Qwen/Qwen3-32B --engine sglang \\
      --tensor-parallel-size 4 --max-nodes 8
""",
    )
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model", type=str, help="Model name/path to serve")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS (default: h200_lowest)")
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition (default: h200)")
    parser.add_argument("--account", type=str, default=None, help="SLURM account (default: dream)")
    parser.add_argument("--max-nodes", type=int, default=None, help="Max SLURM nodes (default: 16)")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node (8 on H200)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="TP size per instance. DP = gpus_per_node / TP.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--engine-extra-args", type=str, default=None,
                        help="Extra arguments passed to vLLM/SGLang server")
    parser.add_argument("--time-limit", type=str, default=None, help="SLURM time limit per worker")
    parser.add_argument("--constraint", type=str, default=None)
    parser.add_argument("--slurm-extra", type=str, default=None,
                        help="Extra sbatch flags (semicolon-separated)")
    parser.add_argument("--port", type=int, default=None, help="Scheduler port (default: 8780)")
    parser.add_argument("--worker-base-port", type=int, default=None,
                        help="Base port for DP instances on each node (default: 8001)")
    parser.add_argument("--heartbeat-timeout", type=float, default=None)
    parser.add_argument("--health-check-interval", type=float, default=None)
    parser.add_argument("--node-acquire-interval", type=float, default=None)
    parser.add_argument("--conda-env", type=str, default=None,
                        help="Conda env to activate on worker nodes")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # Build config
    if args.config:
        config = SchedulerConfig.from_json_file(args.config)
    else:
        config = SchedulerConfig()

    # Override from CLI — only override if explicitly provided
    cli_overrides = {
        "model": args.model,
        "engine": args.engine,
        "qos": args.qos,
        "partition": args.partition,
        "account": args.account,
        "max_nodes": args.max_nodes,
        "gpus_per_node": args.gpus_per_node,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "served_model_name": args.served_model_name,
        "engine_extra_args": args.engine_extra_args,
        "time_limit": args.time_limit,
        "constraint": args.constraint,
        "slurm_extra": args.slurm_extra,
        "port": args.port,
        "worker_base_port": args.worker_base_port,
        "heartbeat_timeout": args.heartbeat_timeout,
        "health_check_interval": args.health_check_interval,
        "node_acquire_interval": args.node_acquire_interval,
        "conda_env": args.conda_env,
        "project_root": args.project_root,
        "log_dir": args.log_dir,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            setattr(config, key, val)

    if not config.model:
        parser.error("--model is required")

    if not config.project_root:
        config.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validate DP/TP
    if config.gpus_per_node % config.tensor_parallel_size != 0:
        parser.error(
            f"gpus_per_node={config.gpus_per_node} not divisible by "
            f"tensor_parallel_size={config.tensor_parallel_size}"
        )

    # Create scheduler and app
    scheduler = AdaptiveScheduler(config)
    app = create_app(scheduler)
    start_background_loops(scheduler)

    logger.info("=" * 60)
    logger.info("Elastic Serving Scheduler")
    logger.info("=" * 60)
    logger.info(f"Model:        {config.model}")
    logger.info(f"Engine:       {config.engine}")
    logger.info(f"QoS:          {config.qos}")
    logger.info(f"Partition:    {config.partition}")
    logger.info(f"Account:      {config.account}")
    logger.info(f"Max nodes:    {config.max_nodes}")
    logger.info(f"GPUs/node:    {config.gpus_per_node}")
    logger.info(f"TP size:      {config.tensor_parallel_size}")
    logger.info(f"DP/node:      {config.dp_per_node}")
    logger.info(f"Max workers:  {config.max_nodes * config.dp_per_node}")
    logger.info(f"Max GPUs:     {config.max_nodes * config.gpus_per_node}")
    logger.info(f"Exclusive:    {config.exclusive}")
    logger.info(f"Port:         {config.port}")
    logger.info(f"Worker ports: {config.worker_base_port}-{config.worker_base_port + config.dp_per_node - 1}")
    logger.info(f"Log dir:      {config.log_dir}")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
