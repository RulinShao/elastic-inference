"""
Configuration and data models for Elastic Serving.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================


class WorkerStatus(str, Enum):
    """Status of a worker node."""
    STARTING = "STARTING"   # SLURM job submitted, waiting for node
    LOADING = "LOADING"     # Node acquired, vLLM/SGLang loading model
    READY = "READY"         # Server is up and accepting requests
    OFFLINE = "OFFLINE"     # Heartbeat timeout / preempted


class NodeRequestStatus(str, Enum):
    """Status of a SLURM node request."""
    PENDING = "PENDING"     # sbatch submitted, waiting in queue
    RUNNING = "RUNNING"     # SLURM job running, worker starting
    COMPLETED = "COMPLETED" # Worker finished (or cancelled cleanly)
    FAILED = "FAILED"       # SLURM job failed or preempted


# =============================================================================
# Worker / Node Info
# =============================================================================


@dataclass
class WorkerInfo:
    """Information about a registered vLLM/SGLang worker."""
    worker_id: str
    hostname: str
    ip_address: str
    port: int                           # vLLM/SGLang server port on this node
    status: WorkerStatus = WorkerStatus.STARTING
    engine: str = "vllm"                # "vllm" or "sglang"
    model_name: str = ""
    n_gpus: int = 8
    tensor_parallel_size: int = 1
    slurm_job_id: Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    requests_served: int = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.ip_address}:{self.port}/v1"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["base_url"] = self.base_url
        return d


@dataclass
class NodeRequest:
    """Tracks a SLURM job submission for a worker node."""
    slurm_job_id: str
    status: NodeRequestStatus = NodeRequestStatus.PENDING
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    worker_ids: List[str] = field(default_factory=list)  # DP instances from this node


# =============================================================================
# Scheduler Config (loaded from env / CLI)
# =============================================================================


@dataclass
class SchedulerConfig:
    """Configuration for the elastic scheduler."""
    # SLURM settings
    qos: str = "h200_lowest"
    partition: str = "h200"
    account: str = "dream"
    max_nodes: int = 16
    gpus_per_node: int = 8
    time_limit: str = "72:00:00"
    exclusive: bool = True                 # request exclusive node access
    constraint: str = ""                   # e.g., "volta32gb"
    slurm_extra: str = ""                  # extra sbatch flags

    # Model serving — DP + TP parallelism
    # Each node runs (gpus_per_node / tensor_parallel_size) server instances.
    # E.g. 8 GPUs, TP=4 → 2 vLLM instances per node (DP=2).
    engine: str = "vllm"                   # "vllm" or "sglang"
    model: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    served_model_name: Optional[str] = None
    engine_extra_args: str = ""            # extra args for vllm/sglang
    enable_prefix_caching: bool = True     # disable for Mamba/hybrid architectures

    # Scheduler behaviour
    port: int = 8780
    worker_base_port: int = 8001           # base port; DP instances use base+0, base+1, ...
    heartbeat_timeout: float = 120.0
    health_check_interval: float = 30.0
    node_acquire_interval: float = 15.0    # how often to try to grab new nodes
    conda_env: str = ""                    # conda env name on workers

    # Paths
    project_root: str = ""
    log_dir: str = "logs"

    @property
    def dp_per_node(self) -> int:
        """Number of data-parallel vLLM/SGLang instances per node."""
        return self.gpus_per_node // self.tensor_parallel_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SchedulerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json_file(cls, path: str) -> "SchedulerConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

