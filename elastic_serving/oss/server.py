import argparse
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from elastic_serving.oss.runtime import OSSEngineRuntime

DEFAULT_BLOCKED_SUBSTRINGS = ["huggingface", "browsecomp"]


class RunOneRequest(BaseModel):
    question: str
    qid: Optional[str] = None
    reasoning_effort: Optional[str] = None


def create_app(runtime: OSSEngineRuntime, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(
        title="OSS Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {"service": "oss", "model": model_name}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "oss",
                }
            ],
        }

    @app.post("/v1/oss/run_one")
    async def run_one(request: RunOneRequest):
        qid = request.qid or uuid.uuid4().hex
        result = await runtime.run_one(
            question=request.question,
            qid=qid,
            reasoning_effort=request.reasoning_effort,
        )
        return {"qid": qid, **result}

    return app


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Worker-local OSS service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--blocked-substrings",
        default=",".join(DEFAULT_BLOCKED_SUBSTRINGS),
        help="Comma-separated substrings to omit from browser search results.",
    )
    args = parser.parse_args()
    blocked_substrings = [
        value.strip()
        for value in args.blocked_substrings.split(",")
        if value.strip()
    ]

    runtime = OSSEngineRuntime(
        model_name_or_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        default_reasoning_effort=args.reasoning_effort,
        blocked_substrings=blocked_substrings,
    )
    app = create_app(runtime, model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
