import argparse
import asyncio
import hashlib
import json
import os
import time
import traceback

import httpx
import datasets

def load_dataset_rows(args):
    dataset_name = args.dataset
    if not dataset_name:
        raise ValueError("--dataset is required.")

    print(f"Loading dataset: {dataset_name} split={args.split}")
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        ds = datasets.load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split=args.split)
    print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    if "question" in ds.column_names:
        q_col = "question"
    elif "query" in ds.column_names:
        q_col = "query"
    else:
        raise ValueError("Dataset must contain either a 'question' or 'query' column.")
    has_id = "id" in ds.column_names

    rows = []
    for idx, row in enumerate(ds):
        item = dict(row)
        question = str(item[q_col])
        qid = str(item["id"]) if has_id else hashlib.md5(question.encode()).hexdigest()[:8]
        item["qid"] = qid
        item["question"] = question
        if "answer" in item and "reference_answer" not in item:
            item["reference_answer"] = item["answer"]
        rows.append(item)

    return dataset_name, rows


async def wait_for_workers(base_url: str):
    async with httpx.AsyncClient() as tmp:
        for _ in range(120):
            try:
                resp = await tmp.get(f"{base_url}/cluster_status", timeout=5)
                if resp.status_code == 200 and resp.json().get("ready_workers", 0) > 0:
                    print(f"Cluster: {resp.json()['ready_workers']} workers ready")
                    return
            except Exception:
                pass
            try:
                resp = await tmp.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    print("Direct OSS worker ready")
                    return
            except Exception:
                pass
            try:
                resp = await tmp.get(f"{base_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    print("Direct OSS worker ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(10)
    raise RuntimeError("Timed out waiting for workers.")


async def main_async(args):
    scheduler_url = args.scheduler_url.rstrip("/")
    endpoint = f"{scheduler_url}/v1/oss/run_one"
    dataset_name, data = load_dataset_rows(args)

    await wait_for_workers(scheduler_url)

    node_rank = int(os.getenv("RANK", 0))
    node_size = int(os.getenv("WORLD_SIZE", 1))
    my_items = data[node_rank::node_size]

    os.makedirs(args.output_dir, exist_ok=True)
    shard_name = "trajectories.jsonl" if node_size == 1 else f"node_{node_rank}.jsonl"
    shard_path = os.path.join(args.output_dir, shard_name)

    completed = {}
    if args.resume and os.path.exists(shard_path):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed[(str(rec["qid"]), int(rec.get("traj_idx", 0)))] = rec
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} trajectories done")

    tasks_to_process = []
    for item in my_items:
        for traj_idx in range(args.num_trajectories):
            key = (str(item["qid"]), traj_idx)
            if key not in completed:
                tasks_to_process.append((item, traj_idx))

    if not tasks_to_process:
        print("Nothing to do.")
        return

    http = httpx.AsyncClient(timeout=1800)
    sem = asyncio.Semaphore(args.concurrency)
    total = len(tasks_to_process)
    metadata = {
        "dataset": dataset_name,
        "split": args.split,
        "num_questions": len(data),
        "num_trajectories": args.num_trajectories,
        "total_trajectories": len(data) * args.num_trajectories,
        "assigned_questions": len(my_items),
        "rank": node_rank,
        "world_size": node_size,
        "reasoning_effort": args.reasoning_effort,
    }
    config_name = "run_oss_config.json" if node_size == 1 else f"run_oss_config_node_{node_rank}.json"
    with open(os.path.join(args.output_dir, config_name), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def process_item(item, traj_idx):
        async with sem:
            qid = str(item["qid"])
            request_qid = qid if args.num_trajectories == 1 else f"{qid}__traj{traj_idx}"
            question = item["question"]
            error_msg = None
            attempt = 0
            t0 = time.time()
            while attempt < args.max_retries:
                attempt += 1
                try:
                    resp = await http.post(
                        endpoint,
                        json={
                            "question": question,
                            "qid": request_qid,
                            "reasoning_effort": args.reasoning_effort,
                        },
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    rec = item.copy()
                    rec.update(
                        {
                            "qid": qid,
                            "traj_idx": traj_idx,
                            "request_qid": payload.get("qid", request_qid),
                            "messages": payload["messages"],
                            "searched_urls": payload.get("searched_urls", []),
                            "search_results": payload.get("search_results", []),
                            "latency_s": time.time() - t0,
                            "error": None,
                            "attempts": attempt,
                            "status": "success",
                        }
                    )
                    return rec
                except Exception:
                    error_msg = traceback.format_exc()

            rec = item.copy()
            rec.update(
                {
                    "qid": qid,
                    "traj_idx": traj_idx,
                    "request_qid": request_qid,
                    "messages": [],
                    "searched_urls": [],
                    "search_results": [],
                    "latency_s": 0.0,
                    "error": error_msg,
                    "attempts": attempt,
                    "status": "fail",
                }
            )
            return rec

    tasks = [asyncio.create_task(process_item(item, traj_idx)) for item, traj_idx in tasks_to_process]
    with open(shard_path, "a", encoding="utf-8") as writer:
        completed_count = 0
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            completed_count += 1
            if completed_count == 1 or completed_count % 10 == 0 or completed_count == total:
                print(f"rank {node_rank}: {completed_count}/{total} complete")
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.flush()

    await http.aclose()


def main():
    parser = argparse.ArgumentParser(description="Run OSS through elastic-inference")
    parser.add_argument("--scheduler-url", default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--dataset", default=None,
                        help="HF dataset name or local .json/.jsonl file")
    parser.add_argument("--split", default="normal")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", "--max-concurrency", dest="concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
