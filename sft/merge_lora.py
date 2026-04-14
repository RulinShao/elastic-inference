#!/usr/bin/env python3
"""Merge LoRA adapter with base model into a standalone checkpoint."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LLaMA-Factory", "src"))


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base-model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving to {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    import shutil
    from pathlib import Path
    base_path = Path(args.base_model)
    if not base_path.exists():
        from huggingface_hub import snapshot_download
        base_path = Path(snapshot_download(args.base_model, allow_patterns=["config.json"]))
    orig_config = base_path / "config.json"
    if orig_config.exists():
        shutil.copy(orig_config, os.path.join(args.output, "config.json"))
        print("Copied original config.json to preserve model architecture metadata.")
    print("Done!")


if __name__ == "__main__":
    main()
