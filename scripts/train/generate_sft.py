#!/usr/bin/env python3
"""
CUDA 推理：基礎模型 + LoRA 適配器。
自專案根目錄：
  python scripts/train/generate_sft.py --prompt "請說明焚風現象"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "sovereign-weather-lora-gemma4-12b"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA CUDA 推理")
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    p.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--prompt", default="請說明焚風現象")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--base-only", action="store_true", help="不載入適配器，只跑基礎模型")
    return p.parse_args()


def main() -> None:
    _load_dotenv()
    args = parse_args()
    if not torch.cuda.is_available():
        print("錯誤：找不到 CUDA", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if not args.base_only:
        if not args.adapter_path.exists():
            print(f"錯誤：找不到適配器 {args.adapter_path}", file=sys.stderr)
            sys.exit(1)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path))

    model.eval()
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # chat template 已含 bos，不可再加 special tokens
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temp > 0,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    if args.temp > 0:
        gen_kw["temperature"] = args.temp
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
