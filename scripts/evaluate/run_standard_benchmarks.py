#!/usr/bin/env python3
"""
氣象主權 AI — 國際學術標準基準評估（Base vs Finetuned）

使用 lm-eval 對基礎模型與微調後模型執行 MMLU、GSM8K 等標準基準，
比較兩模型泛化能力差異。結果寫入 models/eval_standard_benchmarks.json。

前置：
  pip install lm-eval
  需先 fuse：python -m mlx_lm fuse --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path models/sovereign-weather-lora --save-path models/sovereign-weather-fused

用法：
  python scripts/evaluate/run_standard_benchmarks.py
  python scripts/evaluate/run_standard_benchmarks.py --limit 50 --tasks mmlu gsm8k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TASKS_DEFAULT = ["mmlu", "gsm8k", "arc_challenge", "hellaswag"]


def run_mlx_eval(model: str, tasks: list[str], limit: int, output_dir: Path) -> dict | None:
    """呼叫 mlx_lm.evaluate，解析輸出 json 檔。"""
    import importlib.metadata
    model_safe = model.replace("/", "_")
    parts = ["eval", model_safe, importlib.metadata.version("lm_eval")] + tasks
    fname = "_".join(parts)

    cmd = [
        sys.executable, "-m", "mlx_lm.evaluate",
        "--model", model,
        "--tasks", *tasks,
        "--limit", str(limit),
        "--output-dir", str(output_dir),
        "--batch-size", "4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"評估失敗 ({model}): {r.stderr[:500]}", file=sys.stderr)
        return None

    out_path = output_dir / fname
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--fused-path", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    fused = args.fused_path or str(PROJECT_ROOT / "models" / "sovereign-weather-fused")
    out_dir = Path(args.output) if args.output else PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(fused).exists():
        print("請先 fuse：python -m mlx_lm fuse --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path models/sovereign-weather-lora --save-path models/sovereign-weather-fused")
        sys.exit(1)

    print("=== 標準基準評估（Base vs Finetuned）===")
    print(f"  基礎：{args.model}")
    print(f"  微調：{fused}")
    print(f"  任務：{args.tasks}，limit={args.limit}\n")

    results = {}
    print("[1/2] 評估基礎模型...")
    results["base"] = run_mlx_eval(args.model, args.tasks, args.limit, out_dir)
    print("[2/2] 評估微調模型...")
    results["finetuned"] = run_mlx_eval(fused, args.tasks, args.limit, out_dir)

    out_path = out_dir / "eval_standard_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"base_model": args.model, "fused_path": fused, "tasks": args.tasks, "limit": args.limit, "results": results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n結果已儲存至：{out_path}")


if __name__ == "__main__":
    main()
