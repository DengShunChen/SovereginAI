#!/usr/bin/env python3
"""
氣象主權 AI — 國際學術標準基準評估（inline 版本）

直接呼叫 lm_eval API，繞過 mlx_lm.evaluate CLI，確保結果寫入 models/eval_standard_benchmarks.json。
前置：pip install lm-eval，需先 fuse 微調模型。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TASKS_DEFAULT = ["gsm8k", "mmlu_abstract_algebra"]
LIMIT_DEFAULT = 25


def run_eval(model_path: str, tasks: list[str], limit: int, output_dir: Path) -> dict | None:
    """直接呼叫 lm_eval.simple_evaluate + MLXLM，回傳 results。"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import lm_eval
    import mlx.core as mx
    from mlx_lm.evaluate import MLXLM, chat_template_fn, make_sampler

    mx.random.seed(123)
    mx.distributed.init()

    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
    lm = MLXLM(
        model_path,
        batch_size=4,
        use_chat_template=True,
        trust_remote_code=True,
        sampler=sampler,
    )
    MLXLM.apply_chat_template = chat_template_fn()

    print(f"  評估中: {model_path} ...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        fewshot_as_multiturn=False,
        apply_chat_template=lm.use_chat_template,
        num_fewshot=None,
        limit=limit,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123,
    )
    return results.get("results") if results else None


MODEL_CONFIGS = {
    "qwen": {
        "base": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "fused": "models/sovereign-weather-fused",
        "fuse_cmd": "python -m mlx_lm fuse --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path models/sovereign-weather-lora --save-path models/sovereign-weather-fused",
    },
    "llama": {
        "base": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "fused": "models/sovereign-weather-fused-llama",
        "fuse_cmd": "python -m mlx_lm fuse --model mlx-community/Llama-3.2-1B-Instruct-4bit --adapter-path models/sovereign-weather-lora-llama --save-path models/sovereign-weather-fused-llama",
    },
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=LIMIT_DEFAULT)
    parser.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    parser.add_argument("--model-type", choices=["qwen", "llama"], default="qwen")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_type]
    base = cfg["base"]
    fused = str(PROJECT_ROOT / cfg["fused"])
    out_dir = PROJECT_ROOT / "models"
    tasks = args.tasks
    limit = args.limit

    if not Path(fused).exists():
        print(f"請先 fuse（{args.model_type}）：")
        print(f"  {cfg['fuse_cmd']}")
        sys.exit(1)

    print("=== 標準基準評估（Base vs Finetuned）===")
    print(f"  模型類型：{args.model_type}")
    print(f"  基礎：{base}")
    print(f"  微調：{fused}")
    print(f"  任務：{tasks}，limit={limit}\n")

    results = {}
    print("[1/2] 評估基礎模型...")
    results["base"] = run_eval(base, tasks, limit, out_dir)
    print("[2/2] 評估微調模型...")
    results["finetuned"] = run_eval(fused, tasks, limit, out_dir)

    out_path = out_dir / ("eval_standard_benchmarks_llama.json" if args.model_type == "llama" else "eval_standard_benchmarks.json")
    payload = {
        "model_type": args.model_type,
        "base_model": base,
        "fused_path": fused,
        "tasks": tasks,
        "limit": limit,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n結果已儲存至：{out_path}")
    if results.get("base"):
        print("\nBase 結果：", json.dumps(results["base"], indent=2, ensure_ascii=False))
    if results.get("finetuned"):
        print("\nFinetuned 結果：", json.dumps(results["finetuned"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
