#!/usr/bin/env python3
"""
氣象主權 AI — 基礎模型 vs 微調模型 泛化能力比較

比較 Base 模型與 LoRA 微調後模型的基礎能力，確認微調未造成泛化能力退化。
使用非氣象領域的 prompt 測試，避免直接評估訓練領域表現。

用法：
  python scripts/evaluate/compare_base_vs_finetuned.py
  python scripts/evaluate/compare_base_vs_finetuned.py --model mlx-community/Qwen2.5-3B-Instruct-4bit

自專案根目錄執行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_prompts(prompts_path: Path) -> list[dict]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])


def format_chat_prompt(tokenizer, user_content: str) -> str:
    """使用 chat template 格式化 prompt（適用 Qwen 等 Instruct 模型）"""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_content


def run_comparison(
    model_name: str,
    adapter_path: str | Path,
    prompts_path: Path,
    max_tokens: int = 256,
    temp: float = 0.0,
    output_path: Path | None = None,
) -> None:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    adapter_path = Path(adapter_path)
    sampler = make_sampler(temp=temp, top_p=1.0)
    if not adapter_path.exists():
        print(f"錯誤：找不到適配器目錄 {adapter_path}")
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    if not prompts:
        print("錯誤：没有可用的測試 prompt")
        sys.exit(1)

    print("=" * 60)
    print("氣象主權 AI — 基礎 vs 微調 泛化能力比較")
    print("=" * 60)
    print(f"  基礎模型：{model_name}")
    print(f"  微調適配器：{adapter_path}")
    print(f"  測試 prompt 數：{len(prompts)}")
    print(f"  溫度：{temp}（0=確定性輸出）")
    print()

    # 載入基礎模型
    print("載入基礎模型...")
    base_model, tokenizer = load(model_name)
    base_model.eval()

    # 載入微調模型（base + adapter）
    print("載入微調模型（base + LoRA adapter）...")
    ft_model, _ = load(model_name, adapter_path=str(adapter_path))
    ft_model.eval()

    results = []

    for i, item in enumerate(prompts):
        prompt_text = item.get("prompt", item.get("content", ""))
        prompt_id = item.get("id", f"prompt_{i}")
        category = item.get("category", "unknown")

        formatted = format_chat_prompt(tokenizer, prompt_text)

        print(f"\n{'─' * 60}")
        print(f"[{prompt_id}] {category}")
        print(f"  Q: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
        print()

        # 基礎模型
        print("  【基礎模型】")
        base_out = generate(
            base_model,
            tokenizer,
            formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        base_out = base_out.strip()
        print(f"  {base_out[:500]}{'...' if len(base_out) > 500 else ''}")
        print()

        # 微調模型
        print("  【微調模型】")
        ft_out = generate(
            ft_model,
            tokenizer,
            formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        ft_out = ft_out.strip()
        print(f"  {ft_out[:500]}{'...' if len(ft_out) > 500 else ''}")

        results.append(
            {
                "id": prompt_id,
                "category": category,
                "prompt": prompt_text,
                "base_response": base_out,
                "finetuned_response": ft_out,
            }
        )

    # 輸出摘要
    print("\n" + "=" * 60)
    print("比較完成")
    print("=" * 60)
    print("若兩模型輸出在非氣象議題上皆合理、無明顯退化，")
    print("可視為微調未嚴重損害泛化能力。")
    print()
    print("建議：人工檢查各 prompt 的輸出品質，確認微調模型")
    print("在一般知識、推理等任務上與基礎模型相當。")

    # 儲存結果
    out_path = output_path or PROJECT_ROOT / "models" / "evaluation_base_vs_finetuned.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "adapter_path": str(adapter_path),
                "prompts_count": len(prompts),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n詳細結果已儲存至：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比較基礎模型與微調模型的泛化能力"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        help="基礎模型名稱或路徑",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA 適配器路徑（預設依 --model-type）",
    )
    parser.add_argument(
        "--model-type",
        choices=["qwen", "llama"],
        default="qwen",
        help="模型類型（qwen=Qwen2.5，llama=Llama 3.2）",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="測試 prompt JSON 路徑（預設：本目錄 generalization_prompts.json）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="每個回覆最大 token 數",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="取樣溫度（0=確定性）",
    )

    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    prompts_path = Path(args.prompts) if args.prompts else scripts_dir / "generalization_prompts.json"
    MODEL_CONFIGS = {"qwen": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "models/sovereign-weather-lora"), "llama": ("mlx-community/Llama-3.2-1B-Instruct-4bit", "models/sovereign-weather-lora-llama")}
    model_name, adapter_rel = MODEL_CONFIGS[args.model_type]
    adapter_path = args.adapter_path or PROJECT_ROOT / adapter_rel
    model_name = args.model or model_name

    if not prompts_path.exists():
        print(f"錯誤：找不到 prompt 檔 {prompts_path}")
        sys.exit(1)

    out_path = PROJECT_ROOT / "models" / ("evaluation_base_vs_finetuned_llama.json" if args.model_type == "llama" else "evaluation_base_vs_finetuned.json")
    run_comparison(
        model_name=model_name,
        adapter_path=adapter_path,
        prompts_path=prompts_path,
        max_tokens=args.max_tokens,
        temp=args.temp,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
