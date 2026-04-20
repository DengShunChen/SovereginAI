#!/usr/bin/env python3
"""
Perplexity 比較：基礎模型 vs 微調模型

在通用文本上計算 perplexity，若微調後 PPL 顯著升高，可能表示泛化能力退化。
使用非氣象領域的文本（如一般中文維基樣本）較能反映泛化情況。

用法：
  python scripts/evaluate/compare_perplexity.py
  python scripts/evaluate/compare_perplexity.py --text-file data/sample_general.txt

若未提供 --text-file，會使用內建的一般中文樣本。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# 內建的一般中文樣本（非氣象領域，用於泛化能力評估）
DEFAULT_GENERAL_TEXT = """
光合作用是植物、藻類和某些細菌利用光能將二氧化碳和水轉化為有機物並釋放氧氣的過程。
牛頓第一定律指出，物體在不受外力時保持靜止或勻速直線運動。
台灣位於東亞，首都是台北市，人口約兩千三百萬。
成語「守株待兔」比喻不主動努力而僥倖希望得到意外收穫。
程式設計中，變數是用來儲存資料的容器。
數學上，圓周率 π 約等於 3.14159。
地球繞太陽公轉一周約需 365.25 天。
人類呼吸需要吸入氧氣、呼出二氧化碳。
歷史記載，鄭成功曾在十七世紀渡海來台。
科學方法包括觀察、假設、實驗與結論。
"""


def load_text(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_GENERAL_TEXT.strip()


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    seq_len: int = 128,
    batch_size: int = 4,
) -> tuple[float, int]:
    """
    在給定文本上計算 perplexity。
    回傳 (perplexity, 有效 token 數)。
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < seq_len + 1:
        # 文本太短，用整個序列
        tokens = tokens + [tokenizer.eos_token_id or 0] * (seq_len + 1 - len(tokens))

    num_seqs = (len(tokens) - 1) // seq_len
    if num_seqs == 0:
        return float("nan"), 0

    all_losses = []
    for i in range(0, num_seqs * seq_len, seq_len * batch_size):
        batch_losses = []
        for b in range(batch_size):
            idx = i + b * seq_len
            if idx + seq_len + 1 >= len(tokens):
                break
            inp = tokens[idx : idx + seq_len]
            tgt = tokens[idx + 1 : idx + seq_len + 1]
            inp_mx = mx.array(inp)[None, :]
            tgt_mx = mx.array(tgt)[None, :]
            logits = model(inp_mx).astype(mx.float32)
            # logits[b,i,:] 預測下一個 token = tgt[b,i]
            loss = nn.losses.cross_entropy(logits, tgt_mx, reduction="none")
            mx.eval(loss)
            batch_losses.append(loss.flatten())
        if batch_losses:
            all_losses.append(mx.concatenate(batch_losses))

    if not all_losses:
        return float("nan"), 0

    concat = mx.concatenate(all_losses)
    mean_loss = concat.mean().item()
    ppl = math.exp(mean_loss)
    n_tokens = concat.size
    return ppl, n_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="比較基礎與微調模型的 perplexity")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        help="基礎模型",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA 適配器路徑",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="評估用文本檔（預設使用內建一般中文樣本）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="序列長度",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批次大小",
    )
    parser.add_argument(
        "--model-type",
        choices=["qwen", "llama"],
        default="qwen",
        help="模型類型（qwen=Qwen2.5，llama=Llama 3.2）",
    )

    args = parser.parse_args()

    from mlx_lm import load

    MODEL_CONFIGS = {"qwen": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "models/sovereign-weather-lora"), "llama": ("mlx-community/Llama-3.2-1B-Instruct-4bit", "models/sovereign-weather-lora-llama")}
    model_name, adapter_rel = MODEL_CONFIGS[args.model_type]
    model_name = args.model or model_name
    adapter_path = args.adapter_path or str(PROJECT_ROOT / adapter_rel)
    text_path = Path(args.text_file) if args.text_file else None
    text = load_text(text_path)

    print("=" * 60)
    print("Perplexity 比較：基礎 vs 微調（泛化能力指標）")
    print("=" * 60)
    print(f"  模型：{model_name}")
    print(f"  適配器：{adapter_path}")
    print(f"  評估文本長度：{len(text)} 字元")
    print()

    # 基礎模型
    print("載入基礎模型...")
    base_model, tokenizer = load(model_name)
    base_model.eval()
    print("計算基礎模型 perplexity...")
    base_ppl, n_tok = compute_perplexity(
        base_model, tokenizer, text,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print(f"  基礎模型 PPL: {base_ppl:.3f}（{n_tok} tokens）")
    print()

    # 微調模型
    print("載入微調模型（base + adapter）...")
    ft_model, _ = load(model_name, adapter_path=adapter_path)
    ft_model.eval()
    print("計算微調模型 perplexity...")
    ft_ppl, _ = compute_perplexity(
        ft_model, tokenizer, text,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print(f"  微調模型 PPL: {ft_ppl:.3f}")
    print()

    # 比較
    print("=" * 60)
    print("結果")
    print("=" * 60)
    diff = ft_ppl - base_ppl
    pct = (diff / base_ppl * 100) if base_ppl > 0 else 0
    print(f"  基礎模型 PPL: {base_ppl:.3f}")
    print(f"  微調模型 PPL: {ft_ppl:.3f}")
    print(f"  差異：{diff:+.3f}（{pct:+.1f}%）")
    print()
    if pct > 20:
        print("  ⚠ 微調模型 PPL 明顯升高 (>20%)，可能表示泛化能力有些許退化。")
        print("    建議檢查訓練資料多樣性、學習率或迭代次數。")
    elif pct > 5:
        print("  △ 微調模型 PPL 略升，屬常見情況，可接受。")
    else:
        print("  ✓ 微調模型與基礎模型 PPL 相當，泛化能力維持良好。")


if __name__ == "__main__":
    main()
