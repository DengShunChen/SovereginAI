#!/usr/bin/env python3
"""
氣象主權 AI — 定量評估：基礎 vs 微調模型

整合多項定量指標，比較兩模型泛化能力：
- Perplexity（多種文本）
- 準確率（EM / contains）
- 回覆長度統計
- 生成速度（tokens/s）

用法：
  python scripts/evaluate/run_quantitative_eval.py
  python scripts/evaluate/run_quantitative_eval.py --output models/eval_quantitative.json

自專案根目錄執行。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# 定量評估用文本（非氣象）
TEXT_GENERAL = """光合作用是植物、藻類和某些細菌利用光能將二氧化碳和水轉化為有機物並釋放氧氣的過程。
牛頓第一定律指出，物體在不受外力時保持靜止或勻速直線運動。
台灣位於東亞，首都是台北市，人口約兩千三百萬。
成語「守株待兔」比喻不主動努力而僥倖希望得到意外收穫。
程式設計中，變數是用來儲存資料的容器。
數學上，圓周率 π 約等於 3.14159。
地球繞太陽公轉一周約需 365.25 天。
人類呼吸需要吸入氧氣、呼出二氧化碳。
歷史記載，鄭成功曾在十七世紀渡海來台。
科學方法包括觀察、假設、實驗與結論。"""

# 氣象領域文本（用於評估領域 PPL；與訓練集無重疊）
TEXT_WEATHER = """颱風警報發布時，請留意中央氣象署最新預報。
梅雨鋒面影響期間，山區易有局部大雨。
午後熱對流發展時，可能出現劇烈雷雨。
東北季風增強，北部及東北部氣溫下降。
太平洋高壓勢力強時，全台晴朗炎熱。"""

# 額外文本（較長，用於穩定 PPL 估計）
TEXT_EXTRA = """電腦科學中，演算法是解決問題的明確步驟序列。
物理學研究物質、能量與時空的基本性質。
化學反應涉及原子與分子的重新排列。
生物學探討生命的起源、演化與多樣性。
地理學研究地球表面的自然與人文現象。
歷史學透過史料分析過去的人類活動。
經濟學研究資源配置與人類選擇行為。
哲學探究存在、知識與價值的根本問題。"""


def extract_numbers(text: str) -> list[str]:
    """抽出文本中的數字（含小數）"""
    return re.findall(r"-?\d+\.?\d*", text)


def check_match(response: str, expected, match_type: str) -> bool:
    """檢查回覆是否符合預期"""
    if match_type == "contains_number":
        nums = extract_numbers(response)
        exp = str(expected).strip()
        return any(n == exp or n.lstrip("0") == exp.lstrip("0") for n in nums)
    if match_type == "contains_any":
        opts = [expected] if isinstance(expected, str) else expected
        return any(o in response for o in opts)
    if match_type == "contains_all":
        opts = [expected] if isinstance(expected, str) else expected
        return all(o in response for o in opts)
    if match_type == "exact":
        return response.strip() == str(expected).strip()
    return False


def format_chat_prompt(tokenizer, user_content: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_content


def compute_perplexity(model, tokenizer, text: str, seq_len: int = 64, batch_size: int = 4) -> float:
    import math
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < seq_len + 1:
        tokens = tokens + [tokenizer.eos_token_id or 0] * (seq_len + 1 - len(tokens))

    num_seqs = (len(tokens) - 1) // seq_len
    if num_seqs == 0:
        return float("nan")

    all_losses = []
    for i in range(0, min(num_seqs, 8) * seq_len, seq_len * batch_size):  # 最多 8 段
        for b in range(batch_size):
            idx = i + b * seq_len
            if idx + seq_len + 1 >= len(tokens):
                break
            inp = mx.array(tokens[idx : idx + seq_len])[None, :]
            tgt = mx.array(tokens[idx + 1 : idx + seq_len + 1])[None, :]
            logits = model(inp).astype(mx.float32)
            loss = nn.losses.cross_entropy(logits, tgt, reduction="none")
            mx.eval(loss)
            all_losses.append(loss.flatten())
    if not all_losses:
        return float("nan")
    concat = mx.concatenate(all_losses)
    return math.exp(concat.mean().item())


def run_eval(
    model_name: str,
    adapter_path: Path,
    prompts_path: Path,
    max_tokens: int = 128,
) -> dict:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    adapter_path = Path(adapter_path)
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])

    sampler = make_sampler(temp=0.0, top_p=1.0)

    print("載入基礎模型...")
    base_model, tokenizer = load(model_name)
    base_model.eval()

    print("載入微調模型...")
    ft_model, _ = load(model_name, adapter_path=str(adapter_path))
    ft_model.eval()

    metrics = {
        "model": model_name,
        "adapter_path": str(adapter_path),
        "perplexity": {},
        "accuracy": {},
        "response_length": {},
        "latency": {},
    }

    # 1. Perplexity（多種文本）
    print("\n[1/4] Perplexity 比較...")
    for name, text in [
        ("general", TEXT_GENERAL),
        ("general_extra", TEXT_EXTRA),
        ("weather", TEXT_WEATHER),
    ]:
        base_ppl = compute_perplexity(base_model, tokenizer, text)
        ft_ppl = compute_perplexity(ft_model, tokenizer, text)
        metrics["perplexity"][name] = {"base": round(base_ppl, 3), "finetuned": round(ft_ppl, 3)}
        print(f"  {name}: base={base_ppl:.3f}, ft={ft_ppl:.3f}")

    # 2. 準確率 + 回覆長度（單次遍歷全部 prompt）
    graded = [p for p in prompts if "expected" in p]
    print("\n[2/4] 準確率與回覆長度（全部 prompt）...")
    base_correct = 0
    ft_correct = 0
    base_lens = []
    ft_lens = []
    details = []
    for item in prompts:
        formatted = format_chat_prompt(tokenizer, item["prompt"])

        t0 = time.perf_counter()
        base_out = generate(base_model, tokenizer, formatted, max_tokens=max_tokens, sampler=sampler)
        base_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        ft_out = generate(ft_model, tokenizer, formatted, max_tokens=max_tokens, sampler=sampler)
        ft_time = time.perf_counter() - t0

        base_out_s = base_out.strip()
        ft_out_s = ft_out.strip()
        base_lens.append(len(base_out_s))
        ft_lens.append(len(ft_out_s))

        if "expected" in item:
            base_ok = check_match(base_out_s, item["expected"], item.get("match_type", "contains_any"))
            ft_ok = check_match(ft_out_s, item["expected"], item.get("match_type", "contains_any"))
            if base_ok:
                base_correct += 1
            if ft_ok:
                ft_correct += 1
            details.append({
                "id": item["id"],
                "base_correct": base_ok,
                "ft_correct": ft_ok,
                "base_len": len(base_out_s),
                "ft_len": len(ft_out_s),
                "base_time_s": round(base_time, 3),
                "ft_time_s": round(ft_time, 3),
            })

    if graded:
        metrics["accuracy"] = {
            "base": round(base_correct / len(graded), 3),
            "finetuned": round(ft_correct / len(graded), 3),
            "n_graded": len(graded),
            "details": details,
        }
        print(f"  準確率: base={base_correct}/{len(graded)}, ft={ft_correct}/{len(graded)}")

    metrics["response_length"] = {
        "base": {"mean": round(statistics.mean(base_lens), 1), "std": round(statistics.stdev(base_lens) if len(base_lens) > 1 else 0, 1)},
        "finetuned": {"mean": round(statistics.mean(ft_lens), 1), "std": round(statistics.stdev(ft_lens) if len(ft_lens) > 1 else 0, 1)},
        "n_prompts": len(prompts),
    }
    print(f"  回覆長度: base={metrics['response_length']['base']['mean']} chars, ft={metrics['response_length']['finetuned']['mean']} chars")

    # 3. 延遲（從 details 彙總；若有）
    if details:
        base_times = [d["base_time_s"] for d in details]
        ft_times = [d["ft_time_s"] for d in details]
        metrics["latency"] = {
            "base_avg_s": round(statistics.mean(base_times), 3),
            "finetuned_avg_s": round(statistics.mean(ft_times), 3),
            "n": len(details),
        }
        print(f"  延遲(avg): base={metrics['latency']['base_avg_s']}s, ft={metrics['latency']['finetuned_avg_s']}s")
    metrics["summary"] = _build_summary(metrics)
    return metrics


def _build_summary(m: dict) -> dict:
    s = {}
    if m.get("perplexity"):
        p = m["perplexity"].get("general", {})
        if p:
            diff = p.get("finetuned", 0) - p.get("base", 1)
            pct = (diff / p.get("base", 1) * 100) if p.get("base") else 0
            s["ppl_diff_pct"] = round(pct, 1)
            s["ppl_ok"] = pct < 20
    if m.get("accuracy"):
        a = m["accuracy"]
        s["acc_base"] = a.get("base")
        s["acc_ft"] = a.get("finetuned")
        s["acc_diff"] = round((a.get("finetuned", 0) - a.get("base", 0)) * 100, 1)
    return s


MODEL_CONFIGS = {
    "qwen": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "models/sovereign-weather-lora"),
    "llama": ("mlx-community/Llama-3.2-1B-Instruct-4bit", "models/sovereign-weather-lora-llama"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="定量評估：基礎 vs 微調")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--model-type", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)

    args = parser.parse_args()
    scripts_dir = Path(__file__).resolve().parent
    prompts_path = Path(args.prompts) if args.prompts else scripts_dir / "quantitative_prompts.json"
    if args.model and args.adapter_path:
        model_name, adapter_path = args.model, Path(args.adapter_path)
    else:
        model_name, adapter_rel = MODEL_CONFIGS[args.model_type]
        model_name = args.model or model_name
        adapter_path = args.adapter_path or PROJECT_ROOT / adapter_rel
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "models" / ("eval_quantitative_llama.json" if args.model_type == "llama" else "eval_quantitative.json")

    if not prompts_path.exists():
        print(f"錯誤：找不到 {prompts_path}")
        sys.exit(1)

    print("=" * 60)
    print("氣象主權 AI — 定量評估（基礎 vs 微調）")
    print("=" * 60)

    metrics = run_eval(
        model_name=model_name,
        adapter_path=adapter_path,
        prompts_path=prompts_path,
        max_tokens=args.max_tokens,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n[4/4] 評估完成")
    print("=" * 60)
    print("定量摘要")
    print("=" * 60)
    if metrics.get("perplexity"):
        for k, v in metrics["perplexity"].items():
            print(f"  PPL ({k}): base={v['base']}, ft={v['finetuned']}")
    if metrics.get("accuracy"):
        a = metrics["accuracy"]
        print(f"  準確率: base={a['base']:.1%}, ft={a['finetuned']:.1%} (n={a['n_graded']})")
    if metrics.get("response_length"):
        r = metrics["response_length"]
        print(f"  回覆長度(平均字元): base={r['base']['mean']}, ft={r['finetuned']['mean']}")
    print(f"\n結果已儲存至：{out_path}")


if __name__ == "__main__":
    main()
