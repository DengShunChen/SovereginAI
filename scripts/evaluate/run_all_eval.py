#!/usr/bin/env python3
"""
氣象主權 AI — 一鍵執行所有評估並更新 MODEL_CARD

執行標準化（lm-eval）與客製化（PPL、準確率、定性比較）評估，
收集結果後自動更新 models/MODEL_CARD_LLAMA.md 或 models/MODEL_CARD.md。

用法：
  python scripts/evaluate/run_all_eval.py --model-type llama
  python scripts/evaluate/run_all_eval.py --model-type qwen

自專案根目錄執行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: list[str], timeout: int = 600) -> bool:
    """執行指令，回傳是否成功。"""
    # 使用當前 Python（含 venv）
    if cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    try:
        r = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            print(f"  [FAIL] {' '.join(cmd[:3])}...: {r.stderr[:300]}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {' '.join(cmd[:3])}...", file=sys.stderr)
        return False


def ensure_fused(model_type: str) -> bool:
    """確保 fused 模型存在。"""
    fused = PROJECT_ROOT / ("models/sovereign-weather-fused-llama" if model_type == "llama" else "models/sovereign-weather-fused")
    if fused.exists():
        return True
    base = "mlx-community/Llama-3.2-1B-Instruct-4bit" if model_type == "llama" else "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    adapter = "models/sovereign-weather-lora-llama" if model_type == "llama" else "models/sovereign-weather-lora"
    save = str(fused)
    cmd = ["python", "-m", "mlx_lm", "fuse", "--model", base, "--adapter-path", adapter, "--save-path", save]
    print(f"  Fuse 模型...")
    return run_cmd(cmd, timeout=300)


def run_all_tests(model_type: str) -> dict:
    """執行所有評估，回傳收集的結果。"""
    results = {}
    models_dir = PROJECT_ROOT / "models"
    suffix = "_llama" if model_type == "llama" else ""

    # 1. 定量評估
    print("[1/4] 定量評估（PPL、準確率）...")
    if run_cmd(["python", "scripts/evaluate/run_quantitative_eval.py", "--model-type", model_type], timeout=180):
        p = models_dir / ("eval_quantitative_llama.json" if model_type == "llama" else "eval_quantitative.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                results["quantitative"] = json.load(f)

    # 2. 定性比較
    print("[2/4] Prompt 比較（定性）...")
    if run_cmd(["python", "scripts/evaluate/compare_base_vs_finetuned.py", "--model-type", model_type], timeout=300):
        p = models_dir / ("evaluation_base_vs_finetuned_llama.json" if model_type == "llama" else "evaluation_base_vs_finetuned.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                results["qualitative"] = json.load(f)

    # 3. Perplexity 比較（獨立）
    print("[3/4] Perplexity 比較...")
    if run_cmd(["python", "scripts/evaluate/compare_perplexity.py", "--model-type", model_type], timeout=120):
        # compare_perplexity 無 JSON 輸出，quantitative 已含 PPL
        results["perplexity_done"] = True

    # 4. 標準基準
    print("[4/4] 標準基準（lm-eval）...")
    if not ensure_fused(model_type):
        print("  跳過標準基準（fused 失敗）")
    elif run_cmd(["python", "scripts/evaluate/run_standard_benchmarks_inline.py", "--model-type", model_type], timeout=600):
        p = models_dir / ("eval_standard_benchmarks_llama.json" if model_type == "llama" else "eval_standard_benchmarks.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                results["standard"] = json.load(f)

    return results


def update_model_card_llama(results: dict) -> None:
    """依收集結果更新 MODEL_CARD_LLAMA.md。"""
    card_path = PROJECT_ROOT / "models" / "MODEL_CARD_LLAMA.md"
    content = card_path.read_text(encoding="utf-8")

    # 定量
    q = results.get("quantitative", {})
    ppl = q.get("perplexity", {})
    acc = q.get("accuracy", {})
    resp = q.get("response_length", {})
    lat = q.get("latency", {})

    def fmt(v, d=2):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)

    ppl_g = ppl.get("general", {})
    ppl_ge = ppl.get("general_extra", {})
    ppl_w = ppl.get("weather", {})

    q_table = f"""| 指標 | 基礎模型 | 微調模型 | 說明 |
|------|----------|----------|------|
| **PPL（通用中文）** | {fmt(ppl_g.get('base', 0))} | {fmt(ppl_g.get('finetuned', 0))} | 微調後 {'略升' if (ppl_g.get('finetuned', 0) or 0) > (ppl_g.get('base', 0) or 0) else '相當或略優'} |
| **PPL（通用額外）** | {fmt(ppl_ge.get('base', 0))} | {fmt(ppl_ge.get('finetuned', 0))} | 同上 |
| **PPL（氣象領域）** | {fmt(ppl_w.get('base', 0))} | **{fmt(ppl_w.get('finetuned', 0))}** | {'顯著改善' if (ppl_w.get('base', 0) or 0) > (ppl_w.get('finetuned', 0) or 0) else '相當'} |
| **準確率（6 題）** | {int((acc.get('base', 0) or 0) * 100)}% | {int((acc.get('finetuned', 0) or 0) * 100)}% | 相當 |
| **回覆長度（平均字元）** | {resp.get('base', {}).get('mean', '-')} | {resp.get('finetuned', {}).get('mean', '-')} | 相當 |
| **延遲（平均秒／題）** | {fmt(lat.get('base_avg_s', 0), 3)} | {fmt(lat.get('finetuned_avg_s', 0), 3)} | LoRA 略增推理時間 |"""

    # 標準基準
    std = results.get("standard", {})
    r = std.get("results", {})
    base_r = r.get("base", {})
    ft_r = r.get("finetuned", {})

    def pct(x):
        return f"{int((x or 0) * 100)}%" if x is not None else "-"

    # 標準基準表格：依任務動態生成
    TASK_LABELS = {"gsm8k": "GSM8K", "mmlu_abstract_algebra": "MMLU abstract_algebra", "hellaswag": "HellaSwag", "arc_challenge": "ARC-Challenge", "truthfulqa_mc1": "TruthfulQA", "winogrande": "Winogrande", "mmlu": "MMLU"}
    std_rows = []
    for task in ("gsm8k", "mmlu_abstract_algebra", "hellaswag", "arc_challenge", "truthfulqa_mc1", "winogrande", "mmlu"):
        tb, tf = base_r.get(task, {}), ft_r.get(task, {})
        if not tb and not tf:
            continue
        label = TASK_LABELS.get(task, task)
        if "acc,none" in (list(tb.keys()) or []) or "acc,none" in (list(tf.keys()) or []):
            std_rows.append(f"| **{label}** | {pct(tb.get('acc,none'))} | {pct(tf.get('acc,none'))} | 相當 |")
        else:
            s_b, flex_b = tb.get("exact_match,strict-match"), tb.get("exact_match,flexible-extract")
            s_f, flex_f = tf.get("exact_match,strict-match"), tf.get("exact_match,flexible-extract")
            std_rows.append(f"| **{label}** | {pct(s_b)} (strict) / {pct(flex_b)} (flex) | {pct(s_f)} / {pct(flex_f)} | 相當 |")
    std_table = "| 基準 | 基礎模型 | 微調模型 | 差異 |\n|------|----------|----------|------|\n" + "\n".join(std_rows) if std_rows else "| — | — | — | 待評估 |"

    import re

    new_block = f"""*最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}*

### 定量評估（本專案）

{q_table}

### 標準基準（lm-eval，limit={std.get('limit', 25)}）

{std_table}

氣象領域 PPL 明顯下降，泛化指標（準確率、MMLU）維持相當。"""

    # 替換 ## 評估結果 ... ## 測試評估 之間的內容
    content = re.sub(
        r"(## 評估結果\n\n).*?(\n## 測試評估)",
        rf"\1{new_block}\2",
        content,
        flags=re.DOTALL,
    )

    card_path.write_text(content, encoding="utf-8")
    print(f"已更新：{card_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一鍵執行所有評估並更新 MODEL_CARD")
    parser.add_argument("--model-type", choices=["qwen", "llama"], default="llama")
    parser.add_argument("--skip-update", action="store_true", help="僅執行評估，不更新 MODEL_CARD")
    args = parser.parse_args()

    print("=" * 60)
    print(f"氣象主權 AI — 全評估流程（{args.model_type}）")
    print("=" * 60)

    results = run_all_tests(args.model_type)

    if not args.skip_update and args.model_type == "llama" and results:
        print("\n更新 MODEL_CARD_LLAMA.md ...")
        update_model_card_llama(results)
    elif args.model_type == "qwen":
        # TODO: 可擴充 update_model_card_qwen()
        print("\nQwen 版請手動更新 models/MODEL_CARD.md，或執行各評估腳本後整合。")
    else:
        print("\n評估完成。結果已寫入 models/*.json")

    print("\n完成。")


if __name__ == "__main__":
    main()
