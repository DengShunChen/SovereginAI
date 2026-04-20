#!/usr/bin/env python3
"""
將關鍵術語 Q&A 強化語料（Llama chat 格式）合併至訓練集。
用於修正 Llama 基礎模型對「焚風」等術語的錯誤聯想。
執行：python scripts/prepare_training/add_term_qa_to_train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT = REPO_ROOT / "data" / "processed" / "for_training" / "term_qa_supplement.jsonl"
TRAIN_JSONL = REPO_ROOT / "data" / "processed" / "for_training" / "train.jsonl"
REPEAT = 5  # 每筆 Q&A 重複次數以強化學習


def format_llama_chat(user_msg: str, assistant_msg: str) -> str:
    """Llama 3.2 chat template 格式。"""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_msg}<|eot_id|>"
    )


def main() -> None:
    if not SUPPLEMENT.exists():
        print(f"錯誤：找不到 {SUPPLEMENT}", file=sys.stderr)
        sys.exit(1)

    supplement = []
    with open(SUPPLEMENT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                supplement.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not supplement:
        print("無強化語料", file=sys.stderr)
        sys.exit(0)

    # 載入現有 train.jsonl，排除先前已加入的 Q&A（Llama chat 格式）
    qa_prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    existing = []
    if TRAIN_JSONL.exists():
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    text = r.get("text", "")
                    if text.startswith(qa_prefix):
                        continue  # 跳過先前加入的 Q&A，避免重複
                    existing.append(r)
                except json.JSONDecodeError:
                    pass

    # 產生 chat 格式的 Q&A
    added = []
    for r in supplement:
        inst = r.get("instruction") or r.get("input") or ""
        out = r.get("output") or ""
        if not inst or not out:
            continue
        text = format_llama_chat(inst, out)
        for _ in range(REPEAT):
            added.append({"text": text})

    # 合併並寫回
    merged = existing + added
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已合併 {len(added)} 筆 Q&A 強化語料至 train.jsonl")
    print(f"訓練集總計：{len(merged)} 筆")


if __name__ == "__main__":
    main()
