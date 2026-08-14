#!/usr/bin/env python3
"""
將關鍵術語 Q&A 強化語料合併至訓練集。
輸出結構化 instruction/output，由訓練腳本套用各模型 chat template。
執行：python scripts/prepare_training/add_term_qa_to_train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT = REPO_ROOT / "data" / "processed" / "for_training" / "term_qa_supplement.jsonl"
TRAIN_JSONL = REPO_ROOT / "data" / "processed" / "for_training" / "train.jsonl"
SOURCE_TAG = "term_qa_supplement"
REPEAT = 5  # 每筆 Q&A 重複次數以強化學習


def main() -> None:
    if not SUPPLEMENT.exists():
        print(f"略過：找不到 {SUPPLEMENT}", file=sys.stderr)
        sys.exit(0)

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

    existing = []
    if TRAIN_JSONL.exists():
        with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("source") == SOURCE_TAG:
                        continue
                    existing.append(r)
                except json.JSONDecodeError:
                    pass

    added = []
    for r in supplement:
        inst = r.get("instruction") or r.get("input") or ""
        out = r.get("output") or ""
        if not inst or not out:
            continue
        row = {
            "instruction": inst,
            "output": out,
            "source": SOURCE_TAG,
        }
        for _ in range(REPEAT):
            added.append(row)

    merged = existing + added
    TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已合併 {len(added)} 筆 Q&A 強化語料至 train.jsonl")
    print(f"訓練集總計：{len(merged)} 筆")


if __name__ == "__main__":
    main()
