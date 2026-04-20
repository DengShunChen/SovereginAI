#!/usr/bin/env python3
"""
檢查 data/processed/for_training/ 的筆數與一筆範例，確認訓練用資料已就緒。
自專案根目錄執行：python scripts/check_training_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FOR_TRAINING = REPO_ROOT / "data" / "processed" / "for_training"


def main() -> None:
    if not FOR_TRAINING.exists():
        print(f"目錄不存在: {FOR_TRAINING}", file=sys.stderr)
        print("請先執行: ./scripts/run_collect_and_prepare.sh", file=sys.stderr)
        sys.exit(1)

    schema_path = FOR_TRAINING / "schema.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        print("schema:", schema.get("format", ""), schema.get("schema", ""))

    total = 0
    for name in ("train", "valid", "test"):
        path = FOR_TRAINING / f"{name}.jsonl"
        if not path.exists():
            print(f"  {name}.jsonl: 無")
            continue
        count = 0
        first = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if first is None:
                    try:
                        first = json.loads(line)
                    except json.JSONDecodeError:
                        pass
        total += count
        print(f"  {name}.jsonl: {count} 筆")
        if first:
            if "text" in first:
                sample = first["text"][:120] + "..." if len(first.get("text", "")) > 120 else first.get("text", "")
            else:
                sample = json.dumps(first, ensure_ascii=False)[:120] + "..."
            print(f"    範例: {sample}")

    print(f"總計: {total} 筆")
    if total == 0:
        print("尚無訓練資料，請先執行擷取與前處理。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
