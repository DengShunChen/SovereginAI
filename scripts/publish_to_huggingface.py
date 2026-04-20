#!/usr/bin/env python3
"""
將語料目錄下的 JSONL 發布到 Hugging Face Hub。

首次發布：掃描 data/corpus/weather 下所有 .jsonl，建立 Dataset 並 push。
後續增量：--incremental 會從 Hub 載入既有 dataset，合併本地新資料後再 push，
         以 (date, source, title) 去重，避免重複列。

使用前請先登入：
  huggingface-cli login
或設定環境變數 HF_TOKEN。

範例：
  # 首次發布（完整）
  python scripts/publish_to_huggingface.py --repo_id YOUR_USERNAME/sovereign-weather-corpus

  # 後續只推送新增／變更（增量）
  python scripts/publish_to_huggingface.py --repo_id YOUR_USERNAME/sovereign-weather-corpus --incremental

  # 若 Hub 上該 dataset 載入需要遠端程式碼（不建議任意 repo 使用）
  python scripts/publish_to_huggingface.py --repo_id ... --incremental --trust-remote-code

  # 私人 repo
  python scripts/publish_to_huggingface.py --repo_id YOUR_USERNAME/sovereign-weather-corpus --private
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DEFAULT = PROJECT_ROOT / "data" / "corpus" / "weather"

# 載入 .env 僅補齊「尚未設定」的變數，不覆寫既有環境（CI／shell 已 export 的 HF_TOKEN 優先）
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

# 停用 Xet 上傳，改走 LFS，避免部分環境出現 "Attempted to create a NULL object" panic
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def collect_jsonl_records(corpus_dir: Path) -> list[dict]:
    """掃描 corpus_dir 下所有 .jsonl，每行一個 JSON，回傳紀錄列表（每筆加上 file_source）。"""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return []

    records = []
    for p in sorted(corpus_dir.rglob("*.jsonl")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  [skip] {p}:{i} invalid json: {e}", file=sys.stderr)
                        continue
                    if not isinstance(obj, dict):
                        continue
                    # 相對路徑，方便辨識來源
                    try:
                        rel = p.relative_to(corpus_dir)
                    except ValueError:
                        rel = p.name
                    obj["file_source"] = str(rel)
                    records.append(obj)
        except OSError as e:
            print(f"  [skip] {p}: {e}", file=sys.stderr)

    return records


def dedupe_by_key(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """依 keys 去重，保留第一次出現的紀錄。"""
    seen: set[tuple] = set()
    out = []
    for r in records:
        t = tuple(r.get(k) for k in keys)
        if t in seen:
            continue
        seen.add(t)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將語料 JSONL 發布到 Hugging Face Hub，支援首次發布與增量 push。"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face 資料集 repo，例如 username/sovereign-weather-corpus",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=CORPUS_DEFAULT,
        help=f"語料根目錄（預設: {CORPUS_DEFAULT}）",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="建立/更新為私人 repo",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量模式：從 Hub 載入既有 dataset，合併本地資料後再 push（以 date, source, title 去重）",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="增量載入 Hub dataset 時允許 trust_remote_code（預設關閉，避免未審核 dataset 執行遠端程式）",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="自訂 commit 訊息（預設為「Full upload」或「Incremental update」）",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        print("請先安裝: pip install datasets huggingface_hub", file=sys.stderr)
        sys.exit(1)

    corpus_dir = Path(args.corpus_dir)
    local_records = collect_jsonl_records(corpus_dir)
    print(f"本地語料: {len(local_records)} 筆（來自 {corpus_dir}）")

    if args.incremental:
        try:
            existing = load_dataset(
                args.repo_id,
                trust_remote_code=args.trust_remote_code,
            )
            # load_dataset 回傳 DatasetDict（有 split）或單一 Dataset
            if hasattr(existing, "keys"):
                if "train" in existing:
                    ds_old = existing["train"]
                else:
                    first_split = next(iter(existing.keys()))
                    ds_old = existing[first_split]
            else:
                ds_old = existing
            old_records = [ds_old[i] for i in range(len(ds_old))]
            # 合併：舊 + 本地，再以 (date, source, title) 去重（保留第一次出現 = 舊的優先，新資料補在後）
            keys = ("date", "source", "title")
            combined = old_records + local_records
            merged = dedupe_by_key(combined, keys)
            # 若合併後筆數沒變多，可能是本地沒新資料
            new_count = len(merged) - len(old_records)
            print(f"既有: {len(old_records)} 筆，合併去重後: {len(merged)} 筆（+{new_count} 新）")
            records_to_push = merged
            default_msg = "Incremental update"
        except Exception as e:
            print(f"無法載入既有 dataset ({args.repo_id})，改為完整上傳: {e}", file=sys.stderr)
            records_to_push = local_records
            default_msg = "Full upload (incremental load failed)"
    else:
        records_to_push = local_records
        default_msg = "Full upload"

    if not records_to_push:
        print("沒有可推送的紀錄，結束。")
        sys.exit(0)

    commit_message = args.commit_message or default_msg
    ds = Dataset.from_list(records_to_push)
    ds.push_to_hub(args.repo_id, private=args.private, commit_message=commit_message)
    print(f"已 push 至 {args.repo_id}，共 {len(records_to_push)} 筆。commit: {commit_message}")


if __name__ == "__main__":
    main()
