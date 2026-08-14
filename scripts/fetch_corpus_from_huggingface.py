#!/usr/bin/env python3
"""
從 Hugging Face Hub 回拉語料，依 file_source 還原到 data/corpus/weather/。

與 publish_to_huggingface.py 對稱：Hub 上每筆的 file_source 是相對路徑，
寫回本地時剝除該欄位，並與既有本地 JSONL 以 content hash 合併去重
（本地筆數優先保留）。

範例：
  python scripts/fetch_corpus_from_huggingface.py --repo_id dschen/sovereign-weather-corpus
  python scripts/fetch_corpus_from_huggingface.py --repo_id dschen/sovereign-weather-corpus --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DEFAULT = PROJECT_ROOT / "data" / "corpus" / "weather"
LOCAL_SCHEMA = ("title", "content", "date", "source", "type")

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def content_hash(record: dict) -> str:
    c = record.get("content") or record.get("text") or ""
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def safe_relpath(file_source: str, corpus_dir: Path) -> Path | None:
    """拒絕絕對路徑與 ..，確保目標落在 corpus_dir 內。"""
    if not file_source or not isinstance(file_source, str):
        return None
    raw = file_source.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("~"):
        return None
    parts = Path(raw).parts
    if any(p in (".", "..") for p in parts):
        return None
    if not raw.endswith(".jsonl"):
        return None
    target = (corpus_dir / raw).resolve()
    try:
        target.relative_to(corpus_dir.resolve())
    except ValueError:
        return None
    return target


def load_local_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [skip] {path}:{i} invalid json: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def strip_to_local_schema(record: dict) -> dict:
    return {k: record.get(k) for k in LOCAL_SCHEMA}


def merge_by_content_hash(local: list[dict], remote: list[dict]) -> tuple[list[dict], int]:
    """本地優先；回傳 (合併結果, 新增筆數)。"""
    seen: set[str] = set()
    out: list[dict] = []
    for r in local:
        h = content_hash(r)
        if h in seen:
            continue
        seen.add(h)
        out.append(strip_to_local_schema(r))
    added = 0
    for r in remote:
        h = content_hash(r)
        if h in seen:
            continue
        seen.add(h)
        out.append(strip_to_local_schema(r))
        added += 1
    return out, added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="從 Hugging Face Hub 回拉語料，依 file_source 還原本地 JSONL。"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="dschen/sovereign-weather-corpus",
        help="Hugging Face 資料集 repo",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=CORPUS_DEFAULT,
        help=f"語料根目錄（預設: {CORPUS_DEFAULT}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只統計不寫檔",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="載入 Hub dataset 時允許 trust_remote_code（預設關閉）",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("請先安裝: pip install datasets huggingface_hub", file=sys.stderr)
        sys.exit(1)

    corpus_dir = Path(args.corpus_dir)
    print(f"載入 {args.repo_id} ...")
    existing = load_dataset(args.repo_id, trust_remote_code=args.trust_remote_code)
    if hasattr(existing, "keys"):
        if "train" in existing:
            ds = existing["train"]
        else:
            ds = existing[next(iter(existing.keys()))]
    else:
        ds = existing

    by_file: dict[str, list[dict]] = defaultdict(list)
    skipped_path = 0
    missing_source = 0
    for i in range(len(ds)):
        row = dict(ds[i])
        fs = row.get("file_source")
        if not fs:
            missing_source += 1
            continue
        if safe_relpath(str(fs), corpus_dir) is None:
            print(f"  [skip] unsafe file_source: {fs!r}", file=sys.stderr)
            skipped_path += 1
            continue
        by_file[str(fs)].append(row)

    print(
        f"Hub: {len(ds)} 筆 → {len(by_file)} 個檔案"
        f"（缺 file_source {missing_source}，路徑拒絕 {skipped_path}）"
    )

    total_added = 0
    total_written = 0
    files_touched = 0
    for rel, remote_rows in sorted(by_file.items()):
        target = safe_relpath(rel, corpus_dir)
        assert target is not None
        local_rows = load_local_jsonl(target)
        merged, added = merge_by_content_hash(local_rows, remote_rows)
        total_added += added
        if added == 0 and local_rows:
            print(f"  {rel}: 本地 {len(local_rows)}，無新增")
            continue
        files_touched += 1
        total_written += len(merged)
        print(
            f"  {rel}: 本地 {len(local_rows)} + Hub {len(remote_rows)}"
            f" → {len(merged)}（+{added}）"
        )
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    verb = "將寫入" if args.dry_run else "已寫入"
    print(
        f"{'[dry-run] ' if args.dry_run else ''}"
        f"{verb} {files_touched} 個檔案，新增 {total_added} 筆"
        f"（合併後合計列數 {total_written}）"
    )


if __name__ == "__main__":
    main()
