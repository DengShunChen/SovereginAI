"""
擷取中央氣象署臉書「報天氣」粉絲專頁公開貼文，寫入 data/corpus/weather/social/facebook_cwa/。
需設定 FB_ACCESS_TOKEN（config/.env）。取得方式見 README。
僅供台灣氣象語料使用。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(REPO_ROOT / "config" / ".env")
    load_dotenv(REPO_ROOT / ".env")

import requests

from scripts.cwa_fetch.utils import append_record, ensure_dir, update_manifest

# 報天氣 - 中央氣象署 粉絲專頁（依網路查證：https://www.facebook.com/cwa.weather）
FB_PAGE_ID = "cwa.weather"
GRAPH_BASE = "https://graph.facebook.com/v18.0"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "social" / "facebook_cwa"
SOURCE_LABEL = "facebook_cwa"
TYPE_LABEL = "official_social"


def get_token() -> str:
    token = os.environ.get("FB_ACCESS_TOKEN", "").strip()
    if not token:
        print(
            "請設定 FB_ACCESS_TOKEN（config/.env）。\n"
            "取得方式：Facebook 開發者後台建立應用程式，取得 Access Token；\n"
            "或使用 Graph API Explorer 取得短期 Token。詳見 data/corpus/weather/social/README.md",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def fetch_page_posts(access_token: str, limit: int = 25) -> list[dict]:
    url = f"{GRAPH_BASE}/{FB_PAGE_ID}/posts"
    params = {
        "access_token": access_token,
        "fields": "id,message,created_time,full_picture",
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("data") or []


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="擷取臉書報天氣粉絲專頁公開貼文")
    ap.add_argument("--limit", type=int, default=25, help="最多擷取貼文數")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="輸出目錄")
    args = ap.parse_args()

    token = get_token()
    ensure_dir(args.out_dir)

    try:
        posts = fetch_page_posts(token, limit=args.limit)
    except Exception as e:
        print(f"擷取失敗: {e}", file=sys.stderr)
        sys.exit(1)

    prefix = "facebook_cwa"
    year, month = datetime.now().year, datetime.now().month
    jsonl_path = args.out_dir / f"{prefix}_{year:04d}-{month:02d}.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    total = 0
    dates_written: list[str] = []

    for post in posts:
        msg = (post.get("message") or "").strip()
        if len(msg) < 10:
            continue
        created = post.get("created_time") or ""
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
        dates_written.append(date_str)
        title = msg[:80] + "..." if len(msg) > 80 else msg
        record = {
            "title": title[:200],
            "content": msg,
            "date": date_str,
            "source": SOURCE_LABEL,
            "type": TYPE_LABEL,
        }
        append_record(jsonl_path, record)
        total += 1

    if total > 0 and dates_written:
        update_manifest(
            manifest_path,
            jsonl_path.name,
            date_start=min(dates_written),
            date_end=max(dates_written),
            record_count_delta=total,
        )
    print(f"已寫入 {total} 筆至 {jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
