"""
警特報語料：若平台有警特報資料集則直接擷取；
目前從 F-C0044-001 天氣概況內容中解析含「警報」「特報」的段落，寫入 official/alerts/。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("請安裝：pip install requests python-dotenv", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / "config" / ".env")
load_dotenv(REPO_ROOT / ".env")

from scripts.cwa_fetch.utils import append_record, jsonl_path_for_month, update_manifest
from scripts.cwa_fetch.fetch_weather_summary import get_auth_key, fetch_json

DATASET_ID = "F-C0044-001"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "official" / "alerts"
REQUEST_INTERVAL_SEC = 1.5


def extract_alert_paragraphs(data: dict) -> list[str]:
    """從 API 回傳中抽出含警報/特報的段落。"""
    out: list[str] = []
    raw = json.dumps(data, ensure_ascii=False)

    # 依句或段切開，保留含關鍵字者
    for part in re.split(r"[\n。；]", raw):
        part = part.strip()
        if not part or len(part) < 10:
            continue
        if "特報" in part or "警報" in part or "大雨" in part or "豪雨" in part or "低溫" in part or "強風" in part:
            # 移除 JSON 雜訊
            part = re.sub(r'"[^"]*"\s*:', " ", part)
            part = re.sub(r"\s+", " ", part).strip()
            if len(part) >= 10:
                out.append(part)
    return out


def main() -> None:
    auth_key = get_auth_key()
    try:
        data, _ = fetch_json(auth_key)
    except Exception as e:
        print(f"無法取得 API 資料（網路或伺服器錯誤）: {e}", file=sys.stderr)
        sys.exit(0)  # 不視為致命錯誤，稍後可重試
    if data.get("success") is False and "_parsed_text" not in data:
        print("API 回傳失敗", file=sys.stderr)
        sys.exit(0)

    paragraphs = extract_alert_paragraphs(data)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    year, month = now.year, now.month
    jsonl_path = jsonl_path_for_month(OUTPUT_DIR, "official_alerts", year, month)
    manifest_path = OUTPUT_DIR / "manifest.json"

    count = 0
    for i, content in enumerate(paragraphs):
        record = {
            "title": f"警特報 {date_str} #{i+1}",
            "content": content,
            "date": date_str,
            "source": DATASET_ID,
            "type": "alert",
        }
        append_record(jsonl_path, record)
        count += 1

    if count > 0:
        update_manifest(manifest_path, jsonl_path.name, date_start=date_str, date_end=date_str, record_count_delta=count)

    print(f"已寫入 {count} 筆警特報至 {jsonl_path}")


if __name__ == "__main__":
    main()
