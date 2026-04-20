"""
擷取中央氣象署「一週天氣預報」F-C0032-003_006 或單點預報 F-C0032-001，
寫入 data/corpus/weather/official/daily/ 為 JSONL。
需設定環境變數 CWA_API_AUTH_KEY（或 .env）。
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
    from urllib3.exceptions import InsecureRequestWarning
except ImportError:
    print("請安裝：pip install requests python-dotenv", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / "config" / ".env")
load_dotenv(REPO_ROOT / ".env")

from scripts.cwa_fetch.utils import (
    append_record,
    jsonl_path_for_month,
    ssl_verify,
    update_manifest,
)

# 一週天氣預報（縣市／分區）
DATASET_WEEKLY = "F-C0032-003_006"
# 單點預報（可指定 locationName）
DATASET_SINGLE = "F-C0032-001"
BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
FILEAPI_BASE = "https://opendata.cwa.gov.tw/fileapi/v1/opendataapi"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "official" / "daily"
REQUEST_INTERVAL_SEC = 1.5


def get_auth_key() -> str:
    key = os.environ.get("CWA_API_AUTH_KEY", "").strip()
    if not key:
        print("請設定 CWA_API_AUTH_KEY（config/.env 或環境變數）", file=sys.stderr)
        sys.exit(1)
    return key


def _request(url: str, params: dict, verify: bool) -> requests.Response:
    if not verify:
        import urllib3
        urllib3.disable_warnings(InsecureRequestWarning)
    return requests.get(
        url,
        params=params,
        headers={"accept": "application/json"},
        timeout=30,
        verify=verify,
    )


def fetch_json(auth_key: str, dataset_id: str, **params: str) -> dict | None:
    """擷取單一資料集；若 rest 回 404 則改試 fileapi。回傳 None 表示無法取得。"""
    q = {"Authorization": auth_key, "format": "JSON", **params}
    verify = ssl_verify()

    # 1. rest/datastore
    url_rest = f"{BASE_URL}/{dataset_id}"
    for attempt in range(2):
        try:
            r = _request(url_rest, q, verify)
            if r.status_code == 200:
                return r.json()
            if r.status_code != 404:
                r.raise_for_status()
        except requests.RequestException:
            if attempt == 0:
                time.sleep(REQUEST_INTERVAL_SEC)
        break

    # 2. fileapi（一週預報 404 時備援）
    url_file = f"{FILEAPI_BASE}/{dataset_id}"
    for attempt in range(2):
        try:
            r = _request(url_file, q, verify)
            if r.status_code == 200:
                ct = (r.headers.get("Content-Type") or "").lower()
                if "json" in ct:
                    return r.json()
            if r.status_code != 404:
                r.raise_for_status()
        except requests.RequestException:
            if attempt == 0:
                time.sleep(REQUEST_INTERVAL_SEC)
        break

    return None


def extract_forecast_text(data: dict, source: str) -> list[tuple[str, str]]:
    """從預報 API 回傳抽出 (標題, 內容) 列表。"""
    out: list[tuple[str, str]] = []
    records = data.get("records") or data.get("result") or data

    def text_from_location(loc: dict) -> None:
        name = loc.get("locationName") or loc.get("location") or "未知"
        parts: list[str] = []
        for we in loc.get("weatherElement", []) or []:
            elem_name = we.get("elementName") or we.get("name") or ""
            for t in we.get("time", []) or []:
                start = t.get("startTime", "")
                end = t.get("endTime", "")
                val = t.get("parameter") or t.get("value") or t
                if isinstance(val, dict):
                    p = val.get("parameterName") or val.get("parameterValue") or val.get("value")
                else:
                    p = str(val)
                if p:
                    parts.append(f"{elem_name}: {p} ({start}~{end})")
        if parts:
            out.append((f"一週預報 {name}", "\n".join(parts)))

    if isinstance(records, dict) and "location" in records:
        for loc in records.get("location", []) or []:
            text_from_location(loc)
    elif isinstance(records, list):
        for r in records:
            if isinstance(r, dict):
                text_from_location(r)

    if not out:
        # 備援：搜尋長字串
        def collect(obj: dict | list, depth: int = 0) -> None:
            if depth > 8:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and len(v) > 30 and re.search(r"[\u4e00-\u9fff]", v):
                        out.append((k, v.strip()))
                    else:
                        collect(v, depth + 1)
            elif isinstance(obj, list):
                for v in obj:
                    collect(v, depth + 1)
        collect(data)

    return out


def main() -> None:
    auth_key = get_auth_key()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    year, month = now.year, now.month
    jsonl_path = jsonl_path_for_month(OUTPUT_DIR, "official_weekly", year, month)
    manifest_path = OUTPUT_DIR / "manifest.json"

    total = 0
    for dataset_id, source_label in [
        (DATASET_WEEKLY, "F-C0032-003_006"),
        (DATASET_SINGLE, "F-C0032-001"),
    ]:
        try:
            data = fetch_json(auth_key, dataset_id)
        except Exception as e:
            print(f"略過 {dataset_id}: {e}", file=sys.stderr)
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        if data is None:
            print(f"略過 {dataset_id}: rest 與 fileapi 皆無法取得", file=sys.stderr)
            time.sleep(REQUEST_INTERVAL_SEC)
            continue
        if data.get("success") is False:
            print(f"API {dataset_id} 回傳失敗", file=sys.stderr)
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        items = extract_forecast_text(data, source_label)
        for title, content in items:
            if not content or len(content) < 5:
                continue
            record = {
                "title": title[:200] if len(title) > 200 else title,
                "content": content,
                "date": date_str,
                "source": source_label,
                "type": "weekly_forecast",
            }
            append_record(jsonl_path, record)
        total += len([c for _, c in items if c and len(c) >= 5])
        time.sleep(REQUEST_INTERVAL_SEC)

    if total > 0:
        update_manifest(
            manifest_path,
            jsonl_path.name,
            date_start=date_str,
            date_end=date_str,
            record_count_delta=total,
        )
    print(f"已寫入 {total} 筆至 {jsonl_path}")


if __name__ == "__main__":
    main()
