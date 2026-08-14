"""
擷取中央氣象署鄉鎮天氣預報 F-D0047。
優先抓全臺資料集（091 未來 2 天、093 未來 1 週），寫入
data/corpus/weather/official/daily/ 為 JSONL，按鄉鎮 × 預報日拆筆。
需設定 CWA_API_AUTH_KEY。
"""
from __future__ import annotations

import os
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
    extract_forecasts_by_day,
    jsonl_path_for_month,
    ssl_verify,
    update_manifest,
)

BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
FILEAPI_BASE = "https://opendata.cwa.gov.tw/fileapi/v1/opendataapi"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "official" / "daily"
REQUEST_INTERVAL_SEC = 1.5
SOURCE_TAG = "F-D0047"

# 全臺一次取回，避免打 22 縣市 × 2
DATASETS = (
    ("F-D0047-093", "鄉鎮一週預報"),
    ("F-D0047-091", "鄉鎮二日預報"),
)


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
        timeout=120,
        verify=verify,
    )


def fetch_json(auth_key: str, dataset_id: str) -> dict | None:
    q = {"Authorization": auth_key, "format": "JSON"}
    verify = ssl_verify()
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


def main() -> None:
    auth_key = get_auth_key()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    jsonl_path = jsonl_path_for_month(OUTPUT_DIR, "official_township", now.year, now.month)
    manifest_path = OUTPUT_DIR / "manifest.json"

    total = 0
    dates_seen: list[str] = []
    for dataset_id, label in DATASETS:
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

        items = extract_forecasts_by_day(data, title_prefix=label, fallback_date=date_str)
        written = 0
        for title, content, rec_date in items:
            if not content or len(content) < 5:
                continue
            append_record(
                jsonl_path,
                {
                    "title": title[:200],
                    "content": content,
                    "date": rec_date,
                    "source": SOURCE_TAG,
                    "type": "township_forecast",
                },
            )
            dates_seen.append(rec_date)
            written += 1
        print(f"  {dataset_id}: {written} 筆")
        total += written
        time.sleep(REQUEST_INTERVAL_SEC)

    if total > 0:
        update_manifest(
            manifest_path,
            jsonl_path.name,
            date_start=min(dates_seen) if dates_seen else date_str,
            date_end=max(dates_seen) if dates_seen else date_str,
            record_count_delta=total,
        )
    print(f"已寫入 {total} 筆至 {jsonl_path}")


if __name__ == "__main__":
    main()
