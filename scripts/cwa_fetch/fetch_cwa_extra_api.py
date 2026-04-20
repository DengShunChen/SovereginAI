"""
擷取 CWA 開放資料 API 額外資料集（W-C 警特報、E-A 地震、F-A 潮汐、A-B 天文等），
寫入 data/corpus/weather/official/ 對應目錄。
需設定 CWA_API_AUTH_KEY。
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

from scripts.cwa_fetch.utils import append_record, ensure_dir, jsonl_path_for_month, ssl_verify, update_manifest

BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
REQUEST_INTERVAL_SEC = 1.5

# 資料集配置：(dataset_id, output_subdir, prefix, record_type, label)
# label 用於 title 前綴
EXTRA_DATASETS = [
    # W-C 天氣警特報
    ("W-C0033-001", "alerts", "official_alerts_wc", "alert", "各縣市警特報"),
    ("W-C0033-002", "alerts", "official_alerts_wc", "alert", "警特報內容"),
    ("W-C0033-003", "alerts", "official_alerts_wc", "alert", "豪大雨特報"),
    ("W-C0033-004", "alerts", "official_alerts_wc", "alert", "低溫特報"),
    ("W-C0033-005", "alerts", "official_alerts_wc", "alert", "高溫資訊"),
    ("W-C0034-001", "alerts", "official_alerts_typhoon", "alert", "颱風警報"),
    ("W-C0034-005", "alerts", "official_alerts_typhoon", "alert", "熱帶氣旋路徑"),
    # E-A 地震海嘯
    ("E-A0014-001", "alerts", "official_tsunami", "alert", "海嘯資訊"),
    ("E-A0015-001", "alerts", "official_earthquake", "alert", "顯著有感地震"),
    ("E-A0016-001", "alerts", "official_earthquake", "alert", "小區域有感地震"),
    # F-A 潮汐、健康氣象
    ("F-A0021-001", "daily", "official_tide", "forecast", "潮汐預報"),
    ("F-A0085-002", "daily", "official_health", "forecast", "冷傷害指數"),
    ("F-A0085-004", "daily", "official_health", "forecast", "溫差提醒"),
    # A-B 天文
    ("A-B0062-001", "daily", "official_astronomy", "forecast", "日出日沒"),
    ("A-B0063-001", "daily", "official_astronomy", "forecast", "月出月沒"),
]


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


def fetch_dataset(auth_key: str, dataset_id: str, verify: bool) -> dict | None:
    url = f"{BASE_URL}/{dataset_id}"
    params = {"Authorization": auth_key, "format": "JSON"}
    try:
        r = _request(url, params, verify)
        if r.status_code == 200:
            return r.json()
        if r.status_code != 404:
            r.raise_for_status()
    except requests.RequestException:
        pass
    return None


def extract_text_from_json(obj: object, min_len: int = 15) -> list[str]:
    """遞迴從 JSON 抽出含中文的長字串。"""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if len(v) >= min_len and re.search(r"[\u4e00-\u9fff]", v):
                    out.append(v.strip())
            else:
                out.extend(extract_text_from_json(v, min_len))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(extract_text_from_json(item, min_len))
    return out


def json_to_readable_content(data: dict) -> str:
    """將 API JSON 轉成可讀文字。"""
    parts = extract_text_from_json(data)
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen and len(p) >= 10:
            seen.add(p)
            unique.append(p)
    return "\n".join(unique[:50]) if unique else json.dumps(data, ensure_ascii=False)[:3000]


def main() -> None:
    auth_key = get_auth_key()
    verify = ssl_verify()
    base = REPO_ROOT / "data" / "corpus" / "weather" / "official"
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    year, month = now.year, now.month
    total = 0

    for dataset_id, subdir, prefix, rec_type, label in EXTRA_DATASETS:
        output_dir = base / subdir
        ensure_dir(output_dir)
        data = fetch_dataset(auth_key, dataset_id, verify)
        time.sleep(REQUEST_INTERVAL_SEC)

        if data is None:
            continue
        if data.get("success") is False:
            continue

        content = json_to_readable_content(data)
        if not content or len(content) < 20:
            continue

        jsonl_path = jsonl_path_for_month(output_dir, prefix, year, month)
        record = {
            "title": f"{label} {date_str}",
            "content": content[:8000],
            "date": date_str,
            "source": dataset_id,
            "type": rec_type,
        }
        append_record(jsonl_path, record)
        total += 1
        manifest_path = output_dir / "manifest.json"
        update_manifest(manifest_path, jsonl_path.name, date_start=date_str, date_end=date_str, record_count_delta=1)

    print(f"已寫入 {total} 筆至 official/（W-C/E-A/F-A/A-B）")


if __name__ == "__main__":
    main()
