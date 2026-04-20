"""
擷取中央氣象署「氣象報告天氣概況」F-C0044-001，
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

# 專案根目錄
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / "config" / ".env")
load_dotenv(REPO_ROOT / ".env")

from scripts.cwa_fetch.utils import (
    append_record,
    extract_location_forecast_text,
    jsonl_path_for_month,
    ssl_verify,
    update_manifest,
)

DATASET_ID = "F-C0044-001"
# 天氣概況在 rest/datastore 可能回 404，改試 fileapi
BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
FILEAPI_URL = "https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-C0044-001"
# 備援：縣市預報（rest/datastore 通常可用）
FALLBACK_DATASET_ID = "F-C0032-001"
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


def fetch_json(auth_key: str) -> tuple[dict, str]:
    """回傳 (API 資料, 實際使用的 source_id)。"""
    params = {"Authorization": auth_key, "format": "JSON"}
    verify = ssl_verify()

    # 1. 先試 rest/datastore F-C0044-001
    url_rest = f"{BASE_URL}/{DATASET_ID}"
    for attempt in range(2):
        try:
            r = _request(url_rest, params, verify)
            if r.status_code == 200:
                return r.json(), DATASET_ID
            if r.status_code != 404:
                r.raise_for_status()
        except requests.RequestException:
            if attempt == 0:
                time.sleep(REQUEST_INTERVAL_SEC)
            else:
                pass
        break

    # 2. 再試 fileapi F-C0044-001
    for attempt in range(2):
        try:
            r = _request(FILEAPI_URL, params, verify)
            if r.status_code == 200:
                ct = (r.headers.get("Content-Type") or "").lower()
                if "json" in ct:
                    return r.json(), DATASET_ID
                if "xml" in ct:
                    text = _parse_fileapi_xml(r.text)
                    if text:
                        return {"_parsed_text": text}, DATASET_ID
        except Exception:
            if attempt == 0:
                time.sleep(REQUEST_INTERVAL_SEC)
        break

    # 3. 備援：縣市預報 F-C0032-001（rest/datastore 通常可用）
    url_fallback = f"{BASE_URL}/{FALLBACK_DATASET_ID}"
    for attempt in range(3):
        try:
            r = _request(url_fallback, params, verify)
            r.raise_for_status()
            data = r.json()
            if data:
                return data, FALLBACK_DATASET_ID
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(REQUEST_INTERVAL_SEC * (attempt + 1))
            else:
                raise
    return {}, FALLBACK_DATASET_ID


def _parse_fileapi_xml(xml_text: str) -> str:
    """從 fileapi XML 抽出內文。"""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
        parts: list[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip() and len(elem.text.strip()) > 10:
                parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip() and len(elem.tail.strip()) > 10:
                parts.append(elem.tail.strip())
        return "\n".join(parts) if parts else ""
    except ET.ParseError:
        return ""


def extract_text_from_response(data: dict) -> list[tuple[str, str]]:
    """
    從 API 回傳的 JSON 抽出 (標題, 純文本) 列表。
    相容常見結構：records.location、records 陣列、或 result 等。
    """
    out: list[tuple[str, str]] = []
    records = data.get("records") or data.get("result") or data

    def collect_text(obj: dict | list, prefix: str = "") -> None:
        if isinstance(obj, dict):
            # 常見欄位：content、description、weatherSummary、value、elementValue
            for key in ("content", "description", "weatherSummary", "value", "weather", "text"):
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    title = prefix or f"天氣概況 {key}"
                    out.append((title, obj[key].strip()))
            for k, v in obj.items():
                collect_text(v, prefix or k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                collect_text(item, f"{prefix}[{i}]" if prefix else "")

    if isinstance(records, dict) and "location" in records:
        for loc in records.get("location", []) or []:
            name = loc.get("locationName") or loc.get("location") or "全台"
            collect_text(loc, f"天氣概況 {name}")
    elif isinstance(records, dict):
        collect_text(records, "天氣概況")
    elif isinstance(records, list):
        for r in records:
            collect_text(r)

    # 若完全沒抽到，嘗試整份 JSON 中所有字串（備援）
    if not out:
        def all_strings(obj: dict | list, depth: int = 0) -> None:
            if depth > 10:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and len(v) > 20 and re.search(r"[\u4e00-\u9fff]", v):
                        out.append((k, v.strip()))
                    else:
                        all_strings(v, depth + 1)
            elif isinstance(obj, list):
                for v in obj:
                    all_strings(v, depth + 1)
        all_strings(data)

    return out


def main() -> None:
    auth_key = get_auth_key()
    data, source_id = fetch_json(auth_key)

    # fileapi 解析結果
    if "_parsed_text" in data:
        items = [("天氣概況", data["_parsed_text"])]
    else:
        success = data.get("success")
        if success is False:
            print("API 回傳失敗", data.get("result") or data, file=sys.stderr)
            sys.exit(1)
        items = extract_text_from_response(data)

    if not items:
        items = extract_location_forecast_text(data)
    if not items:
        raw = json.dumps(data, ensure_ascii=False)[:2000]
        items = [("天氣概況（無文本）", raw)]

    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    jsonl_path = jsonl_path_for_month(OUTPUT_DIR, "official_daily", year, month)
    manifest_path = OUTPUT_DIR / "manifest.json"
    date_str = now.strftime("%Y-%m-%d")

    count = 0
    for i, (title, content) in enumerate(items):
        if not content or len(content) < 5:
            continue
        record = {
            "title": title if len(title) < 200 else title[:197] + "...",
            "content": content,
            "date": date_str,
            "source": source_id,
            "type": "daily_summary",
        }
        append_record(jsonl_path, record)
        count += 1

    if count > 0:
        update_manifest(manifest_path, jsonl_path.name, date_start=date_str, date_end=date_str, record_count_delta=count)

    print(f"已寫入 {count} 筆至 {jsonl_path}")


if __name__ == "__main__":
    main()
