"""
擷取政府資料開放平臺（data.gov.tw）中與「氣象」相關之資料集詮釋資料
（標題、簡介），寫入 data/corpus/weather/academic/tech_reports/ 為 JSONL。
僅使用平台提供之詮釋資料 API，遵守使用規範與請求頻率。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("請安裝：pip install requests", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "academic" / "tech_reports"
# 政府資料開放平臺詮釋資料 API（依平台文件為準；部分端點可能需 API KEY 或已變更，見 https://data.gov.tw）
DATA_GOV_TW_API = "https://data.gov.tw/api/v2/rest/dataset"
REQUEST_INTERVAL_SEC = 1.5


def fetch_dataset_metadata(query: str = "氣象", limit: int = 100) -> list[dict]:
    """
    查詢 data.gov.tw 資料集詮釋資料，回傳 (title, content, date, source, type)。
    content 為資料集簡介（description），為可信任之技術說明文字。
    """
    records: list[dict] = []
    params = {"qs": query, "limit": min(limit, 100)}
    try:
        time.sleep(REQUEST_INTERVAL_SEC)
        r = requests.get(
            DATA_GOV_TW_API,
            params=params,
            headers={"accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"無法取得政府開放平臺 API: {e}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"API 回傳非 JSON: {e}", file=sys.stderr)
        return []

    # 平台回傳結構可能為 { "result": { "results": [...] } } 或直接陣列
    results = data.get("result", data)
    if isinstance(results, dict):
        results = results.get("results", results.get("data", []))
    if not isinstance(results, list):
        results = []

    for item in results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or item.get("title_zh") or item.get("name") or "").strip()
        desc = (item.get("notes") or item.get("description") or item.get("簡介") or "").strip()
        if not title and not desc:
            continue
        if not title:
            title = "(無標題)"
        # 日期：若有 modified 或 issued
        date_str = (item.get("metadata_modified") or item.get("issued") or item.get("date") or "")[:10]
        if not date_str or len(date_str) < 10:
            date_str = "0000-00-00"
        content = desc if desc else f"資料集：{title}（見政府開放平臺）。"
        records.append({
            "title": title[:500],
            "content": content[:5000],
            "date": date_str,
            "source": "data_gov_tw",
            "type": "tech_report",
        })
    return records


def main() -> None:
    records = fetch_dataset_metadata(query="氣象", limit=80)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "data_gov_tw_weather_metadata.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if records:
        print(f"已寫入 {len(records)} 筆政府開放平臺氣象資料集詮釋資料至 {out_path}")
    else:
        print("未取得資料（API 可能需金鑰或結構已變更）。請見 data/corpus/weather/academic/SOURCES.md。", file=sys.stderr)


if __name__ == "__main__":
    main()
