"""
擷取中央氣象署「研究出版與年度報告」頁面之出版品列表（詮釋資料），
寫入 data/corpus/weather/academic/tech_reports/ 為 JSONL。
僅擷取官網公開之列表（標題、出版日、類型等），不下載 PDF；摘要或內文需手動取得。
需遵守 CWA 網站使用規範，勿高頻請求。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("請安裝：pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "academic" / "tech_reports"
CWA_PUB_URL = "https://www.cwa.gov.tw/V8/E/D/publication.html"
REQUEST_INTERVAL_SEC = 2.0


def _norm_date(s: str) -> str:
    """盡量轉成 YYYY-MM-DD。"""
    s = (s or "").strip()
    if not s:
        return "0000-00-00"
    m = re.search(r"(\d{4})[/\-]?(\d{1,2})?[/\-]?(\d{1,2})?", s)
    if m:
        y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return s[:10] if len(s) >= 10 else "0000-00-00"


def fetch_publication_list() -> list[dict]:
    """
    取得 CWA 出版品頁面 HTML，解析表格產出 (title, content, date, source, type)。
    若頁面為 JS 動態載入，表格可能為空，則回傳空列表。
    """
    try:
        r = requests.get(
            CWA_PUB_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WeatherCorpus/1.0)"},
            timeout=30,
        )
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
    except requests.RequestException as e:
        print(f"無法取得 CWA 出版品頁面: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    records: list[dict] = []

    # 嘗試多種常見表格結構
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        # 第一列當標題，取欄位名
        header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        if not header_cells:
            continue
        name_col = None
        date_col = None
        type_col = None
        for i, h in enumerate(header_cells):
            h_lower = h.lower()
            if "name" in h_lower or "名稱" in h or "出版" in h or "題" in h:
                name_col = i
            if "date" in h_lower or "日期" in h or "日" in h:
                date_col = i
            if "type" in h_lower or "類型" in h or "類" in h:
                type_col = i
        if name_col is None:
            # 若無明確名稱欄，取第一欄當標題
            name_col = 0
        for tr in rows[1:]:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            if name_col < len(texts) and texts[name_col]:
                title = texts[name_col]
                date_str = _norm_date(texts[date_col]) if date_col is not None and date_col < len(texts) else "0000-00-00"
                type_val = texts[type_col] if type_col is not None and type_col < len(texts) else ""
                # 組 content 為簡短說明（非全文）
                parts = []
                if type_val:
                    parts.append(f"類型：{type_val}")
                parts.append("摘要或內文請自官網連結取得。")
                content = "；".join(parts)
                records.append({
                    "title": title[:500],
                    "content": content,
                    "date": date_str,
                    "source": "cwa_tech_report",
                    "type": "tech_report",
                })
    return records


def main() -> None:
    time.sleep(REQUEST_INTERVAL_SEC)
    records = fetch_publication_list()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cwa_publications_list.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if records:
        print(f"已寫入 {len(records)} 筆 CWA 出版品列表至 {out_path}")
    else:
        print("未解析到出版品表格（頁面可能為 JS 動態載入）。請改用手動下載：", CWA_PUB_URL, file=sys.stderr)


if __name__ == "__main__":
    main()
