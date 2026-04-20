"""
以 Playwright 渲染 CWA「研究出版與年度報告」頁面後擷取表格，
寫入 data/corpus/weather/academic/tech_reports/cwa_publications_list.jsonl。
需先安裝：pip install playwright && playwright install chromium
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("請安裝：pip install playwright && playwright install chromium", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "academic" / "tech_reports"
CWA_PUB_URL = "https://www.cwa.gov.tw/V8/E/D/publication.html"


def _norm_date(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "0000-00-00"
    m = re.search(r"(\d{4})[/\-]?(\d{1,2})?[/\-]?(\d{1,2})?", s)
    if m:
        y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return s[:10] if len(s) >= 10 else "0000-00-00"


def fetch_publication_list() -> list[dict]:
    records: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(CWA_PUB_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)
            # 表格：表頭 + 資料列
            rows = page.locator("table tbody tr").all()
            if not rows:
                rows = page.locator("table tr").all()
            if len(rows) < 2:
                return []
            header_texts = [c.inner_text().strip() for c in page.locator("table tr").first.locator("th, td").all()]
            name_idx = date_idx = type_idx = 0
            for i, h in enumerate(header_texts):
                h_lower = h.lower()
                if "name" in h_lower or "名稱" in h or "出版" in h:
                    name_idx = i
                if "date" in h_lower or "日期" in h:
                    date_idx = i
                if "type" in h_lower or "類型" in h:
                    type_idx = i
            for tr in rows[1:]:
                cells = tr.locator("th, td").all()
                texts = [c.inner_text().strip() for c in cells]
                if not texts:
                    continue
                title = texts[name_idx] if name_idx < len(texts) else texts[0]
                if not title:
                    continue
                date_str = _norm_date(texts[date_idx] if date_idx < len(texts) else "")
                type_val = texts[type_idx] if type_idx < len(texts) else ""
                parts = []
                if type_val:
                    parts.append(f"類型：{type_val}")
                parts.append("摘要或內文請自 CWA 官網取得。")
                records.append({
                    "title": title[:500],
                    "content": "；".join(parts),
                    "date": date_str,
                    "source": "cwa_tech_report",
                    "type": "tech_report",
                })
        finally:
            browser.close()
    return records


def main() -> None:
    records = fetch_publication_list()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cwa_publications_list.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if records:
        print(f"已寫入 {len(records)} 筆 CWA 出版品至 {out_path}")
    else:
        print("未解析到出版品列，請改用手動匯出：", CWA_PUB_URL, file=sys.stderr)


if __name__ == "__main__":
    main()
