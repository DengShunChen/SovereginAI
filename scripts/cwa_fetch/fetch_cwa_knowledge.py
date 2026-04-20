"""
擷取 CWA 官網「知識與天文」區塊的百科、百問內容，
寫入 data/corpus/weather/official/knowledge/cwa_knowledge.jsonl。
使用 requests + BeautifulSoup，內容自 .txt 子頁取得。
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from urllib3.exceptions import InsecureRequestWarning
except ImportError:
    print("請安裝：pip install requests beautifulsoup4 python-dotenv", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / "config" / ".env")
load_dotenv(REPO_ROOT / ".env")

from scripts.cwa_fetch.utils import append_record, ensure_dir, ssl_verify

BASE_URL = "https://www.cwa.gov.tw"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "official" / "knowledge"
REQUEST_INTERVAL_SEC = 1.5

# 百科配置：(list_pages, id_prefix_pattern, source, type)
# id_prefix_pattern: 用於過濾 href，如 r"typhoon-\d+" 或 r"sun-\d+|star-\d+"
ENCYCLOPEDIA_CONFIG = [
    # 颱風百問：多頁
    (
        [
            "/V8/C/K/Encyclopedia/typhoon/index.html",
            "/V8/C/K/Encyclopedia/typhoon/typhoon_list02.html",
            "/V8/C/K/Encyclopedia/typhoon/typhoon_list03.html",
            "/V8/C/K/Encyclopedia/typhoon/typhoon_list04.html",
            "/V8/C/K/Encyclopedia/typhoon/typhoon_list05.html",
            "/V8/C/K/Encyclopedia/typhoon/typhoon_appendix.html",
        ],
        "/V8/C/K/Encyclopedia/typhoon/",
        r"typhoon-\d+",
        "cwa_typhoon_faq",
        "faq",
    ),
    # 天文百問
    (
        [
            "/V8/C/K/Encyclopedia/astronomy/index.html",
            "/V8/C/K/Encyclopedia/astronomy/star_list.html",
            "/V8/C/K/Encyclopedia/astronomy/observation_list.html",
            "/V8/C/K/Encyclopedia/astronomy/calendar_list.html",
        ],
        "/V8/C/K/Encyclopedia/astronomy/",
        r"sun-\d+|star-\d+|observation-\d+|calendar-\d+",
        "cwa_astronomy_faq",
        "faq",
    ),
    # 氣候百問
    (
        ["/V8/C/K/Encyclopedia/climate/index.html"],
        "/V8/C/K/Encyclopedia/climate/",
        r"climate-\d+",
        "cwa_climate_faq",
        "faq",
    ),
    # 海象百問
    (
        [
            "/V8/C/K/Encyclopedia/sea/index.html",
            "/V8/C/K/Encyclopedia/sea/wave_list.html",
            "/V8/C/K/Encyclopedia/sea/tidal_list.html",
            "/V8/C/K/Encyclopedia/sea/surge_list.html",
            "/V8/C/K/Encyclopedia/sea/tsunami_list.html",
            "/V8/C/K/Encyclopedia/sea/weather_list.html",
            "/V8/C/K/Encyclopedia/sea/danger_list.html",
        ],
        "/V8/C/K/Encyclopedia/sea/",
        r"current-\d+|wave-\d+|tidal-\d+|surge-\d+|tsunami-\d+|weather-\d+|danger-\d+",
        "cwa_sea_faq",
        "faq",
    ),
    # 氣象常識
    (
        [
            "/V8/C/K/Encyclopedia/nous/index.html",
            "/V8/C/K/Encyclopedia/nous/overview_list.html",
            "/V8/C/K/Encyclopedia/nous/weather_list.html",
            "/V8/C/K/Encyclopedia/nous/climate_list.html",
            "/V8/C/K/Encyclopedia/nous/rain2_list.html",
        ],
        "/V8/C/K/Encyclopedia/nous/",
        r"introduction-\d+|overview-\d+|weather-\d+|climate-\d+|rain-\d+",
        "cwa_encyclopedia",
        "encyclopedia",
    ),
    # 氣象儀器
    (
        ["/V8/C/K/Encyclopedia/inst/index.html"],
        "/V8/C/K/Encyclopedia/inst/",
        r"instrument-\d+",
        "cwa_encyclopedia",
        "encyclopedia",
    ),
    # 氣候變遷問答
    (
        ["/V8/C/K/Qa/index.html"],
        "/V8/C/K/Qa/",
        r"qa_[\d-]+",
        "cwa_climate_faq",
        "faq",
    ),
    # 常見問答
    (
        ["/V8/C/K/CommonFaq/index.html"],
        "/V8/C/K/CommonFaq/",
        r"web-\d+",
        "cwa_popular_science",
        "faq",
    ),
]


def _request(url: str, verify: bool) -> requests.Response:
    if not verify:
        import urllib3
        urllib3.disable_warnings(InsecureRequestWarning)
    return requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CWA-Knowledge-Fetch/1.0)"},
        timeout=30,
        verify=verify,
    )


def extract_question_ids(soup: BeautifulSoup, id_pattern: re.Pattern) -> list[tuple[str, str]]:
    """從百科列表頁抽出 (id, question_text) 列表。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = re.search(r"#([a-zA-Z0-9_-]+)", href)
        if not m:
            continue
        qid = m.group(1)
        if qid in seen:
            continue
        if not id_pattern.search(qid):
            continue
        text = (a.get_text() or "").strip()
        if text and len(text) > 2:
            seen.add(qid)
            out.append((qid, text))
    return out


def fetch_content_txt(base_path: str, qid: str, verify: bool) -> tuple[str, str] | None:
    """擷取 .txt 內容，回傳 (title, content) 或 None。"""
    # 檔案名與 id 一致，如 typhoon-01.txt, qa_1-01.txt
    txt_url = urljoin(BASE_URL, base_path + qid + ".txt")
    try:
        r = _request(txt_url, verify)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        title_el = soup.find("h3", class_="cbp-l-inline-title")
        title = (title_el.get_text() or "").strip() if title_el else ""
        desc = soup.find("div", class_="cbp-l-inline-desc")
        if not desc:
            desc = soup.find("div", class_="cubmargin")
        if not desc:
            desc = soup
        content = (desc.get_text(separator=" ", strip=True) if desc else "").strip()
        if not content or len(content) < 10:
            return None
        if not title:
            title = qid
        return (title[:500], content[:8000])
    except Exception:
        return None


def main() -> None:
    verify = ssl_verify()
    ensure_dir(OUTPUT_DIR)
    out_path = OUTPUT_DIR / "cwa_knowledge.jsonl"

    # 覆寫模式：知識語料為靜態，每次全量擷取
    if out_path.exists():
        out_path.unlink()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    total = 0

    for list_urls, base_path, id_pat, source, rec_type in ENCYCLOPEDIA_CONFIG:
        id_pattern = re.compile(id_pat)
        seen_ids: set[str] = set()
        for list_url in list_urls:
            url = urljoin(BASE_URL, list_url)
            try:
                r = _request(url, verify)
                if r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                items = extract_question_ids(soup, id_pattern)
                for qid, question in items:
                    if qid in seen_ids:
                        continue
                    seen_ids.add(qid)
                    result = fetch_content_txt(base_path, qid, verify)
                    if result:
                        title, content = result
                        record = {
                            "title": title or question,
                            "content": content,
                            "date": date_str,
                            "source": source,
                            "type": rec_type,
                        }
                        append_record(out_path, record)
                        total += 1
                    time.sleep(REQUEST_INTERVAL_SEC)
            except Exception as e:
                print(f"擷取 {list_url} 失敗: {e}", file=sys.stderr)
            time.sleep(REQUEST_INTERVAL_SEC)

    if total > 0:
        print(f"已寫入 {total} 筆至 {out_path}")
    else:
        print("未擷取到任何語料。若遇 SSL 錯誤，請在 .env 設定 CWA_SSL_VERIFY=0", file=sys.stderr)


if __name__ == "__main__":
    main()
