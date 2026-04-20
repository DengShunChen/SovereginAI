"""
擷取 CWA 數位科普網（pweb.cwa.gov.tw/PopularScience）文章，
寫入 data/corpus/weather/official/knowledge/cwa_popular_science.jsonl。
"""
from __future__ import annotations

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

BASE_URL = "https://pweb.cwa.gov.tw"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "official" / "knowledge"
REQUEST_INTERVAL_SEC = 1.5

# 數位科普分類頁
CATEGORY_PAGES = [
    "/PopularScience/wt_cate.html",   # 氣象
    "/PopularScience/ec_cate.html",   # 地震
    "/PopularScience/ma_cate.html",   # 海象
    "/PopularScience/as_cate.html",   # 天文
    "/PopularScience/pr_cate.html",   # 防災
    "/PopularScience/kids_cate.html", # 兒童
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


def extract_article_links(soup: BeautifulSoup, base_path: str) -> list[str]:
    """從分類頁抽出文章連結（相對路徑）。"""
    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.endswith(".html") or "cate" in href or "index" in href:
            continue
        if href.startswith("/"):
            path = href
        else:
            path = base_path + href
        if path in seen:
            continue
        if re.search(r"/(wt|ec|ma|as|pr|kids)/[a-z0-9_]+\.html", path):
            seen.add(path)
            out.append(path)
    return out


def extract_article_content(soup: BeautifulSoup) -> tuple[str, str]:
    """從文章頁抽出標題與內文。"""
    title = ""
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = (h.get_text() or "").strip()
        if t and len(t) > 2 and re.search(r"[\u4e00-\u9fff]", t):
            title = t[:300]
            break
    if not title:
        title_el = soup.find("title")
        if title_el:
            title = (title_el.get_text() or "").replace("中央氣象署數位科普網-", "").strip()[:300]

    content_parts: list[str] = []
    for tag in soup.find_all(["p", "li", "h2", "h3", "h4"]):
        t = (tag.get_text() or "").strip()
        if t and len(t) >= 15 and re.search(r"[\u4e00-\u9fff]", t):
            content_parts.append(t)
    content = "\n".join(content_parts)[:8000].strip()
    return title or "數位科普", content


def main() -> None:
    verify = ssl_verify()
    ensure_dir(OUTPUT_DIR)
    out_path = OUTPUT_DIR / "cwa_popular_science.jsonl"
    if out_path.exists():
        out_path.unlink()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    all_links: set[str] = set()
    for cat_path in CATEGORY_PAGES:
        url = urljoin(BASE_URL, cat_path)
        try:
            r = _request(url, verify)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            base = "/".join(cat_path.split("/")[:-1]) + "/"
            links = extract_article_links(soup, base)
            for lnk in links:
                full = urljoin(BASE_URL, lnk) if lnk.startswith("/") else urljoin(url, lnk)
                all_links.add(full)
        except Exception as e:
            print(f"擷取 {cat_path} 失敗: {e}", file=sys.stderr)
        time.sleep(REQUEST_INTERVAL_SEC)

    total = 0
    for art_url in sorted(all_links):
        try:
            r = _request(art_url, verify)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            title, content = extract_article_content(soup)
            if not content or len(content) < 50:
                continue
            record = {
                "title": title,
                "content": content,
                "date": date_str,
                "source": "cwa_popular_science",
                "type": "encyclopedia",
            }
            append_record(out_path, record)
            total += 1
        except Exception as e:
            print(f"擷取 {art_url} 失敗: {e}", file=sys.stderr)
        time.sleep(REQUEST_INTERVAL_SEC)

    if total > 0:
        print(f"已寫入 {total} 筆數位科普至 {out_path}")
    else:
        print("未擷取到數位科普語料。若遇 SSL 錯誤，請在 .env 設定 CWA_SSL_VERIFY=0", file=sys.stderr)


if __name__ == "__main__":
    main()
