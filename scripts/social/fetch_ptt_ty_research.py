"""
擷取 PTT 大氣科學板 (TY_Research) 文章，寫入 data/corpus/weather/social/ptt_ty_research/。
僅供台灣氣象語料使用，遵守 PTT 站規與合理使用；請勿高頻請求。
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
from bs4 import BeautifulSoup

from scripts.cwa_fetch.utils import append_record, ensure_dir, update_manifest, read_manifest

BOARD_URL = "https://www.ptt.cc/bbs/TY_Research"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "social" / "ptt_ty_research"
REQUEST_INTERVAL_SEC = 1.5
SOURCE_LABEL = "ptt_ty_research"
TYPE_LABEL = "forum_post"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TaiwanWeatherCorpus/1.0; +research)",
}


def get_index_url(page: int | None) -> str:
    """最新頁 index.html；上一頁 index1444.html 等。page=None 為最新。"""
    if page is None or page <= 1:
        return f"{BOARD_URL}/index.html"
    return f"{BOARD_URL}/index{page}.html"


def fetch_html(url: str) -> str | None:
    r = requests.get(url, headers=HEADERS, cookies={"over18": "1"}, timeout=15)
    r.raise_for_status()
    return r.text


def parse_index(html: str) -> list[tuple[str, str]]:
    """回傳 (文章標題, 文章連結) 列表。"""
    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[str, str]] = []
    for div in soup.select("div.r-ent"):
        title_a = div.select_one("div.title a")
        if not title_a or not title_a.get("href"):
            continue
        if "公告" in (title_a.get_text() or ""):
            continue
        href = title_a["href"]
        title = (title_a.get_text() or "").strip()
        full_url = urljoin(BOARD_URL + "/", href)
        out.append((title, full_url))
    return out


def parse_article_content(html: str) -> str:
    """從文章頁取出內文（發信站之前）。"""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("#main-content")
    if not main:
        return ""
    for tag in main.select(".article-metaline, .article-metaline-right, span.f2"):
        tag.decompose()
    text = main.get_text(separator="\n")
    if "※ 發信站:" in text:
        text = text.split("※ 發信站:")[0]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def parse_article_date(html: str) -> str:
    """從文章頁盡量取得日期（例如 Mon Oct  8 12:30:00 2025）。"""
    soup = BeautifulSoup(html, "html.parser")
    for meta in soup.select("meta"):
        if meta.get("property") == "article:published_time" and meta.get("content"):
            try:
                dt = datetime.fromisoformat(meta["content"].replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    for span in soup.select("span.article-meta-value"):
        raw = (span.get_text() or "").strip()
        if re.match(r"^\w{3}\s+\w{3}\s+\d{1,2}\s+[\d:]+\s+\d{4}$", raw):
            try:
                dt = datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.now().strftime("%Y-%m-%d")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="擷取 PTT TY_Research 看板文章為語料")
    ap.add_argument("--pages", type=int, default=2, help="擷取頁數（從最新頁往前）")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="輸出目錄")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    prefix = "ptt_ty_research"
    year, month = datetime.now().year, datetime.now().month
    jsonl_path = args.out_dir / f"{prefix}_{year:04d}-{month:02d}.jsonl"
    manifest_path = args.out_dir / "manifest.json"

    total = 0
    seen_urls: set[str] = set()

    for page_idx in range(args.pages):
        page_num = page_idx + 1
        index_url = get_index_url(None if page_idx == 0 else page_num)
        print(f"擷取列表: {index_url}", file=sys.stderr)
        try:
            html = fetch_html(index_url)
        except Exception as e:
            print(f"略過列表頁: {e}", file=sys.stderr)
            time.sleep(REQUEST_INTERVAL_SEC)
            continue
        time.sleep(REQUEST_INTERVAL_SEC)

        items = parse_index(html or "")
        for title, art_url in items:
            if art_url in seen_urls:
                continue
            seen_urls.add(art_url)
            try:
                art_html = fetch_html(art_url)
            except Exception as e:
                print(f"略過文章 {art_url}: {e}", file=sys.stderr)
                time.sleep(REQUEST_INTERVAL_SEC)
                continue
            time.sleep(REQUEST_INTERVAL_SEC)

            content = parse_article_content(art_html or "")
            if len(content) < 20:
                continue
            date_str = parse_article_date(art_html or "")
            record = {
                "title": title[:200] if len(title) > 200 else title,
                "content": content,
                "date": date_str,
                "source": SOURCE_LABEL,
                "type": TYPE_LABEL,
            }
            append_record(jsonl_path, record)
            total += 1
            print(f"  + {title[:50]}...", file=sys.stderr)

    if total > 0:
        update_manifest(
            manifest_path,
            jsonl_path.name,
            date_start=datetime.now().strftime("%Y-%m-%d"),
            date_end=datetime.now().strftime("%Y-%m-%d"),
            record_count_delta=total,
        )
    print(f"已寫入 {total} 筆至 {jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
