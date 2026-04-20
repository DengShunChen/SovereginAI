"""
將手動匯出的 CWA 出版品表格（CSV）轉成與專案一致的 JSONL，
寫入 data/corpus/weather/academic/tech_reports/，供 preprocess 納入。
欄位名可為英文（Publication Name, Publish Date, Type）或中文（出版名稱、出版日期、類型）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "corpus" / "weather" / "academic" / "tech_reports"


def _norm_date(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "0000-00-00"
    m = re.search(r"(\d{4})[/\-]?(\d{1,2})?[/\-]?(\d{1,2})?", s)
    if m:
        y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return s[:10] if len(s) >= 10 else "0000-00-00"


def _find_col(row: dict, *candidates: str) -> str:
    for c in candidates:
        for k in row:
            if k and c.lower() in k.lower():
                return (row.get(k) or "").strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="將 CWA 出版品 CSV 轉成 JSONL")
    ap.add_argument("input", type=Path, help="手動匯出的 CWA 表格 CSV")
    ap.add_argument("--out", type=str, default="cwa_publications_list.jsonl", help="輸出檔名")
    ap.add_argument("--dry-run", action="store_true", help="僅預覽不寫檔")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"錯誤：找不到 {args.input}")
        raise SystemExit(1)

    records = []
    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = _find_col(row, "publication name", "出版名稱", "名稱", "name", "title", "題名")
            if not title:
                title = _find_col(row, "type", "類型") or next(iter(row.values()) or [""], "")
            if not title:
                continue
            date_str = _norm_date(_find_col(row, "publish date", "date", "出版日期", "日期", "update time"))
            type_val = _find_col(row, "type", "類型")
            parts = []
            if type_val:
                parts.append(f"類型：{type_val}")
            parts.append("摘要或內文請自 CWA 官網取得。")
            content = "；".join(parts)
            records.append({
                "title": title[:500],
                "content": content,
                "date": date_str,
                "source": "cwa_tech_report",
                "type": "tech_report",
            })

    if not records:
        print("未解析到有效列，請確認 CSV 含「出版名稱」或「Publication Name」等欄位。")
        raise SystemExit(0)

    out_path = OUT_DIR / args.out
    if args.dry_run:
        print(f"將寫入 {len(records)} 筆至 {out_path}")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已寫入 {len(records)} 筆至 {out_path}")


if __name__ == "__main__":
    main()
