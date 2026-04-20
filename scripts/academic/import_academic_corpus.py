"""
將 CSV 或 JSON 格式的學術語料轉成與官方語料一致的 JSONL，
寫入 data/corpus/weather/academic/ 下指定子目錄，供 preprocess.py 一併讀取。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACADEMIC_ROOT = REPO_ROOT / "data" / "corpus" / "weather" / "academic"

# 與 official 語料一致的欄位
FIELDS = ("title", "content", "date", "source", "type")


def norm_record(raw: dict) -> dict | None:
    """正規化單筆為 title/content/date/source/type，缺必要欄位則略過。"""
    title = (raw.get("title") or raw.get("標題") or "").strip()
    content = (raw.get("content") or raw.get("摘要") or raw.get("內文") or "").strip()
    if not content:
        return None
    date = (raw.get("date") or raw.get("日期") or "").strip()
    if not date or len(date) < 10:
        date = "0000-00-00"
    else:
        date = date[:10]
    source = (raw.get("source") or raw.get("來源") or "academic").strip() or "academic"
    type_ = (raw.get("type") or raw.get("類型") or "academic_abstract").strip() or "academic_abstract"
    return {
        "title": title or "(無標題)",
        "content": content,
        "date": date,
        "source": source,
        "type": type_,
    }


def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return [data]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="匯入學術語料：從 CSV/JSON 轉成 JSONL，寫入 academic 目錄"
    )
    ap.add_argument(
        "input",
        type=Path,
        help="輸入檔（.csv 或 .json）；CSV 需含 title/content/date/source/type 或 標題/摘要/日期/來源/類型",
    )
    ap.add_argument(
        "--subdir",
        type=str,
        default="thesis",
        choices=["thesis", "conference", "tech_reports"],
        help="寫入 academic 下哪個子目錄",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="輸出檔名（不含路徑）；預設為 input 檔名改副檔名 .jsonl",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示將寫入的筆數與一筆範例，不寫檔",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"錯誤：找不到輸入檔 {args.input}")
        raise SystemExit(1)

    if args.input.suffix.lower() == ".csv":
        rows = load_csv(args.input)
    elif args.input.suffix.lower() in (".json", ".jsonl"):
        rows = load_json(args.input)
    else:
        print("錯誤：僅支援 .csv 或 .json 輸入")
        raise SystemExit(1)

    records = []
    for r in rows:
        rec = norm_record(r)
        if rec:
            records.append(rec)

    if not records:
        print("沒有可匯出的筆數（需至少含 content）。")
        raise SystemExit(0)

    out_name = args.out or args.input.stem + ".jsonl"
    out_dir = ACADEMIC_ROOT / args.subdir
    out_path = out_dir / out_name

    if args.dry_run:
        print(f"將寫入 {len(records)} 筆至 {out_path}")
        print("範例：", json.dumps(records[0], ensure_ascii=False, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已寫入 {len(records)} 筆至 {out_path}")


if __name__ == "__main__":
    main()
