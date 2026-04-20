"""
將前處理後的語料轉成訓練用格式，並做 train/valid/test 切分，寫入 data/processed/for_training/。
支援：純文本（氣象文案生成）、指令微調（instruction/input/output）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED = REPO_ROOT / "data" / "processed" / "preprocessed.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed" / "for_training"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALID_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


def load_preprocessed(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def to_plain_text(records: list[dict]) -> list[dict]:
    """氣象文案生成：每筆為 {"text": content}。"""
    out = []
    for r in records:
        content = (r.get("content") or r.get("text") or "").strip()
        if not content:
            continue
        out.append({"text": content})
    return out


def to_instruction(records: list[dict], instruction_template: str = "請根據以下資料寫今日天氣概況。") -> list[dict]:
    """指令微調：每筆為 {"instruction": ..., "input": ..., "output": ...}。"""
    out = []
    for r in records:
        content = (r.get("content") or r.get("text") or "").strip()
        if not content:
            continue
        title = r.get("title") or ""
        out.append({
            "instruction": instruction_template,
            "input": title or "天氣資料",
            "output": content,
        })
    return out


def _allocate_splits_largest_remainder(n: int, w_train: float, w_valid: float, w_test: float) -> tuple[int, int, int]:
    """依比例分配 n 筆到三份，總和必為 n（最大餘數法），避免 int 截斷造成空 split 或漏筆。"""
    s = w_train + w_valid + w_test
    if n == 0:
        return 0, 0, 0
    if s <= 0:
        return n, 0, 0
    p = (w_train / s, w_valid / s, w_test / s)
    exact = [n * p[0], n * p[1], n * p[2]]
    seats = [int(x) for x in exact]
    rem = n - sum(seats)
    order = sorted(range(3), key=lambda i: exact[i] - seats[i], reverse=True)
    for k in range(rem):
        seats[order[k]] += 1
    return seats[0], seats[1], seats[2]


def split(records: list[dict], train: float, valid: float, test: float, seed: int = 42) -> tuple[list, list, list]:
    total = train + valid + test
    train_w, valid_w, test_w = train / total, valid / total, test / total
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n = len(indices)
    if n == 0:
        return [], [], []
    n_tr, n_va, n_te = _allocate_splits_largest_remainder(n, train_w, valid_w, test_w)
    a = n_tr
    b = a + n_va
    tr = [records[indices[i]] for i in range(0, a)]
    va = [records[indices[i]] for i in range(a, b)]
    te = [records[indices[i]] for i in range(b, n)]
    return tr, va, te


def main() -> None:
    ap = argparse.ArgumentParser(description="產出訓練用資料集並切分")
    ap.add_argument("--input", type=Path, default=PREPROCESSED, help="前處理後 JSONL")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="輸出目錄")
    ap.add_argument("--format", choices=["plain", "instruction"], default="plain", help="輸出格式")
    ap.add_argument("--train", type=float, default=DEFAULT_TRAIN_RATIO)
    ap.add_argument("--valid", type=float, default=DEFAULT_VALID_RATIO)
    ap.add_argument("--test", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_preprocessed(args.input)
    if not records:
        print("無前處理語料，請先執行 preprocess.py", file=sys.stderr)
        sys.exit(1)

    if args.format == "plain":
        converted = to_plain_text(records)
    else:
        converted = to_instruction(records)

    tr, va, te = split(converted, args.train, args.valid, args.test, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", tr), ("valid", va), ("test", te)]:
        path = args.out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(data)} 筆")

    schema = {
        "format": args.format,
        "train_ratio": args.train,
        "valid_ratio": args.valid,
        "test_ratio": args.test,
        "schema": "每行一 JSON；plain 含 text；instruction 含 instruction, input, output",
    }
    with open(args.out / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"已寫入 {args.out}，schema 見 schema.json")


if __name__ == "__main__":
    main()
