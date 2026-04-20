#!/usr/bin/env python3
"""
語料就緒檢查：確保語料足以支撐「氣象主權 AI」訓練。
檢查項目：筆數與來源分布、詞表覆蓋率、內容長度、去重、簡體污染、訓練用切分。
自專案根目錄執行：python scripts/check_corpus_readiness.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "corpus" / "weather"
VOCAB_PATH = REPO_ROOT / "vocab" / "weather_terms_tw.txt"
FOR_TRAINING = REPO_ROOT / "data" / "processed" / "for_training"
PREPROCESSED = REPO_ROOT / "data" / "processed" / "preprocessed.jsonl"

# 簡體特徵字（與 preprocess 一致；不含「里」因「公里」繁簡同形）
SIMPLIFIED_CHARS = set(
    "国发说这会时过们来对产学个经为头无从两还见与长问开闻间页题车东转较边选单亚业点总认"
    "识习风险贝质门问关动现论当样变条种导数实级党际题无农体几较着气"
)


def _cjk_chars(text: str) -> str:
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")


def is_likely_simplified(text: str, min_cjk: int = 5, ratio: float = 0.25) -> bool:
    cjk = _cjk_chars(text)
    if len(cjk) < min_cjk:
        return False
    sc = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)
    return (sc / len(cjk)) >= ratio


def load_vocab(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line.split("\t")[0].strip())
    return out


def iter_corpus_jsonl(root: Path) -> list[dict]:
    records: list[dict] = []
    for p in root.rglob("*.jsonl"):
        if "manifest" in p.name:
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def main() -> None:
    print("=== 氣象主權 AI 語料就緒檢查 ===\n")

    # 1. 語料來源與筆數
    records = iter_corpus_jsonl(CORPUS_ROOT)
    if not records:
        print("未發現語料（data/corpus/weather/**/*.jsonl）。請先執行擷取與學術匯入。")
        sys.exit(1)

    by_type: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    # 依階段統計：官方 / 學術 / 社群
    phase_official = {
        "daily_summary",
        "weekly_forecast",
        "alert",
        "encyclopedia",
        "faq",
        "official_social",
    }
    phase_academic = {"academic_abstract", "thesis_abstract", "tech_report"}
    phase_social = {"forum_post", "social_post"}
    by_phase: dict[str, int] = defaultdict(int)
    dates: set[str] = set()
    lengths: list[int] = []
    content_hashes: set[str] = set()
    dup_count = 0
    simplified_count = 0
    schema_ok = 0
    required = {"title", "content", "date", "source", "type"}

    for r in records:
        t = r.get("type") or "unknown"
        by_type[t] += 1
        by_source[r.get("source") or "unknown"] += 1
        if t in phase_official:
            by_phase["official"] += 1
        elif t in phase_academic:
            by_phase["academic"] += 1
        elif t in phase_social:
            by_phase["social"] += 1
        else:
            by_phase["other"] += 1
        if r.get("date"):
            dates.add(r["date"])
        content = (r.get("content") or r.get("text") or "").strip()
        lengths.append(len(content))
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in content_hashes:
            dup_count += 1
        content_hashes.add(h)
        if is_likely_simplified(content):
            simplified_count += 1
        if required.issubset(set(r.keys())):
            schema_ok += 1

    total = len(records)
    unique = len(content_hashes)
    date_list = sorted(dates)
    date_span = f"{date_list[0]} ~ {date_list[-1]}" if date_list else "—"

    print("【1】語料筆數與來源")
    print(f"  總筆數: {total}（去重後 {unique}）")
    print(f"  日期範圍: {date_span}")
    print(f"  依階段: 官方 {by_phase['official']} / 學術 {by_phase['academic']} / 社群 {by_phase['social']} / 其他 {by_phase['other']}")
    print(f"  依 type: {dict(by_type)}")
    print(f"  依 source: {dict(by_source)}")

    # 2. 內容長度
    if lengths:
        lengths.sort()
        print("\n【2】內容長度（字元）")
        print(f"  最小/中位/最大: {min(lengths)} / {lengths[len(lengths)//2]} / {max(lengths)}")
        avg = sum(lengths) / len(lengths)
        print(f"  平均: {avg:.0f}")
        short = sum(1 for x in lengths if x < 50)
        ok = sum(1 for x in lengths if 50 <= x <= 2000)
        long_ = sum(1 for x in lengths if x > 2000)
        print(f"  分布: <50 字 {short} 筆, 50~2000 字 {ok} 筆, >2000 字 {long_} 筆")

    # 3. 詞表覆蓋率
    vocab = load_vocab(VOCAB_PATH)
    if vocab:
        found: set[str] = set()
        for r in records:
            c = (r.get("content") or r.get("text") or "").lower()
            for w in vocab:
                if w in c:
                    found.add(w)
        pct = 100.0 * len(found) / len(vocab) if vocab else 0
        print("\n【3】台灣氣象詞表覆蓋率")
        print(f"  詞表總數: {len(vocab)}, 語料中出現: {len(found)} ({pct:.1f}%)")
        missing = vocab - found
        if missing and len(missing) <= 20:
            print(f"  未出現詞例: {sorted(missing)[:20]}")
        elif missing:
            print(f"  未出現詞數: {len(missing)}（前 15 例: {sorted(missing)[:15]}）")

    # 4. 品質與污染
    print("\n【4】品質與污染")
    print(f"  重複內容筆數: {dup_count}")
    print(f"  疑似簡體內容筆數: {simplified_count} ({100*simplified_count/total:.1f}%)")
    print(f"  符合標準 schema 筆數: {schema_ok} / {total}")

    # 5. 訓練用切分
    train_count = valid_count = test_count = 0
    if FOR_TRAINING.exists():
        for name, path in [
            ("train", FOR_TRAINING / "train.jsonl"),
            ("valid", FOR_TRAINING / "valid.jsonl"),
            ("test", FOR_TRAINING / "test.jsonl"),
        ]:
            if path.exists():
                n = sum(1 for _ in open(path, "r", encoding="utf-8") if _.strip())
                if name == "train":
                    train_count = n
                elif name == "valid":
                    valid_count = n
                else:
                    test_count = n
    print("\n【5】訓練用切分（data/processed/for_training/）")
    print(f"  train: {train_count}, valid: {valid_count}, test: {test_count}")

    # 6. 就緒評估（氣象主權 AI）
    print("\n=== 就緒評估（氣象主權 AI）===")
    official_count = by_phase.get("official", 0)
    checks = []
    if total >= 100:
        checks.append(("語料總筆數 >= 100", True))
    else:
        checks.append(("語料總筆數 >= 100", False))

    if official_count >= 30:
        checks.append(("官方語料（預報/概況/警報）>= 30", True))
    else:
        checks.append(("官方語料（預報/概況/警報）>= 30", False))

    if vocab and len(found) / len(vocab) >= 0.2:
        checks.append(("詞表覆蓋率 >= 20%", True))
    elif vocab:
        checks.append(("詞表覆蓋率 >= 20%", False))
    else:
        checks.append(("詞表覆蓋率（詞表存在）", True))

    if simplified_count == 0:
        checks.append(("無簡體污染", True))
    else:
        checks.append(("無簡體污染", False))

    if train_count >= 10:
        checks.append(("訓練集筆數 >= 10", True))
    else:
        checks.append(("訓練集筆數 >= 10", False))

    if len(dates) >= 7:
        checks.append(("日期跨度 >= 7 天（利於多樣性）", True))
    else:
        checks.append(("日期跨度 >= 7 天（利於多樣性）", False))

    for label, ok in checks:
        status = "通過" if ok else "未達"
        print(f"  {status}: {label}")

    passed = sum(1 for _, ok in checks if ok)
    total_checks = len(checks)
    print(f"\n小計: {passed}/{total_checks} 項通過")
    if passed < total_checks:
        print("建議: 持續擷取官方語料（每日執行 run_collect_and_prepare.sh）、擴充學術語料、並重新執行前處理與切分。")
    else:
        print("語料已具備基本訓練條件，可接訓練程式。")


if __name__ == "__main__":
    main()
