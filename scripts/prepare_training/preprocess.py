"""
讀取 data/corpus/weather/**/*.jsonl，做清理（去重、長度過濾、可選 type/source 篩選），
並排除簡體中文／中國大陸污染語料（簡體偵測、允許來源名單）。
輸出統計與可選的清洗後 JSONL，供後續格式轉換使用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "data" / "corpus" / "weather"
VOCAB_PATH = REPO_ROOT / "vocab" / "weather_terms_tw.txt"
OUT_PREPROCESSED = REPO_ROOT / "data" / "processed" / "preprocessed.jsonl"
ALLOWED_SOURCES_PATH = REPO_ROOT / "config" / "allowed_corpus_sources.txt"
MAINLAND_TERMS_PATH = REPO_ROOT / "vocab" / "mainland_terms_filter.txt"
MAINLAND_TO_TW_PATH = REPO_ROOT / "vocab" / "mainland_to_tw_replace.txt"

# 簡體中文常見字（與繁體不同碼位），用於粗判是否為簡體語料，避免中國大陸／簡體污染
# 不含「里」：公里／海里 繁簡同形，避免誤殺官方換算表等語料
SIMPLIFIED_CHARS = set(
    "国发说这会时过们来对产学个经为头无从两还见与长问开闻间页题车东转较边选单亚业点总认"
    "识习风险贝质门问关动现论当样变条种导数实级党际题无农体几较着气"
)


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
                    continue
    return records


def content_hash(record: dict) -> str:
    c = record.get("content") or record.get("text") or ""
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def _cjk_chars(text: str) -> str:
    """取出 CJK 字元（含中文、日韓漢字）。"""
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")


def is_likely_simplified_chinese(
    text: str,
    *,
    min_cjk: int = 5,
    ratio_threshold: float = 0.25,
) -> bool:
    """
    粗判內容是否偏向簡體中文，以利排除中國大陸／簡體污染語料。
    若 CJK 字數不足 min_cjk 則不判定；否則當「簡體特徵字佔 CJK 字數」>= ratio_threshold 視為簡體。
    """
    cjk = _cjk_chars(text)
    if len(cjk) < min_cjk:
        return False
    sc_count = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)
    return (sc_count / len(cjk)) >= ratio_threshold


def load_allowed_sources(path: Path) -> set[str] | None:
    """讀取允許來源名單；若檔案不存在或為空則回傳 None（不依來源過濾）。"""
    if not path.exists():
        return None
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out if out else None


def load_mainland_terms(path: Path) -> list[str]:
    """讀取中國用語詞表，回傳列表（長詞優先，利於匹配）。"""
    if not path.exists():
        return []
    terms: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line.split("\t")[0].strip())
    return sorted(terms, key=len, reverse=True)


def load_mainland_to_tw_replace(path: Path) -> list[tuple[str, str]]:
    """讀取中國用語→台灣用語替換表，回傳 (中國用語, 台灣用語) 列表，長詞優先。"""
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "\t" in line:
                a, b = line.split("\t", 1)
                a, b = a.strip(), b.strip()
                if a and b and a != b:
                    out.append((a, b))
    return sorted(out, key=lambda x: len(x[0]), reverse=True)


def count_mainland_usage(content: str, terms: list[str]) -> tuple[int, int]:
    """回傳 (出現之中國用語種類數, 總出現次數)。"""
    distinct: set[str] = set()
    total = 0
    for t in terms:
        if not t or t not in content:
            continue
        distinct.add(t)
        total += content.count(t)
    return len(distinct), total


def has_too_much_mainland(
    content: str,
    terms: list[str],
    *,
    max_distinct: int = 2,
    max_ratio: float = 0.02,
) -> bool:
    """若內容出現過多中國用語（種類數或出現次數／字數比），回傳 True，建議排除該筆。"""
    if not terms or len(content) < 10:
        return False
    distinct, total = count_mainland_usage(content, terms)
    if distinct >= max_distinct:
        return True
    if total > 0 and (total / max(1, len(content))) >= max_ratio:
        return True
    return False


def normalize_mainland_to_tw(content: str, replace_list: list[tuple[str, str]]) -> str:
    """將內容中的中國用語替換為台灣用語（長詞先替換）。"""
    for mainland, tw in replace_list:
        content = content.replace(mainland, tw)
    return content


UI_JUNK_MARKERS = (
    "該使用哪一版本的瀏覽器",
    "建議您使用較新版本的瀏覽器",
    "網頁暫存快取",
    "設定每次開啟瀏覽器",
    "Home Page（首頁）",
    "新版找不到",
    "新版 V8 「天氣」",
    "Safari】→【清除瀏覽",
    "點選【儲存空間】",
    "字級大小",
    "網站建議使用",
    "自動更新功能",
    "清除您電腦裡的網頁暫存",
)


def is_ui_junk(record: dict) -> bool:
    blob = f"{record.get('title') or ''}\n{record.get('content') or record.get('text') or ''}"
    return any(m in blob for m in UI_JUNK_MARKERS)


def preprocess(
    records: list[dict],
    *,
    min_content_len: int = 10,
    max_content_len: int = 0,
    types: list[str] | None = None,
    sources: list[str] | None = None,
    dedup: bool = True,
    exclude_simplified: bool = True,
    allowed_sources: set[str] | None = None,
    exclude_mainland_usage: bool = False,
    mainland_terms: list[str] | None = None,
    max_mainland_distinct: int = 2,
    max_mainland_ratio: float = 0.02,
    normalize_mainland_to_tw_list: list[tuple[str, str]] | None = None,
    drop_ui_junk: bool = False,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for r in records:
        content = (r.get("content") or r.get("text") or "").strip()
        if len(content) < min_content_len:
            continue
        if max_content_len and len(content) > max_content_len:
            content = content[:max_content_len]
            r = {**r, "content": content}
        if types and (r.get("type") or "") not in types:
            continue
        if sources and (r.get("source") or "") not in sources:
            continue
        if drop_ui_junk and is_ui_junk(r):
            continue
        if exclude_simplified and is_likely_simplified_chinese(content):
            continue
        if allowed_sources is not None:
            src = (r.get("source") or "").strip()
            if src not in allowed_sources:
                continue
        if exclude_mainland_usage and mainland_terms and has_too_much_mainland(
            content, mainland_terms, max_distinct=max_mainland_distinct, max_ratio=max_mainland_ratio
        ):
            continue
        if normalize_mainland_to_tw_list:
            content = normalize_mainland_to_tw(content, normalize_mainland_to_tw_list)
            r = {**r, "content": content}
        if dedup:
            h = content_hash(r)
            if h in seen:
                continue
            seen.add(h)
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="語料前處理：去重、長度過濾、篩選")
    ap.add_argument("--corpus", type=Path, default=CORPUS_ROOT, help="語料根目錄")
    ap.add_argument("--out", type=Path, default=OUT_PREPROCESSED, help="輸出 JSONL")
    ap.add_argument("--min-len", type=int, default=10, help="最少字數")
    ap.add_argument("--max-len", type=int, default=0, help="最多字數（0 不截斷）")
    ap.add_argument("--types", type=str, default="", help="逗號分隔 type 篩選，空則不篩")
    ap.add_argument("--sources", type=str, default="", help="逗號分隔 source 篩選，空則不篩")
    ap.add_argument("--no-dedup", action="store_true", help="不做去重")
    ap.add_argument(
        "--no-exclude-simplified",
        action="store_true",
        help="不排除簡體中文（預設會排除，避免中國大陸／簡體污染語料）",
    )
    ap.add_argument(
        "--allowed-sources-file",
        type=Path,
        default=ALLOWED_SOURCES_PATH,
        help="允許來源名單檔（每行一個 source）；若存在則只保留名單內來源",
    )
    ap.add_argument(
        "--no-allowed-sources",
        action="store_true",
        help="不使用允許來源名單，保留所有來源",
    )
    ap.add_argument(
        "--exclude-mainland-usage",
        action="store_true",
        help="排除含過多中國用語的筆數（依 vocab/mainland_terms_filter.txt）",
    )
    ap.add_argument(
        "--mainland-terms-file",
        type=Path,
        default=MAINLAND_TERMS_PATH,
        help="中國用語過濾詞表（每行一詞）",
    )
    ap.add_argument(
        "--max-mainland-distinct",
        type=int,
        default=2,
        help="出現幾種中國用語即排除該筆（預設 2）",
    )
    ap.add_argument(
        "--max-mainland-ratio",
        type=float,
        default=0.02,
        help="中國用語出現次數／字數比超過此值即排除（預設 0.02）",
    )
    ap.add_argument(
        "--normalize-mainland-to-tw",
        action="store_true",
        help="將中國用語替換為台灣用語（依 mainland_to_tw_replace.txt），不刪筆數",
    )
    ap.add_argument(
        "--mainland-to-tw-file",
        type=Path,
        default=MAINLAND_TO_TW_PATH,
        help="中國用語→台灣用語替換表（格式：中國用語\\t台灣用語）",
    )
    ap.add_argument(
        "--drop-ui-junk",
        action="store_true",
        help="排除官網操作 FAQ（瀏覽器設定、新版選單路徑等）",
    )
    ap.add_argument("--vocab-stats", action="store_true", help="輸出詞表出現次數統計")
    args = ap.parse_args()

    records = iter_corpus_jsonl(args.corpus)
    types = [x.strip() for x in args.types.split(",") if x.strip()] or None
    sources = [x.strip() for x in args.sources.split(",") if x.strip()] or None
    allowed_sources = None
    if not args.no_allowed_sources and args.allowed_sources_file:
        allowed_sources = load_allowed_sources(args.allowed_sources_file)
    mainland_terms = load_mainland_terms(args.mainland_terms_file) if args.exclude_mainland_usage else None
    normalize_list = load_mainland_to_tw_replace(args.mainland_to_tw_file) if args.normalize_mainland_to_tw else None
    cleaned = preprocess(
        records,
        min_content_len=args.min_len,
        max_content_len=args.max_len or 0,
        types=types,
        sources=sources,
        dedup=not args.no_dedup,
        exclude_simplified=not args.no_exclude_simplified,
        allowed_sources=allowed_sources,
        exclude_mainland_usage=args.exclude_mainland_usage,
        mainland_terms=mainland_terms,
        max_mainland_distinct=args.max_mainland_distinct,
        max_mainland_ratio=args.max_mainland_ratio,
        normalize_mainland_to_tw_list=normalize_list,
        drop_ui_junk=args.drop_ui_junk,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"讀取 {len(records)} 筆，清洗後 {len(cleaned)} 筆，已寫入 {args.out}")

    if args.vocab_stats and VOCAB_PATH.exists():
        vocab = load_vocab(VOCAB_PATH)
        counts: dict[str, int] = {w: 0 for w in vocab}
        for r in cleaned:
            c = (r.get("content") or "").lower()
            for w in vocab:
                if w in c:
                    counts[w] += 1
        for w, n in sorted(counts.items(), key=lambda x: -x[1])[:30]:
            if n:
                print(f"  {w}: {n}")


if __name__ == "__main__":
    main()
