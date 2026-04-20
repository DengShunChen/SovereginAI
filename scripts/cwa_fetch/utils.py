"""
台灣氣象語料庫 — CWA 擷取共用工具
寫入 JSONL 語料與 manifest，供後續前處理與訓練使用。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ssl_verify() -> bool:
    """依環境變數 CWA_SSL_VERIFY 決定是否驗證 SSL（預設 True）。若遇 CERTIFICATE_VERIFY_FAILED 可設為 0。"""
    v = os.environ.get("CWA_SSL_VERIFY", "1").strip().lower()
    return v not in ("0", "false", "no")

# 語料單筆建議欄位
RECORD_KEYS = ("title", "content", "date", "source", "type")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_record(
    filepath: Path,
    record: dict[str, Any],
    keys: tuple[str, ...] = RECORD_KEYS,
) -> None:
    """將一筆語料追加寫入 JSONL 檔。"""
    ensure_dir(filepath.parent)
    row = {k: record.get(k) for k in keys}
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """讀取 manifest.json；若不存在則回傳空結構。"""
    if not manifest_path.exists():
        return {"files": [], "date_range": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_manifest(
    manifest_path: Path,
    filename: str,
    date_start: str | None = None,
    date_end: str | None = None,
    record_count_delta: int = 0,
) -> None:
    """更新 manifest：新增一筆檔案紀錄與可選的日期範圍、筆數。"""
    ensure_dir(manifest_path.parent)
    data = read_manifest(manifest_path)
    if "files" not in data:
        data["files"] = []
    entry = {"file": filename}
    if date_start:
        entry["date_start"] = date_start
    if date_end:
        entry["date_end"] = date_end
    if record_count_delta:
        entry["record_count_delta"] = record_count_delta
    data["files"].append(entry)

    if date_start or date_end:
        dr = data.setdefault("date_range", {})
        if date_start and (not dr.get("start") or date_start < dr.get("start", "")):
            dr["start"] = date_start
        if date_end and (not dr.get("end") or date_end > dr.get("end", "")):
            dr["end"] = date_end

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def jsonl_path_for_month(base_dir: Path, prefix: str, year: int, month: int) -> Path:
    """依年月產生 JSONL 檔名，例如 official_daily_2025-01.jsonl。"""
    return base_dir / f"{prefix}_{year:04d}-{month:02d}.jsonl"


def extract_location_forecast_text(data: dict) -> list[tuple[str, str]]:
    """
    從具 records.location 結構的預報 API 回傳抽出 (標題, 可讀內容) 列表，
    每縣市一筆，供語料擴充用。
    """
    out: list[tuple[str, str]] = []
    records = data.get("records") or data.get("result") or data
    if not isinstance(records, dict) or "location" not in records:
        return out
    for loc in records.get("location", []) or []:
        name = loc.get("locationName") or loc.get("location") or "未知"
        parts: list[str] = []
        for we in loc.get("weatherElement", []) or []:
            elem_name = we.get("elementName") or we.get("name") or ""
            for t in we.get("time", []) or []:
                start = t.get("startTime", "")
                end = t.get("endTime", "")
                val = t.get("parameter") or t.get("value") or t
                if isinstance(val, dict):
                    p = val.get("parameterName") or val.get("parameterValue") or val.get("value")
                else:
                    p = str(val)
                if p:
                    parts.append(f"{elem_name}: {p} ({start}~{end})")
        if parts:
            out.append((f"天氣概況 {name}", "\n".join(parts)))
    return out
