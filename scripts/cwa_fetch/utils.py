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


def _first_key(obj: dict, *keys: str):
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None


def iter_forecast_locations(data: dict) -> list[dict]:
    """相容舊版 records.location[] 與新版 records.Locations[].Location[]。"""
    records = data.get("records") or data.get("result") or data
    if not isinstance(records, dict):
        return []
    blocks = records.get("Locations") or records.get("locations")
    if blocks:
        if isinstance(blocks, dict):
            blocks = [blocks]
        out: list[dict] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            county = block.get("LocationsName") or block.get("locationsName") or ""
            inner = block.get("Location") or block.get("location") or []
            for loc in inner:
                if not isinstance(loc, dict):
                    continue
                row = dict(loc)
                row["_county"] = county
                out.append(row)
        if out:
            return out
    locs = records.get("location") or records.get("Location") or []
    return [loc for loc in locs if isinstance(loc, dict)]


def forecast_element_value(slot: dict) -> str:
    val = slot.get("parameter") or slot.get("value")
    if isinstance(val, dict):
        p = val.get("parameterName") or val.get("parameterValue") or val.get("value")
        return str(p) if p not in (None, "") else ""
    if val not in (None, "", slot) and not isinstance(val, (dict, list)):
        return str(val)
    ev = slot.get("ElementValue") or slot.get("elementValue")
    if isinstance(ev, list) and ev:
        first = ev[0]
        if isinstance(first, dict):
            for v in first.values():
                if v not in (None, ""):
                    return str(v)
        return str(first)
    if isinstance(ev, dict):
        for v in ev.values():
            if v not in (None, ""):
                return str(v)
    return ""


def forecast_start_time(slot: dict) -> str:
    raw = _first_key(slot, "startTime", "StartTime", "dataTime", "DataTime") or ""
    return str(raw)


def date_from_start_time(start: str, fallback: str) -> str:
    """從 ISO / 'YYYY-MM-DD HH:MM:SS' 取預報生效日。"""
    s = (start or "").strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return fallback


def extract_forecasts_by_day(
    data: dict,
    *,
    title_prefix: str,
    fallback_date: str,
) -> list[tuple[str, str, str]]:
    """
    依鄉鎮／縣市 × 預報生效日拆筆。
    回傳 (標題, 內容, date YYYY-MM-DD)。
    """
    out: list[tuple[str, str, str]] = []
    for loc in iter_forecast_locations(data):
        township = loc.get("LocationName") or loc.get("locationName") or loc.get("location") or "未知"
        county = loc.get("_county") or loc.get("CountyName") or loc.get("countyName") or ""
        place = f"{county}{township}" if county and county not in str(township) else str(township)
        by_day: dict[str, list[str]] = {}
        elements = loc.get("weatherElement") or loc.get("WeatherElement") or []
        for we in elements:
            if not isinstance(we, dict):
                continue
            elem_name = we.get("elementName") or we.get("ElementName") or we.get("name") or ""
            for slot in we.get("time") or we.get("Time") or []:
                if not isinstance(slot, dict):
                    continue
                start = forecast_start_time(slot)
                end = str(_first_key(slot, "endTime", "EndTime") or "")
                val = forecast_element_value(slot)
                if not val:
                    continue
                day = date_from_start_time(start, fallback_date)
                by_day.setdefault(day, []).append(f"{elem_name}: {val} ({start}~{end})")
        for day, parts in sorted(by_day.items()):
            if not parts:
                continue
            title = f"{title_prefix} {place} {day}"
            out.append((title, "\n".join(parts), day))
    return out


def extract_location_forecast_text(data: dict) -> list[tuple[str, str]]:
    """
    從具 records.location 結構的預報 API 回傳抽出 (標題, 可讀內容) 列表，
    每縣市一筆，供語料擴充用。
    """
    out: list[tuple[str, str]] = []
    for loc in iter_forecast_locations(data):
        name = loc.get("locationName") or loc.get("LocationName") or loc.get("location") or "未知"
        parts: list[str] = []
        elements = loc.get("weatherElement") or loc.get("WeatherElement") or []
        for we in elements:
            if not isinstance(we, dict):
                continue
            elem_name = we.get("elementName") or we.get("ElementName") or we.get("name") or ""
            for slot in we.get("time") or we.get("Time") or []:
                if not isinstance(slot, dict):
                    continue
                start = forecast_start_time(slot)
                end = str(_first_key(slot, "endTime", "EndTime") or "")
                val = forecast_element_value(slot)
                if val:
                    parts.append(f"{elem_name}: {val} ({start}~{end})")
        if parts:
            out.append((f"天氣概況 {name}", "\n".join(parts)))
    return out
