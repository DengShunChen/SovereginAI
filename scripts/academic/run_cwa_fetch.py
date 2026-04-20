"""
CWA 出版品擷取：先試 Playwright（可解析 JS 表格），失敗則試 requests，皆無則提示手動匯出。
本機執行（需網路）：
  pip install requests beautifulsoup4
  pip install playwright && playwright install chromium   # 可選，用於 JS 頁面
  python scripts/academic/run_cwa_fetch.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYWRIGHT_SCRIPT = Path(__file__).parent / "fetch_cwa_publications_playwright.py"
REQUESTS_SCRIPT = Path(__file__).parent / "fetch_cwa_publications.py"
CWA_URL = "https://www.cwa.gov.tw/V8/E/D/publication.html"


def main() -> None:
    # 1. 試 Playwright（可取得 JS 動態表格）
    if PLAYWRIGHT_SCRIPT.exists():
        try:
            r = subprocess.run(
                [sys.executable, str(PLAYWRIGHT_SCRIPT)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            if r.returncode == 0 and "已寫入" in out and "筆" in out:
                print(out)
                return
            if err:
                print(err, file=sys.stderr)
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            print("Playwright 擷取逾時", file=sys.stderr)
        except Exception as e:
            print(f"Playwright 擷取失敗: {e}", file=sys.stderr)

    # 2. 試 requests
    try:
        r = subprocess.run(
            [sys.executable, str(REQUESTS_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (r.stdout or "").strip()
        if r.returncode == 0 and "已寫入" in out and "筆" in out:
            print(out)
            return
    except Exception as e:
        print(f"Requests 擷取失敗: {e}", file=sys.stderr)

    # 3. 提示手動匯出
    print("自動擷取未取得資料（CWA 表格為 JS 動態載入）。", file=sys.stderr)
    print("請手動：", file=sys.stderr)
    print(f"  1. 開啟 {CWA_URL}", file=sys.stderr)
    print("  2. 複製表格 → 貼到 Excel/CSV 存檔", file=sys.stderr)
    print("  3. 執行：python scripts/academic/import_cwa_publications_csv.py <你的.csv>", file=sys.stderr)


if __name__ == "__main__":
    main()
