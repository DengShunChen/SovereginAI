#!/usr/bin/env bash
# 依序：擷取官方語料 → 前處理 → 產出訓練用切分 →（可選）發布至 Hugging Face Hub
# 自專案根目錄執行：./scripts/run_collect_and_prepare.sh
# 若需含學術／社群語料，請先執行 scripts/academic/、scripts/social/ 下對應擷取腳本，
# 並確認 config/allowed_corpus_sources.txt 含 academic_thesis、ptt_ty_research 等。
# 若設定 HF_REPO_ID（config/.env），完成後會自動 push 至 Hub。

set -e
cd "$(dirname "$0")/.."

# 載入 .env（若有），使 HF_REPO_ID、HF_TOKEN 等生效
if [ -f config/.env ]; then
    set -a
    # shellcheck source=config/.env
    . config/.env
    set +a
fi

echo "=== 1. 擷取天氣概況 ==="
python scripts/cwa_fetch/fetch_weather_summary.py

echo "=== 2. 擷取一週／單點預報 ==="
python scripts/cwa_fetch/fetch_weekly_forecast.py || true

echo "=== 3. 擷取警特報 ==="
python scripts/cwa_fetch/fetch_alerts.py || true

echo "=== 4. 擷取額外 API 資料（警特報、地震、潮汐、天文等） ==="
python scripts/cwa_fetch/fetch_cwa_extra_api.py || true

echo "=== 5. 擷取官網知識與天文 ==="
python scripts/cwa_fetch/fetch_cwa_knowledge.py || true
python scripts/cwa_fetch/fetch_cwa_popular_science.py || true

echo "=== 6. 前處理語料 ==="
python scripts/prepare_training/preprocess.py --vocab-stats

echo "=== 7. 產出訓練用切分 ==="
python scripts/prepare_training/build_instruction_dataset.py --format plain

echo "完成。訓練用資料在 data/processed/for_training/"

# 若設定 HF_REPO_ID，則推送到 Hugging Face Hub
if [ -n "${HF_REPO_ID:-}" ]; then
    echo "=== 8. 發布至 Hugging Face Hub ==="
    python scripts/publish_to_huggingface.py --repo_id "$HF_REPO_ID" --incremental || true
fi
