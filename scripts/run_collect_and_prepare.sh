#!/usr/bin/env bash
# 依序：擷取官方語料 → 前處理 → 產出訓練用切分 →（可選）發布至 Hugging Face Hub
# 自專案根目錄執行：./scripts/run_collect_and_prepare.sh
# 外網需 ~/.proxy（腳本會自動 source）。需 CWA_API_AUTH_KEY（config/.env）。
# 若需含學術／社群語料，請先執行 scripts/academic/、scripts/social/ 下對應擷取腳本，
# 並確認 config/allowed_corpus_sources.txt 含 academic_thesis、ptt_ty_research 等。
# 若設定 HF_REPO_ID（config/.env），完成後會自動 push 至 Hub。

set -euo pipefail
cd "$(dirname "$0")/.."

# 外網：~/.proxy（localhost:8888 → proxy.cwa.gov.tw）
# shellcheck source=train/load_proxy.sh
source "$(dirname "$0")/train/load_proxy.sh"
load_cluster_proxy

# 載入 .env（若有），使 CWA_API_AUTH_KEY、HF_REPO_ID、HF_TOKEN 等生效
if [ -f config/.env ]; then
    set -a
    # shellcheck source=config/.env
    . config/.env
    set +a
fi
# 非互動 shell 不會跑 ~/.bashrc（開頭有 return）；補抓 export
if [[ -z "${CWA_API_AUTH_KEY:-}" && -f "${HOME}/.bashrc" ]]; then
    CWA_LINE=$(grep -E '^export CWA_API_AUTH_KEY=' "${HOME}/.bashrc" || true)
    eval "$CWA_LINE"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
    HF_LINE=$(grep -E '^export HUGGING_FACE_HUB_TOKEN=' "${HOME}/.bashrc" || true)
    eval "$HF_LINE"
    export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"
# 走叢集 proxy 時常見 CERTIFICATE_VERIFY_FAILED
if [[ -n "${https_proxy:-}" ]]; then
    export CWA_SSL_VERIFY="${CWA_SSL_VERIFY:-0}"
fi
echo "PYTHON=$PYTHON  proxy=${https_proxy:-unset}  CWA_SSL_VERIFY=${CWA_SSL_VERIFY:-1}"

if [[ -z "${CWA_API_AUTH_KEY:-}" ]]; then
    echo "無 CWA_API_AUTH_KEY，跳過 API 擷取（步驟 1–4）。官網知識仍會抓。"
    echo "有 key 後放到 config/.env 再重跑即可補官方預報／警報。"
else
    echo "=== 1. 擷取天氣概況 ==="
    "$PYTHON" scripts/cwa_fetch/fetch_weather_summary.py

    echo "=== 2. 擷取一週／單點預報 ==="
    "$PYTHON" scripts/cwa_fetch/fetch_weekly_forecast.py || true

    echo "=== 2b. 擷取鄉鎮預報 F-D0047 ==="
    "$PYTHON" scripts/cwa_fetch/fetch_township_forecast.py || true

    echo "=== 3. 擷取警特報 ==="
    "$PYTHON" scripts/cwa_fetch/fetch_alerts.py || true

    echo "=== 4. 擷取額外 API 資料（警特報、地震、潮汐、天文等） ==="
    "$PYTHON" scripts/cwa_fetch/fetch_cwa_extra_api.py || true
fi

echo "=== 5. 擷取官網知識與天文 ==="
"$PYTHON" scripts/cwa_fetch/fetch_cwa_knowledge.py || true
"$PYTHON" scripts/cwa_fetch/fetch_cwa_popular_science.py || true

echo "=== 6. 前處理語料 ==="
"$PYTHON" scripts/prepare_training/preprocess.py --min-len 50 --vocab-stats --drop-ui-junk

echo "=== 7. 產出訓練用切分 ==="
"$PYTHON" scripts/prepare_training/build_instruction_dataset.py --format plain

echo "=== 7b. 合併術語 Q&A 強化語料 ==="
"$PYTHON" scripts/prepare_training/add_term_qa_to_train.py || true

echo "完成。訓練用資料在 data/processed/for_training/"

# 若設定 HF_REPO_ID，則推送到 Hugging Face Hub
if [ -n "${HF_REPO_ID:-}" ]; then
    echo "=== 8. 發布至 Hugging Face Hub ==="
    "$PYTHON" scripts/publish_to_huggingface.py --repo_id "$HF_REPO_ID" --incremental || true
fi
