#!/usr/bin/env bash
# 氣象主權 AI — A100 80G × N  LoRA 微調（Gemma 4 12B IT）
# 自專案根目錄：./scripts/train/train_lora_a100.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# shellcheck source=load_proxy.sh
source "$SCRIPT_DIR/load_proxy.sh"
load_cluster_proxy

if [ -f config/.env ]; then
    set -a
    # shellcheck source=config/.env
    . config/.env
    set +a
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data/processed/for_training}"
ADAPTER_DIR="${ADAPTER_DIR:-models/sovereign-weather-lora-gemma4-12b}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NUM_GPUS="$SLURM_GPUS_ON_NODE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        NUM_GPUS=8
    fi
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MODEL
export HF_HUB_DISABLE_XET=1

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "錯誤：.venv 沒有 torch（PYTHON=$PYTHON）。請先在 login 跑："
    echo "  ./scripts/train/setup_uv_env.sh"
    exit 1
fi
if ! "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "錯誤：找不到 CUDA。此腳本給 A100 節點用，Mac 請改跑 ./scripts/train/train_lora_mlx.sh"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "錯誤：找不到訓練資料 $DATA_DIR/train.jsonl"
    echo "請先執行：./scripts/run_collect_and_prepare.sh"
    exit 1
fi

"$PYTHON" scripts/prepare_training/add_term_qa_to_train.py 2>/dev/null || true

TRAIN_COUNT="$(wc -l < "$DATA_DIR/train.jsonl" | tr -d ' ')"
echo "訓練資料筆數：$TRAIN_COUNT"
if [ "$TRAIN_COUNT" -lt 10 ]; then
    echo "錯誤：訓練筆數 < 10"
    exit 1
fi
if [ "$TRAIN_COUNT" -lt 1000 ]; then
    echo "警告：筆數 < 1000，12B LoRA 容易過擬合。建議先累積語料。"
fi

GLOBAL_BS=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
echo "=== 氣象主權 AI LoRA（Gemma 4 12B / A100）==="
echo "  模型：$MODEL"
echo "  資料：$DATA_DIR"
echo "  適配器：$ADAPTER_DIR"
echo "  GPU × $NUM_GPUS   micro=$BATCH_SIZE  accum=$GRAD_ACCUM  global_batch=$GLOBAL_BS"
echo "  epochs=$EPOCHS  lr=$LR  max_length=$MAX_LENGTH  lora_r=$LORA_R"
echo "  proxy=${https_proxy:-unset}"
echo ""

mkdir -p "$ADAPTER_DIR"

echo "prefetch $MODEL（走 proxy=${https_proxy:-none}）..."
"$PYTHON" -c "
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['MODEL']))
" || {
    echo "錯誤：模型下載失敗。檢查 ~/.proxy 與網路；Gemma 4 為 Apache 2.0，不需 gated 授權。"
    exit 1
}

if [ "$NUM_GPUS" -le 1 ]; then
    "$PYTHON" scripts/train/train_lora_sft.py \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --adapter-dir "$ADAPTER_DIR" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-length "$MAX_LENGTH" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA"
else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
        scripts/train/train_lora_sft.py \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --adapter-dir "$ADAPTER_DIR" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-length "$MAX_LENGTH" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA"
fi

echo ""
echo "推理範例："
echo "  $PYTHON scripts/train/generate_sft.py --adapter-path $ADAPTER_DIR --prompt '請說明焚風現象'"
