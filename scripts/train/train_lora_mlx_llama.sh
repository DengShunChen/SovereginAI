#!/usr/bin/env bash
# 氣象主權 AI — 非中國基礎模型微調（Llama 3.2，Meta USA）
# 使用 mlx-community/Llama-3.2-1B-Instruct-4bit 作為基礎模型
# 自專案根目錄執行：./scripts/train/train_lora_mlx_llama.sh

set -e
cd "$(dirname "$0")/../.."

DATA_DIR="${DATA_DIR:-data/processed/for_training}"
ADAPTER_DIR="${ADAPTER_DIR:-models/sovereign-weather-lora-llama}"
MODEL="${MODEL:-mlx-community/Llama-3.2-1B-Instruct-4bit}"  # Meta（美國），非中國
BATCH_SIZE="${BATCH_SIZE:-2}"
ITERS="${ITERS:-500}"
LR="${LR:-1e-5}"

# 檢查訓練資料
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "錯誤：找不到訓練資料 $DATA_DIR/train.jsonl"
    echo "請先執行：./scripts/run_collect_and_prepare.sh"
    exit 1
fi

# 合併關鍵術語 Q&A 強化語料（修正焚風等術語的錯誤聯想）
python scripts/prepare_training/add_term_qa_to_train.py 2>/dev/null || true

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl" | tr -d ' ')
echo "訓練資料筆數：$TRAIN_COUNT"
if [ "$TRAIN_COUNT" -lt 10 ]; then
    echo "警告：訓練筆數過少（< 10），建議擴充語料後再訓練"
    read -p "是否繼續？(y/N) " -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
fi

echo "=== 氣象主權 AI LoRA 微調（Llama 3.2 基礎）==="
echo "  基礎模型：$MODEL（Meta，美國）"
echo "  資料：$DATA_DIR"
echo "  適配器輸出：$ADAPTER_DIR"
echo "  迭代：$ITERS，批次：$BATCH_SIZE，學習率：$LR"
echo ""

# 建立輸出目錄
mkdir -p "$ADAPTER_DIR"

# 使用 mlx_lm lora 進行 LoRA 微調
python -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --batch-size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --learning-rate "$LR"

echo ""
echo "訓練完成。適配器已儲存至：$ADAPTER_DIR"
echo "推理範例：python -m mlx_lm.generate --model $MODEL --adapter-path $ADAPTER_DIR --prompt '請說明焚風現象'"
