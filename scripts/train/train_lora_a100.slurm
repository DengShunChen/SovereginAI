#!/bin/bash
# 氣象主權 AI — Slurm：單節點 A100 80G × 8，Gemma 4 12B LoRA
#
# 在專案根提交：
#   sbatch scripts/train/train_lora_a100.slurm
#   sbatch -p ftest --time=04:00:00 scripts/train/train_lora_a100.slurm
#
# 下載走 ~/.proxy（login/compute 皆 source；8888 已開就 reuse）
# 前置：login 先跑 ./scripts/train/setup_uv_env.sh（uv --extra cu124）
#SBATCH --job-name=sovereign-gemma4-12b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/gemma4_12b_%j.out
#SBATCH --error=logs/gemma4_12b_%j.out

set -euo pipefail

if [[ -z "${WORK_DIR:-}" ]]; then
    WORK_DIR="${SLURM_SUBMIT_DIR:-}"
fi
if [[ -z "$WORK_DIR" ]]; then
    echo "error: 在專案根執行 sbatch，或 export WORK_DIR=專案根" >&2
    exit 1
fi
cd "$WORK_DIR"
mkdir -p logs

# shellcheck source=load_proxy.sh
source "$WORK_DIR/scripts/train/load_proxy.sh"
load_cluster_proxy

export PATH="${HOME}/.local/bin:${PATH}"
if [[ ! -x "$WORK_DIR/.venv/bin/python" ]]; then
    echo "error: 找不到 $WORK_DIR/.venv 。請先在 login node 跑：" >&2
    echo "  ./scripts/train/setup_uv_env.sh" >&2
    exit 1
fi
export PYTHON="${PYTHON:-$WORK_DIR/.venv/bin/python}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== Slurm $SLURM_JOB_ID ==="
echo "  node=$SLURM_NODELIST  gres=${SLURM_GPUS:-$SLURM_JOB_GPUS}  cpus=$SLURM_CPUS_PER_TASK"
echo "  WORK_DIR=$WORK_DIR"
echo "  PYTHON=$PYTHON  ($(command -v "$PYTHON" || echo "$PYTHON"))"
echo "  http_proxy=${http_proxy:-}"
nvidia-smi -L || true

exec ./scripts/train/train_lora_a100.sh
