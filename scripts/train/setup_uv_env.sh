#!/usr/bin/env bash
# 用 uv 建立 A100 CUDA 訓練環境（login node 跑，走 ~/.proxy 下載）
#   ./scripts/train/setup_uv_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# shellcheck source=load_proxy.sh
source "$SCRIPT_DIR/load_proxy.sh"
load_cluster_proxy

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    echo "找不到 uv，安裝到 ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv $(uv --version)"
echo "proxy=${https_proxy:-unset}"
echo "==> uv python install 3.12"
uv python install 3.12

echo "==> uv sync --extra cu124"
uv sync --extra cu124

echo "==> 驗證"
uv run --extra cu124 python -c "
import torch, transformers, peft, accelerate
print('python ', __import__('sys').version.split()[0])
print('torch  ', torch.__version__, 'cuda_built', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('transformers', transformers.__version__, 'peft', peft.__version__)
"
echo
echo "完成。.venv 已就緒。Slurm：sbatch scripts/train/train_lora_a100.slurm"
