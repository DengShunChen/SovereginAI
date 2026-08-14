# 氣象主權 AI — 訓練

兩條路徑，**不要混在同一個 venv**：

| 路徑 | 硬體 | 基礎模型 | 腳本 |
|------|------|----------|------|
| **CUDA LoRA（主）** | A100 80G × 8 / node | `google/gemma-4-12B-it` | `uv` + `sbatch scripts/train/train_lora_a100.slurm` |
| MLX LoRA | M2/M3 Mac | Qwen 1.5B / Llama 3.2 1B（4-bit） | `./scripts/train/train_lora_mlx.sh` |

---

## A100：Gemma 4 12B LoRA

### 1. 語料

```bash
python scripts/check_corpus_readiness.py && python scripts/check_training_data.py
```

未就緒：`./scripts/run_collect_and_prepare.sh`

12B LoRA 建議 train ≥ **1000** 筆。`AGENTS.md` 的 ≥ 10 只是 pipeline smoke test。

`plain` / `instruction` / `messages` 都會轉成 chat，且只對 assistant token 算 loss。Gemma 4 的空 thought channel 由訓練腳本直接拼進 prompt，不靠前綴比對。

### 2. 訓練環境（uv，login node 建一次）

**還沒建。** 不要 conda、不要系統 python pip。叢集已有 `uv`（`~/.local/bin/uv`）。

Gemma 4 為 Apache 2.0，不需 gated 授權。Hub 限流時可設 `HF_TOKEN`。

```bash
# login node，走 ~/.proxy 下載 torch cu124 + transformers/peft
./scripts/train/setup_uv_env.sh
```

等價手動：

```bash
source ~/.proxy
uv python install 3.12
uv sync --extra cu124
```

Mac MLX 另開 extra，跟 CUDA 互斥：`uv sync --extra mlx`。

### 3. 開練（Slurm）

這是 Slurm 叢集。下載走 `~/.proxy`（ssh tunnel `localhost:8888` → `proxy.cwa.gov.tw`）。作業裡會自動 source；`:8888` 已開就 reuse，不會被 `ssh -Nf` 撞 port 打死。

```bash
# 專案根（先有 .venv）
mkdir -p logs
sbatch scripts/train/train_lora_a100.slurm

# 短測
sbatch -p ftest --time=04:00:00 scripts/train/train_lora_a100.slurm
```

預設：`--partition=normal --gres=gpu:8 --nodes=1 --mem=0`（整節點 8×A100 80G）。log：`logs/gemma4_12b_<jobid>.out`。

已在 GPU 節點上（`srun --pty`）可直接：

```bash
source ~/.proxy   # 或讓腳本自己 load
./scripts/train/train_lora_a100.sh
```

torchrun 前會先 `snapshot_download` 一次，避免 8 rank 同時打 HF。

預設（8 GPU）：

```
model     google/gemma-4-12B-it
lora      r=64  alpha=128  q/k/v/o/gate/up/down（語言塔，不含 vision/audio）
batch     micro=1  accum=1  → global 8
seq       2048  packing + completion-only
lr        2e-4 cosine  warmup 3%
epochs    5
dtype     bf16 + gradient checkpointing
out       models/sovereign-weather-lora-gemma4-12b/
```

環境變數可覆寫：

```bash
NUM_GPUS=8 BATCH_SIZE=1 GRAD_ACCUM=1 EPOCHS=5 LR=2e-4 \
MODEL=google/gemma-4-12B-it \
ADAPTER_DIR=models/sovereign-weather-lora-gemma4-12b \
./scripts/train/train_lora_a100.sh
```

單卡 debug：`NUM_GPUS=1 ./scripts/train/train_lora_a100.sh`

格式自檢（login node 可跑，不載入權重）：

```bash
.venv/bin/python scripts/train/train_lora_sft.py --format-check --max-train-rows 8 --min-train 1
```

### 4. 推理

```bash
python3 scripts/train/generate_sft.py \
  --adapter-path models/sovereign-weather-lora-gemma4-12b \
  --prompt "請說明焚風現象"
```

現有 `scripts/evaluate/*` 仍走 MLX，A100 上不要跑那些。

OOM：先砍 `MAX_LENGTH=1024`。12B LoRA + 262K 詞表，micro batch 必須是 1；不要把 `BATCH_SIZE` 拉回 8。

語料太少硬上 31B：改 `MODEL=google/gemma-4-31B-it` 並加 FSDP，這專案不建議當第一槍。

---

## M2 Mac：MLX LoRA（1B–3B）

本機 Apple Silicon 微調。需 `[train]` extra。

```bash
pip install -r requirements-train.txt
```

| 機型 | 建議模型規模 | 批次大小 | 備註 |
|------|-------------|----------|------|
| M2 MacBook Pro（8GB） | 0.5B–1.5B | 2 | 使用 4-bit 量化模型 |
| M2 MacBook Pro（16GB+） | 1.5B–3B | 4 | 可嘗試較大模型 |
| M3/M3 Pro | 同上或更大 | 4–8 | 訓練速度較快 |

### 方式一：使用腳本（推薦）

**Qwen 基礎（預設）**：
```bash
./scripts/train/train_lora_mlx.sh
```

**Llama 基礎（非中國）**：
```bash
./scripts/train/train_lora_mlx_llama.sh
```

可透過環境變數調整：

```bash
# Qwen 版本
DATA_DIR=data/processed/for_training \
ADAPTER_DIR=models/sovereign-weather-lora \
MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
BATCH_SIZE=2 ITERS=500 LR=1e-5 \
./scripts/train/train_lora_mlx.sh

# Llama 版本（非中國基礎）
DATA_DIR=data/processed/for_training \
ADAPTER_DIR=models/sovereign-weather-lora-llama \
MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit \
BATCH_SIZE=2 ITERS=500 LR=1e-5 \
./scripts/train/train_lora_mlx_llama.sh
```

### 方式二：直接呼叫 mlx_lm lora

```bash
python -m mlx_lm lora \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --train \
    --data data/processed/for_training \
    --adapter-path models/sovereign-weather-lora \
    --batch-size 2 \
    --iters 500 \
    --learning-rate 1e-5
```

（注意：新版 mlx-lm 使用 `python -m mlx_lm lora`，不再支援 `--lora-layers`）

## 模型選擇（MLX）

### 氣象主權 AI 支援兩種基礎模型

| 腳本 | 基礎模型 | 來源 | 大小 | 說明 |
|------|----------|------|------|------|
| `train_lora_mlx.sh` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 阿里巴巴（中國） | 約 1.5B | 中文表現佳（預設） |
| `train_lora_mlx_llama.sh` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | Meta（美國） | 約 1B | **非中國**基礎模型 |

### 其他選用模型

| 模型 | 大小 | 說明 |
|------|------|------|
| `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | 約 0.5B | 最小、M2 8GB 可跑 |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | 約 3B | 需 16GB+ 記憶體 |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 約 3B | Meta（美國），較大規模 |

更多 MLX 社群模型：[Hugging Face mlx-community](https://huggingface.co/mlx-community)

## 訓練完成後（MLX）

### 泛化能力評估（建議）

微調完成後，建議比較基礎模型與微調模型的泛化能力，確認未因微調造成一般能力退化：

```bash
# Prompt 比較（定性）
python scripts/evaluate/compare_base_vs_finetuned.py

# Perplexity 比較（定量）
python scripts/evaluate/compare_perplexity.py
```

詳見 `scripts/evaluate/README.md`。

### 推理測試

```bash
# Qwen 版本
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --adapter-path models/sovereign-weather-lora \
    --prompt "請說明焚風現象"

# Llama 版本（非中國基礎）
python -m mlx_lm.generate \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --adapter-path models/sovereign-weather-lora-llama \
    --prompt "請說明焚風現象"
```

### 發布至 Hugging Face

訓練完成後可將模型發布至 Hugging Face Hub：

```bash
# 先登入
huggingface-cli login

# 合併 LoRA 與基礎模型後上傳（推薦，產生完整可獨立使用的模型）
python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-ai

# 或僅上傳 LoRA 適配器（較小，使用者需搭配基礎模型）
python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-lora --mode adapter
```

可在 `config/.env` 設定 `HF_MODEL_REPO=dschen/sovereign-weather-ai` 以省略 `--repo_id`。

A100 Gemma 4 12B 適配器請先指定路徑與基礎模型：

```bash
python scripts/publish_model_to_huggingface.py \
  --repo_id dschen/sovereign-weather-ai-gemma4-12b \
  --adapter_path models/sovereign-weather-lora-gemma4-12b \
  --model google/gemma-4-12B-it \
  --mode adapter
```

（`publish_model_to_huggingface.py` 的 fuse 路徑仍走 MLX；CUDA 權重請用 `--mode adapter`。）

### 與 Ollama 整合（可選）

可將適配器合併回基礎模型，再匯入 Ollama 使用。詳見 [mlx-examples](https://github.com/ml-explore/mlx-examples)。

## 資料格式

訓練資料位於 `data/processed/for_training/`，格式為 JSONL，每行：

```json
{"text": "焚風為一種出現在山脈背風面之乾熱風..."}
```

或 instruction：

```json
{"instruction": "請根據以下資料寫今日天氣概況。", "input": "天氣資料", "output": "..."}
```

A100 腳本兩種都吃；MLX 預設吃 `{"text": "..."}`。

## 常見問題

**Q: A100 訓練時記憶體不足？**  
A: `MAX_LENGTH=1024`。12B + 262K 詞表不要把 `BATCH_SIZE` 拉回 8。

**Q: Mac 訓練時記憶體不足？**  
A: 降低 `--batch-size` 至 1，或改用更小的模型（如 0.5B）。

**Q: 語料只有數百筆夠嗎？**  
A: Mac 1B 可驗證流程。12B 建議 ≥ 1000 筆（每日執行 `run_collect_and_prepare.sh`）。
