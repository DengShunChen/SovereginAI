# 氣象主權 AI — M2 Mac 本機訓練

本目錄提供在 **Apple Silicon（M2/M3）MacBook Pro** 上微調氣象主權 AI 的腳本與說明。

## 前置需求

### 1. 語料就緒

確認訓練資料已產生：

```bash
python scripts/check_corpus_readiness.py && python scripts/check_training_data.py
```

若尚未就緒，執行一鍵流程：

```bash
./scripts/run_collect_and_prepare.sh
```

### 2. 安裝訓練依賴（MLX）

**注意**：需使用 `[train]` 才包含 LoRA 微調功能。

```bash
pip install "mlx-lm[train]"
```

或使用專案提供的依賴檔：

```bash
pip install -r requirements-train.txt
```

### 3. 硬體建議

| 機型 | 建議模型規模 | 批次大小 | 備註 |
|------|-------------|----------|------|
| M2 MacBook Pro（8GB） | 0.5B–1.5B | 2 | 使用 4-bit 量化模型 |
| M2 MacBook Pro（16GB+） | 1.5B–3B | 4 | 可嘗試較大模型 |
| M3/M3 Pro | 同上或更大 | 4–8 | 訓練速度較快 |

## 訓練方式

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

## 模型選擇

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

## 訓練完成後

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

### 與 Ollama 整合（可選）

可將適配器合併回基礎模型，再匯入 Ollama 使用。詳見 [mlx-examples](https://github.com/ml-explore/mlx-examples)。

## 資料格式

訓練資料位於 `data/processed/for_training/`，格式為 JSONL，每行：

```json
{"text": "焚風為一種出現在山脈背風面之乾熱風..."}
```

此格式與 mlx-lm 預設 SFT 格式相容，無需額外轉換。

## 常見問題

**Q: 訓練時記憶體不足？**  
A: 降低 `--batch-size` 至 1，或改用更小的模型（如 0.5B）。

**Q: 訓練速度很慢？**  
A: M2 訓練 500 迭代約需數分鐘至十多分鐘，屬正常。可減少 `--iters` 先驗證流程。

**Q: 語料只有數百筆夠嗎？**  
A: 可用於驗證流程與初步測試。若要較佳效果，建議累積至 1000+ 筆（每日執行 `run_collect_and_prepare.sh` 可逐步擴充）。
