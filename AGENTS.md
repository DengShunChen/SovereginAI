# 氣象主權 AI 語料庫 - Agent 任務清單

本文件定義各階段 Agent 的任務與完成條件，用於追蹤語料流程是否就緒。

## 一鍵檢查

執行以下指令可快速確認所有 Agent 是否完成任務：

```bash
python scripts/check_corpus_readiness.py && python scripts/check_training_data.py
```

若語料就緒檢查顯示 **6/6 項通過**，且訓練資料總計 > 0，則表示各階段任務已就緒。

---

## Agent 清單

| Agent | 任務 | 完成條件 | 驗證指令 |
|-------|------|----------|----------|
| **官方語料擷取** | 擷取 CWA 預報／警報 | `data/corpus/weather/official/daily/` 或 `official/alerts/` 存在 `.jsonl` | 檢查目錄 |
| **學術語料擷取** | 擷取論文／CWA 出版品 | `data/corpus/weather/academic/` 下有 `.jsonl` | 檢查目錄 |
| **社群語料擷取** | 擷取 PTT／社群來源 | `data/corpus/weather/social/` 下有 `.jsonl` | 檢查目錄 |
| **前處理** | 去重、過濾、產出 `preprocessed.jsonl` | `data/processed/preprocessed.jsonl` 存在 | 檢查檔案 |
| **訓練切分** | 產出 train/valid/test.jsonl | `data/processed/for_training/` 下有 `train.jsonl`、`valid.jsonl`、`test.jsonl` | `python scripts/check_training_data.py` |
| **語料就緒** | 滿足氣象主權 AI 訓練門檻 | 語料就緒檢查 6/6 項通過 | `python scripts/check_corpus_readiness.py` |
| **發布（選用）** | 推送到 Hugging Face | 依需手動執行 | `python scripts/publish_to_huggingface.py` |
| **A100 訓練（Gemma 4）** | 8×A100 LoRA 微調 Gemma 4 12B | `models/sovereign-weather-lora-gemma4-12b/` 有 adapter | `sbatch scripts/train/train_lora_a100.slurm` |

---

## 各 Agent 詳細說明

### 1. 官方語料擷取 Agent

- **腳本**：`scripts/cwa_fetch/fetch_weather_summary.py`、`fetch_weekly_forecast.py`、`fetch_alerts.py`、`fetch_cwa_knowledge.py`
- **一鍵執行**：`./scripts/run_collect_and_prepare.sh`（含官方擷取）
- **完成條件**：`data/corpus/weather/official/daily/` 或 `official/alerts/` 或 `official/knowledge/` 下至少一個 `.jsonl` 且有內容
- **門檻**：官方語料 ≥ 30 筆（語料就緒檢查會驗證）

### 2. 學術語料擷取 Agent

- **腳本**：`scripts/academic/generate_sample_academic.py`、`fetch_cwa_publications.py`、`import_cwa_publications_csv.py`、`fetch_data_gov_tw_metadata.py`
- **完成條件**：`data/corpus/weather/academic/thesis/` 或 `academic/tech_reports/` 下至少一個 `.jsonl`
- **注意**：需確認 `config/allowed_corpus_sources.txt` 含 `academic_thesis`、`cwa_tech_report` 等

### 3. 社群語料擷取 Agent

- **腳本**：`scripts/social/fetch_ptt_ty_research.py`、`fetch_facebook_cwa.py`
- **完成條件**：`data/corpus/weather/social/ptt_ty_research/` 等下有 `.jsonl`
- **注意**：需確認 `config/allowed_corpus_sources.txt` 含 `ptt_ty_research` 等

### 4. 前處理 Agent

- **腳本**：`scripts/prepare_training/preprocess.py`
- **完成條件**：`data/processed/preprocessed.jsonl` 存在且非空
- **常用參數**：`--vocab-stats` 顯示詞表統計

### 5. 訓練切分 Agent

- **腳本**：`scripts/prepare_training/build_instruction_dataset.py`
- **完成條件**：`data/processed/for_training/train.jsonl` 存在且 train 筆數 ≥ 10
- **驗證**：`python scripts/check_training_data.py`

### 6. 語料就緒 Agent（總體驗證）

- **腳本**：`scripts/check_corpus_readiness.py`
- **完成條件**：以下 6 項全部通過
  - 語料總筆數 ≥ 100
  - 官方語料（預報/概況/警報）≥ 30
  - 詞表覆蓋率 ≥ 20%
  - 無簡體污染
  - 訓練集筆數 ≥ 10
  - 日期跨度 ≥ 7 天

### 7. 發布 Agent（選用）

- **腳本**：`scripts/publish_to_huggingface.py`
- **完成條件**：依需求手動執行，無強制門檻

### 8. A100 訓練 Agent（Gemma 4 12B）

- **腳本**：`scripts/train/train_lora_a100.slurm`、`scripts/train/train_lora_a100.sh`、`scripts/train/train_lora_sft.py`
- **硬體**：Slurm 單節點 `--gres=gpu:8`（A100 80G × 8；CUDA，與 Mac MLX 路徑分開）
- **下載**：compute node source `~/.proxy`（localhost:8888）；HF 權重先 prefetch 再 torchrun
- **完成條件**：`models/sovereign-weather-lora-gemma4-12b/` 存在 LoRA adapter
- **建議門檻**：train ≥ 1000 筆再練 12B（上面 6 項語料檢查仍是 pipeline smoke test，≥ 10 不夠）
- **推理**：`python scripts/train/generate_sft.py --prompt "請說明焚風現象"`

---

## 建議流程

1. 先執行：`./scripts/run_collect_and_prepare.sh`（官方擷取 → 前處理 → 切分）
2. 若需學術／社群：執行對應擷取腳本，再執行 `preprocess.py` → `build_instruction_dataset.py`
3. 定期執行：`python scripts/check_corpus_readiness.py` 檢視各 Agent 完成狀態
4. 語料就緒後：`sbatch scripts/train/train_lora_a100.slurm`（Gemma 4 12B LoRA；下載走 `~/.proxy`）
5. 若要自動發布至 Hugging Face Hub：在 `config/.env` 設定 `HF_REPO_ID=使用者名/repo 名稱`，一鍵流程完成後會自動 push（增量模式）
