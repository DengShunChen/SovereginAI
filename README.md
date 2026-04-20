# 台灣氣象語料庫

結構化語料目錄、CWA 開放資料擷取腳本、JSONL 語料與詞表，供後續前處理與訓練使用。

**語料來源總覽**：見 [docs/CWA_CORPUS_SOURCES.md](docs/CWA_CORPUS_SOURCES.md)，彙整 CWA 官網所有可作為台灣氣象主權 AI 訓練的語料。

## 目錄結構

```
data/corpus/weather/
  official/           # 第一階段：CWA 預報／警報
  │   daily/          #   每日概況、一週預報（official_daily_*.jsonl, official_weekly_*.jsonl）
  │   alerts/         #   警特報（official_alerts_*.jsonl）
  academic/           # 第二階段：學術／技術文獻
  │   thesis/         #   論文摘要（sample_thesis_abstracts.jsonl, thesis_abstracts_expanded.jsonl）
  │   tech_reports/   #   CWA 出版品列表、政府開放平臺詮釋資料（cwa_publications_list.jsonl 等）
  │   conference/     #   研討會（預留）
  social/             # 第三階段：社群／論壇（台灣／繁體來源，需授權）
  │   ptt_ty_research/ #   PTT 大氣板（ptt_ty_research_*.jsonl）
  hpc_logs/            # 第四階段（預留）：WRF／Lustre／GPU 日誌
data/processed/
  preprocessed.jsonl  # 前處理後語料（preprocess.py 產出）
  for_training/       # 訓練用切分（train/valid/test.jsonl, schema.json）
scripts/
  cwa_fetch/           # 官方 CWA API 擷取（fetch_weather_summary, fetch_weekly_forecast, fetch_alerts）
  academic/            # 學術擷取與匯入（generate_sample_academic, import_cwa_publications_csv, fetch_cwa_publications, fetch_data_gov_tw_metadata）
  social/              # 社群擷取（fetch_ptt_ty_research, fetch_facebook_cwa）
  prepare_training/    # 前處理與格式轉換（preprocess, build_instruction_dataset）
  check_training_data.py    # 檢查 for_training 筆數與範例
  check_corpus_readiness.py # 語料就緒檢查（氣象主權 AI）
  run_collect_and_prepare.sh # 一鍵：官方擷取 → 前處理 → 切分
vocab/
  weather_terms_tw.txt      # 台灣氣象專有名詞
  mainland_terms_filter.txt # 中國用語過濾（前處理 --exclude-mainland-usage）
  mainland_to_tw_replace.txt# 中國用語→台灣用語替換（前處理 --normalize-mainland-to-tw）
config/
  .env.example              # 複製為 .env，填入 CWA_API_AUTH_KEY 等
  allowed_corpus_sources.txt # 允許來源（前處理依此過濾；須含 F-C0032-001、academic_thesis、ptt_ty_research 等）
```

## 環境與授權

1. 複製 `config/.env.example` 為 `config/.env`（或專案根目錄 `.env`）。
2. 至 [氣象資料開放平臺](https://opendata.cwa.gov.tw/user/authkey) 申請授權碼，寫入 `CWA_API_AUTH_KEY=`。
3. 若擷取時出現 `CERTIFICATE_VERIFY_FAILED`（如 Missing Subject Key Identifier），在 `.env` 加上 `CWA_SSL_VERIFY=0` 可略過 SSL 驗證（僅限本機除錯）。
4. 安裝依賴：`pip install -r requirements.txt`。

## 第一階段：官方資料擷取

自專案根目錄執行（需先設定 `CWA_API_AUTH_KEY`）。亦可一次跑完整流程：`./scripts/run_collect_and_prepare.sh`

```bash
# 天氣概況（F-C0044-001）
python scripts/cwa_fetch/fetch_weather_summary.py

# 一週／單點預報（F-C0032-003_006、F-C0032-001）
python scripts/cwa_fetch/fetch_weekly_forecast.py

# 警特報（自天氣概況解析）
python scripts/cwa_fetch/fetch_alerts.py
```

語料寫入 `data/corpus/weather/official/daily/` 與 `official/alerts/`，格式為 JSONL，每筆含 `title`、`content`、`date`、`source`、`type`。天氣概況擷取時會依縣市產出可讀文本（每縣市一筆），一週預報亦為每縣市一筆，單次擷取即可累積數十筆官方語料。

## 前處理與訓練用產出

前處理**預設排除簡體中文**並可依 `config/allowed_corpus_sources.txt` 僅保留允許來源，避免中國大陸或簡體中文污染語料。詳見 `data/corpus/weather/social/README.md`。

```bash
# 1. 前處理：讀取語料、去重、長度過濾、排除簡體／僅允許來源，輸出 preprocessed.jsonl
python scripts/prepare_training/preprocess.py [--vocab-stats]
# 不使用來源名單：--no-allowed-sources；不排除簡體：--no-exclude-simplified
# 排除含過多中國用語的筆數（清洗 PTT 等社群）：--exclude-mainland-usage
# 將中國用語替換為台灣用語（不刪筆數）：--normalize-mainland-to-tw
# 詞表：vocab/mainland_terms_filter.txt、vocab/mainland_to_tw_replace.txt

# 2. 格式轉換與切分：產出 train/valid/test.jsonl 至 data/processed/for_training/
python scripts/prepare_training/build_instruction_dataset.py --format plain
# 或 --format instruction 產出指令微調格式
```

訓練程式約定從 `data/processed/for_training/` 讀取；格式見該目錄下 `schema.json`。

**驗證訓練用資料**：`python scripts/check_training_data.py` 可檢查 train/valid/test 筆數與一筆範例。

**語料就緒檢查（氣象主權 AI）**：`python scripts/check_corpus_readiness.py` 會檢查語料總筆數、官方／學術分布、台灣氣象詞表覆蓋率、內容長度、重複與簡體污染、訓練用切分，並產出「就緒評估」清單（如：語料總筆數 >= 100、官方語料 >= 30、詞表覆蓋率 >= 20%、無簡體污染、訓練集 >= 10、日期跨度 >= 7 天）。建議在擴充語料或重新前處理後定期執行，以確保語料足以支撐氣象主權 AI 訓練。

## 發布到 Hugging Face

可將語料發布到 Hugging Face Hub，供遠端載入或分享。**首次發布**會掃描 `data/corpus/weather` 下所有 `.jsonl` 並上傳；**後續增量**使用 `--incremental` 會從 Hub 載入既有 dataset、合併本地新資料後再 push（以 `date`、`source`、`title` 去重）。

本專案維護者公開的 Models／Datasets 見 [huggingface.co/dschen](https://huggingface.co/dschen)（例如語料集 `dschen/sovereign-weather-corpus`）。

```bash
# 安裝依賴（若尚未安裝）
pip install datasets huggingface_hub

# 登入 Hugging Face（首次需執行）
huggingface-cli login

# 首次發布（完整）
python scripts/publish_to_huggingface.py --repo_id dschen/sovereign-weather-corpus

# 後續只推送新增／變更（增量）
python scripts/publish_to_huggingface.py --repo_id dschen/sovereign-weather-corpus --incremental

# 私人 repo
python scripts/publish_to_huggingface.py --repo_id dschen/sovereign-weather-corpus --private
```

每次 push 會產生新 revision，他人可固定版本或使用 `main` 取得最新語料。

## 下一步

1. **每日定時擷取**：以 cron 或 `schedule` 每日執行 `./scripts/run_collect_and_prepare.sh`（或僅執行 `fetch_weather_summary.py`、`fetch_weekly_forecast.py`），累積更多語料。每次擷取會產出多筆縣市級可讀語料，長期可顯著增加官方語料量。
2. **接訓練程式**：從 `data/processed/for_training/train.jsonl`、`valid.jsonl`、`test.jsonl` 讀取；`plain` 格式每行為 `{"text": "..."}`，`instruction` 格式為 `instruction`/`input`/`output`。可搭配 Hugging Face `datasets` 或自訂 DataLoader。**M2 Mac 本機 LoRA 微調**：見 `scripts/train/README.md`，使用 Apple MLX 執行 `./scripts/train/train_lora_mlx.sh`。
3. **擴充語料**：學術／技術語料**以 CWA 出版品為主**（見 `data/corpus/weather/academic/tech_reports/README_CWA.md`）：官網表格建議手動匯出為 CSV，再執行 `python scripts/academic/import_cwa_publications_csv.py your.csv`；或執行 `fetch_cwa_publications.py` 嘗試自動擷取。勿使用自行假造語料。記得在 `config/allowed_corpus_sources.txt` 加入 `cwa_tech_report` 等來源，前處理才會納入。

## 第二～四階段

- **第二階段（學術）**：見 `data/corpus/weather/academic/README.md`。
- **第三階段（社群／官方社群）**：見 `data/corpus/weather/social/README.md`；僅收錄台灣／繁體中文來源，排除中國大陸與簡體中文。可執行 `scripts/social/fetch_ptt_ty_research.py`（PTT 大氣板）、`scripts/social/fetch_facebook_cwa.py`（報天氣臉書，需 `FB_ACCESS_TOKEN`）擷取語料，僅在取得授權或合理使用範圍內使用。
- **第四階段（HPC／WRF）**：見 `data/corpus/weather/hpc_logs/README.md`，可匯入 WRF 錯誤、Lustre I/O、GPU 日誌。

## 專案整合與一致性

- **單一語料格式**：所有階段（official / academic / social）的 JSONL 均含 `title`、`content`、`date`、`source`、`type`。前處理 `preprocess.py` 會讀取 `data/corpus/weather/**/*.jsonl`（排除檔名含 manifest 者），並依 `config/allowed_corpus_sources.txt` 過濾來源。
- **建議流程**：  
  1. 官方：`./scripts/run_collect_and_prepare.sh`（或個別執行 cwa_fetch → preprocess → build_instruction_dataset）。  
  2. 若需含學術／社群：先執行 `scripts/academic/`、`scripts/social/` 下對應擷取腳本，確認 `allowed_corpus_sources.txt` 已含 `academic_thesis`、`cwa_tech_report`、`ptt_ty_research` 等，再執行 preprocess → build_instruction_dataset。  
  3. 定期執行 `python scripts/check_corpus_readiness.py` 檢視語料就緒度（依階段統計官方／學術／社群筆數）。
- **詞表與過濾**：`vocab/weather_terms_tw.txt` 供就緒檢查與標記；`vocab/mainland_terms_filter.txt`、`mainland_to_tw_replace.txt` 供前處理排除或替換中國用語（見 `data/corpus/weather/social/README.md`）。

## 授權與隱私

- 語料僅使用中央氣象署開放資料 API，不爬取官網 HTML。
- `.env` 與授權碼已列入 `.gitignore`，請勿提交。
