# 氣象主權 AI — 基礎 vs 微調 泛化能力評估

本目錄提供比較**基礎模型**與**微調後模型**的腳本，用以確認 LoRA 微調沒有造成泛化能力退化。

## 評估方式

### 0. 一鍵執行所有評估（推薦）

自動執行定量、定性、標準基準評估，並更新 `models/MODEL_CARD_LLAMA.md`：

```bash
python scripts/evaluate/run_all_eval.py --model-type llama
```

### 1. 整合定量評估

一次執行多項定量指標：Perplexity（通用／氣象）、準確率、回覆長度、延遲：

```bash
python scripts/evaluate/run_quantitative_eval.py
```

可選參數：`--model-type`（qwen／llama）、`--model`、`--adapter-path`、`--prompts`、`--output`、`--max-tokens`。例如測試 Llama 版：`--model-type llama`。

輸出 JSON 寫入 `models/eval_quantitative.json`，含：
- `perplexity`：general、general_extra、weather 三種文本
- `accuracy`：有預期答案的 6 題（數學、事實、程式）
- `response_length`：平均回覆字元數
- `latency`：平均每題推理時間（秒）

### 2. Prompt 比較（定性）

對多個**非氣象領域**的 prompt 產生回答，比較兩模型輸出品質：

- 一般知識（光合作用、成語、牛頓定律等）
- 數學推理
- 常識
- 程式碼

```bash
python scripts/evaluate/compare_base_vs_finetuned.py
```

可選參數：

- `--model-type`：qwen（預設）或 llama
- `--model`：基礎模型（預設依 model-type）
- `--adapter-path`：LoRA 適配器路徑（預設：qwen=`models/sovereign-weather-lora`，llama=`models/sovereign-weather-lora-llama`）
- `--prompts`：自訂 prompt JSON 檔
- `--max-tokens`：每回覆最大 token 數
- `--temp`：取樣溫度（0=確定性）

輸出會寫入 `models/evaluation_base_vs_finetuned.json`。

### 3. Perplexity 比較（ quantitative）

在通用中文文本上計算 perplexity。若微調後 PPL 顯著升高，可能表示泛化能力退化：

```bash
python scripts/evaluate/compare_perplexity.py
```

可選參數：

- `--model-type`：qwen（預設）或 llama
- `--model`：基礎模型
- `--adapter-path`：LoRA 適配器路徑
- `--text-file`：評估用文本檔（預設使用內建一般中文樣本）
- `--seq-len`：序列長度
- `--batch-size`：批次大小

### 4. 標準基準評估（lm-eval）

使用國際學術標準基準（MMLU、GSM8K、HellaSwag、ARC、TruthfulQA、Winogrande）比較基礎與微調模型：

```bash
# 1. 安裝 lm-eval
pip install lm-eval

# 2. Fuse 微調模型（Qwen 版；Llama 版見 run_all_eval / MODEL_CARD_LLAMA）
python -m mlx_lm fuse --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path models/sovereign-weather-lora --save-path models/sovereign-weather-fused

# 3. 評估（建議使用 inline 腳本）
python scripts/evaluate/run_standard_benchmarks_inline.py --model-type qwen
# Llama 版：python scripts/evaluate/run_standard_benchmarks_inline.py --model-type llama
```

或透過 `run_all_eval.py --model-type llama` 一鍵執行並更新 `models/MODEL_CARD_LLAMA.md`。

## 定量 prompt 格式

`quantitative_prompts.json` 支援預期答案以計算準確率，格式如下：

```json
{
  "id": "math_1",
  "category": "math",
  "prompt": "若一個蘋果 5 元，買 3 個要多少錢？請只回答數字。",
  "expected": "15",
  "match_type": "contains_number"
}
```

`match_type` 可選：`contains_number`（含數字）、`contains_any`（含任一字串）、`contains_all`（含全部字串）、`exact`（完全匹配）。

## 解讀結果

- **Prompt 比較**：人工檢視兩模型輸出，確認微調模型在一般知識、推理等任務上與基礎模型相當。
- **定量評估**：PPL 在通用文本上應相當或略優；氣象文本 PPL 應明顯下降（領域改善）。準確率、長度應與基礎模型相當。
- **Perplexity**：
  - 差異 < 5%：正常，泛化能力維持良好
  - 差異 5–20%：略升，可接受
  - 差異 > 20%：可能退化，建議檢查訓練資料、學習率或迭代次數

## 自訂測試 prompt

可編輯 `generalization_prompts.json` 新增或修改 prompt，格式如下：

```json
{
  "prompts": [
    {
      "id": "unique_id",
      "category": "general_knowledge",
      "prompt": "你的問題..."
    }
  ]
}
```

**注意**：測試 prompt 應為**非氣象領域**，才能有效評估泛化能力是否下降。
