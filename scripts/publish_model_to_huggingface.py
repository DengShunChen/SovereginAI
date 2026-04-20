#!/usr/bin/env python3
"""
將訓練後的 LoRA 模型發布到 Hugging Face Hub。

支援兩種模式：
1. fuse（預設）：將 LoRA 適配器與基礎模型合併後上傳完整模型，使用者可直接 load。
2. adapter：僅上傳 LoRA 適配器檔案（較小），使用者需搭配基礎模型下載後使用。

使用前請先登入：
  huggingface-cli login
或設定環境變數 HF_TOKEN。

範例：
  # 合併基礎模型 + LoRA 後上傳（推薦）
  python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-ai

  # 僅上傳 LoRA 適配器（較小）
  python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-lora --mode adapter

  # 私人 repo
  python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-ai --private

  # Llama 版（指定適配器與 model card）
  python scripts/publish_model_to_huggingface.py --repo_id dschen/sovereign-weather-ai-llama \\
    --adapter_path models/sovereign-weather-lora-llama \\
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \\
    --model_card models/MODEL_CARD_LLAMA.md
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DEFAULT = PROJECT_ROOT / "models" / "sovereign-weather-lora"
MODEL_DEFAULT = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _load_model_card_content(model_card_path: Path | None, repo_id: str) -> str | None:
    """讀取 model card 並替換 repo 佔位符。"""
    path = model_card_path or (PROJECT_ROOT / "models" / "MODEL_CARD.md")
    if not path.exists():
        return None
    content = Path(path).read_text(encoding="utf-8")
    for stub in ("YOUR_USERNAME", "dschen"):
        content = content.replace(f"{stub}/sovereign-weather-ai", repo_id)
        content = content.replace(f"{stub}/sovereign-weather-lora", repo_id)
        content = content.replace(f"{stub}/sovereign-weather-ai-llama", repo_id)
        content = content.replace(f"{stub}/sovereign-weather-lora-llama", repo_id)
    return content


def publish_fused(
    repo_id: str,
    adapter_path: Path,
    model: str,
    private: bool,
    model_card_path: Path | None = None,
) -> None:
    """合併 LoRA 與基礎模型後上傳至 Hub。"""
    try:
        from mlx_lm import fuse
    except ImportError:
        print("請先安裝: pip install 'mlx-lm[train]'", file=sys.stderr)
        sys.exit(1)

    fuse_dir = PROJECT_ROOT / "models" / "fused_for_upload"
    fuse_dir.mkdir(parents=True, exist_ok=True)

    print(f"合併 LoRA 適配器與基礎模型：{model}")
    print(f"  適配器：{adapter_path}")
    print(f"  暫存目錄：{fuse_dir}")
    print("")

    # 使用 mlx_lm fuse 合併，但不直接上傳（以便我們控制 private 等選項）
    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model,
        "--adapter-path", str(adapter_path),
        "--save-path", str(fuse_dir),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("fuse 失敗，請檢查適配器路徑與基礎模型。", file=sys.stderr)
        sys.exit(1)

    # 上傳
    from huggingface_hub import HfApi, ModelCard, ModelCardData

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # 建立 README（優先使用指定的 model card）
    readme_path = fuse_dir / "README.md"
    content = _load_model_card_content(model_card_path, repo_id)
    if content:
        readme_path.write_text(content, encoding="utf-8")
    elif not readme_path.exists():
        card = ModelCard.from_template(
            ModelCardData(
                language="zh",
                library_name="mlx",
                pipeline_tag="text-generation",
                tags=["mlx", "weather", "taiwan", "qwen", "lora"],
                base_model=model,
            )
        )
        card.text = f"""# 氣象主權 AI（Sovereign Weather AI）

本模型由 [{model}](https://huggingface.co/{model}) 經 LoRA 微調，專注於臺灣氣象術語與預報風格。

## 使用方式（MLX）

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")

prompt = "請說明焚風現象"
messages = [{{"role": "user", "content": prompt}}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_dict=False,
)
response = generate(model, tokenizer, prompt=prompt, verbose=True)
```

或命令列：

```bash
python -m mlx_lm.generate --model {repo_id} --prompt "請說明焚風現象"
```
"""
        card.save(str(readme_path))

    api.upload_folder(
        folder_path=str(fuse_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"已上傳至 https://huggingface.co/{repo_id}")

    # 清理暫存
    shutil.rmtree(fuse_dir, ignore_errors=True)


def publish_adapter_only(
    repo_id: str,
    adapter_path: Path,
    model: str,
    private: bool,
    model_card_path: Path | None = None,
) -> None:
    """僅上傳 LoRA 適配器檔案。"""
    from huggingface_hub import HfApi, ModelCard, ModelCardData

    if not (adapter_path / "adapters.safetensors").exists():
        print(f"錯誤：找不到適配器 {adapter_path}/adapters.safetensors", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # 僅上傳必要檔案
    files_to_upload = ["adapters.safetensors", "adapter_config.json"]
    for fname in files_to_upload:
        fp = adapter_path / fname
        if fp.exists():
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )

    # README（優先使用指定的 model card；否則用精簡說明）
    content = _load_model_card_content(model_card_path, repo_id)
    if content:
        readme = content
    else:
        readme = f"""# 氣象主權 AI LoRA 適配器

本 LoRA 適配器需搭配基礎模型使用：[{model}](https://huggingface.co/{model})

## 使用方式（MLX）

1. 下載適配器至本地：
```bash
huggingface-cli download {repo_id} --local-dir ./sovereign-weather-lora
```

2. 推理：
```bash
python -m mlx_lm.generate \\
    --model {model} \\
    --adapter-path ./sovereign-weather-lora \\
    --prompt "請說明焚風現象"
```

訓練細節、限制與風險等完整說明請見專案內 `models/MODEL_CARD.md`。
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    finally:
        os.unlink(readme_path)
    print(f"已上傳至 https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將訓練後的 LoRA 模型發布到 Hugging Face Hub。"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=os.environ.get("HF_MODEL_REPO", ""),
        help="Hugging Face 模型 repo（或設定 config/.env 的 HF_MODEL_REPO）",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=ADAPTER_DEFAULT,
        help=f"LoRA 適配器目錄（預設: {ADAPTER_DEFAULT}）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_DEFAULT,
        help=f"基礎模型（預設: {MODEL_DEFAULT}）",
    )
    parser.add_argument(
        "--mode",
        choices=["fuse", "adapter"],
        default="fuse",
        help="fuse=合併後上傳完整模型（推薦）；adapter=僅上傳適配器",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="建立/更新為私人 repo",
    )
    parser.add_argument(
        "--model_card",
        type=Path,
        default=None,
        help="Model card 路徑（預設: models/MODEL_CARD.md；Llama 版可用 models/MODEL_CARD_LLAMA.md）",
    )
    args = parser.parse_args()

    if not args.repo_id:
        print("錯誤：請指定 --repo_id 或在 config/.env 設定 HF_MODEL_REPO", file=sys.stderr)
        sys.exit(1)

    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_dir():
        print(f"錯誤：適配器目錄不存在 {adapter_path}", file=sys.stderr)
        sys.exit(1)

    model_card = Path(args.model_card) if args.model_card else None

    if args.mode == "fuse":
        publish_fused(
            args.repo_id, adapter_path, args.model, args.private, model_card
        )
    else:
        publish_adapter_only(
            args.repo_id, adapter_path, args.model, args.private, model_card
        )


if __name__ == "__main__":
    main()
