#!/usr/bin/env python3
"""
氣象主權 AI — Causal LM LoRA SFT（CUDA / A100）。

預設基座 google/gemma-4-12B-it。單機 8 GPU：
  torchrun --standalone --nproc_per_node=8 scripts/train/train_lora_sft.py
建議用 scripts/train/train_lora_a100.sh。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "for_training"
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "sovereign-weather-lora-gemma4-12b"
DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_USER_PROMPT = "請以台灣中央氣象署用語，撰寫或說明以下氣象內容。"

# 用 sentinel 反推 chat template 在 assistant 內容後的收尾字串（Gemma 4 為 <turn|>\n）
ASSISTANT_TAIL_SENTINEL = "\u0001A\u0001"

LORA_TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_SKIP_NAME_PARTS = ("vision", "audio", "embed_vision", "embed_audio")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _as_id_list(ids) -> list[int]:
    # transformers 5.x：apply_chat_template(tokenize=True) 可能回 BatchEncoding
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        try:
            ids = ids["input_ids"]
        except Exception:
            pass
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    out = list(ids)
    if out and not isinstance(out[0], int):
        raise TypeError(f"預期 token id list，得到 {type(out[0])}: {out[:5]!r}")
    return out


def infer_assistant_tail(tokenizer) -> str:
    """從 chat template 反推 assistant 內容後的收尾字串，避免寫死模型家族。"""
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": ASSISTANT_TAIL_SENTINEL},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if ASSISTANT_TAIL_SENTINEL not in rendered:
        raise RuntimeError("chat template 未保留 sentinel，無法推斷 assistant 收尾字串")
    _, tail = rendered.split(ASSISTANT_TAIL_SENTINEL, 1)
    return tail


def row_to_messages(row: dict) -> list[dict] | None:
    if isinstance(row.get("messages"), list) and row["messages"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        return msgs if any(m["role"] == "assistant" for m in msgs) else None

    text = (row.get("text") or "").strip()
    if text:
        return [
            {"role": "user", "content": DEFAULT_USER_PROMPT},
            {"role": "assistant", "content": text},
        ]

    inst = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if not out:
        return None
    user = f"{inst}\n{inp}".strip() if inp else inst
    if not user:
        user = DEFAULT_USER_PROMPT
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def tokenize_supervised(
    tokenizer,
    messages: list[dict],
    max_length: int,
    assistant_tail: str,
) -> tuple[list[int], list[int]] | None:
    """prompt_ids + answer_ids 直接拼接。不比對前綴，避開 Gemma 4 空 thought channel。"""
    if not messages or messages[-1]["role"] not in ("assistant", "model"):
        return None
    answer = messages[-1]["content"]
    if not isinstance(answer, str) or not answer.strip():
        return None

    prompt_ids = _as_id_list(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            enable_thinking=False,
        )
    )
    answer_ids = _as_id_list(
        tokenizer(answer + assistant_tail, add_special_tokens=False)
    )
    if not prompt_ids or not answer_ids:
        return None

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(x == -100 for x in labels):
            return None
    return input_ids, labels


def format_preview(tokenizer, input_ids: list[int], labels: list[int], limit: int = 800) -> str:
    labeled = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    masked_n = sum(1 for lab in labels if lab == -100)
    full = tokenizer.decode(input_ids, skip_special_tokens=False)
    ans = tokenizer.decode(labeled, skip_special_tokens=False) if labeled else ""
    if len(full) > limit:
        full = full[:limit] + "…"
    if len(ans) > limit:
        ans = ans[:limit] + "…"
    return (
        f"  tokens={len(input_ids)} masked={masked_n} labeled={len(labeled)}\n"
        f"  --- full ---\n{full}\n"
        f"  --- labeled (loss) ---\n{ans}"
    )


def pack_sequences(
    pairs: list[tuple[list[int], list[int]]],
    max_length: int,
    pad_id: int,
) -> list[dict]:
    packs: list[dict] = []
    cur_ids: list[int] = []
    cur_lab: list[int] = []

    def flush() -> None:
        nonlocal cur_ids, cur_lab
        if not cur_ids:
            return
        attn = [1] * len(cur_ids)
        pad_n = max_length - len(cur_ids)
        if pad_n > 0:
            cur_ids = cur_ids + [pad_id] * pad_n
            cur_lab = cur_lab + [-100] * pad_n
            attn = attn + [0] * pad_n
        packs.append(
            {
                "input_ids": cur_ids,
                "labels": cur_lab,
                "attention_mask": attn,
            }
        )
        cur_ids, cur_lab = [], []

    for ids, lab in pairs:
        if len(ids) > max_length:
            ids = ids[:max_length]
            lab = lab[:max_length]
        if cur_ids and len(cur_ids) + len(ids) > max_length:
            flush()
        cur_ids.extend(ids)
        cur_lab.extend(lab)
    flush()
    return packs


class PackedSFTDataset(Dataset):
    def __init__(self, packs: list[dict]):
        self.packs = packs

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict:
        p = self.packs[idx]
        return {
            "input_ids": torch.tensor(p["input_ids"], dtype=torch.long),
            "labels": torch.tensor(p["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(p["attention_mask"], dtype=torch.long),
        }


def build_dataset(
    rows: list[dict],
    tokenizer,
    max_length: int,
    seed: int,
    assistant_tail: str,
) -> tuple[PackedSFTDataset, list[tuple[list[int], list[int]]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    pairs: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for row in rows:
        messages = row_to_messages(row)
        if not messages:
            skipped += 1
            continue
        tok = tokenize_supervised(tokenizer, messages, max_length, assistant_tail)
        if tok is None:
            skipped += 1
            continue
        pairs.append(tok)
    if is_main():
        med = sorted(len(p[0]) for p in pairs)[len(pairs) // 2] if pairs else 0
        print(f"  可用樣本 {len(pairs)}，略過 {skipped}，packing 前 token 中位數 {med}")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer.pad_token_id 未設定")
    packs = pack_sequences(pairs, max_length, pad_id)
    if is_main():
        print(f"  packing 後序列數 {len(packs)}（max_length={max_length}）")
    return PackedSFTDataset(packs), pairs


def pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def resolve_lora_targets(model) -> list[str]:
    """只掛語言塔的 q/k/v/o/gate/up/down，避開 vision/audio。"""
    found: list[str] = []
    leaves: set[str] = set()
    for name, _mod in model.named_modules():
        lname = name.lower()
        if any(p in lname for p in LORA_SKIP_NAME_PARTS):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in LORA_TARGET_LEAVES:
            found.append(name)
            leaves.add(leaf)
    if not found:
        raise RuntimeError("找不到 LoRA 目標模組（語言塔 q/k/v/o/mlp）")
    if is_main():
        print(f"  LoRA leaves={sorted(leaves)}  modules={len(found)}")
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal LM LoRA SFT（A100 / CUDA）")
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", DEFAULT_DATA)))
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path(os.environ.get("ADAPTER_DIR", DEFAULT_ADAPTER)),
    )
    p.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "2048")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    p.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", "5")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "2e-4")))
    p.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "64")))
    p.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "128")))
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--min-train", type=int, default=10, help="少於此筆數直接失敗")
    p.add_argument("--max-train-rows", type=int, default=0, help=">0 時只取前 N 筆（smoke）")
    p.add_argument(
        "--format-check",
        action="store_true",
        help="只 tokenize + 印格式自檢，不載入模型",
    )
    return p.parse_args()


def main() -> None:
    _load_dotenv()
    args = parse_args()

    if not args.format_check and not torch.cuda.is_available():
        print("錯誤：找不到 CUDA。A100 訓練請在 GPU 節點跑，不要用 MLX 腳本。", file=sys.stderr)
        sys.exit(1)

    train_path = args.data_dir / "train.jsonl"
    valid_path = args.data_dir / "valid.jsonl"
    if not train_path.exists():
        print(f"錯誤：找不到 {train_path}", file=sys.stderr)
        print("請先執行：./scripts/run_collect_and_prepare.sh", file=sys.stderr)
        sys.exit(1)

    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
        valid_rows = valid_rows[: max(1, args.max_train_rows // 10)]
    if len(train_rows) < args.min_train:
        print(f"錯誤：train.jsonl 只有 {len(train_rows)} 筆（< {args.min_train}）", file=sys.stderr)
        sys.exit(1)
    if is_main() and len(train_rows) < 1000:
        print(f"警告：訓練筆數 {len(train_rows)} < 1000，LoRA 容易過擬合、通用能力掉很快。")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_bs = args.batch_size * args.grad_accum * world
    if is_main():
        print("=== 氣象主權 AI LoRA（Gemma 4 12B / CUDA）===")
        print(f"  模型：{args.model}")
        print(f"  資料：{args.data_dir}  train={len(train_rows)} valid={len(valid_rows)}")
        print(f"  適配器：{args.adapter_dir}")
        if torch.cuda.is_available():
            print(f"  GPU：{world} × {torch.cuda.get_device_name(0)}")
        print(f"  LoRA r={args.lora_r} alpha={args.lora_alpha}  lr={args.lr}  epochs={args.epochs}")
        print(f"  micro={args.batch_size} accum={args.grad_accum}  global_batch={global_bs}")
        print(f"  max_length={args.max_length}  attn={pick_attn_implementation()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assistant_tail = infer_assistant_tail(tokenizer)
    if is_main():
        print(f"  assistant_tail={assistant_tail!r}")

    if is_main():
        print("tokenize + pack train...")
    train_ds, train_pairs = build_dataset(
        train_rows, tokenizer, args.max_length, args.seed, assistant_tail
    )
    eval_ds = None
    if valid_rows:
        if is_main():
            print("tokenize + pack valid...")
        eval_ds, _ = build_dataset(
            valid_rows, tokenizer, args.max_length, args.seed + 1, assistant_tail
        )
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None
    if len(train_ds) == 0:
        print("錯誤：沒有可訓練序列", file=sys.stderr)
        sys.exit(1)

    if is_main() and train_pairs:
        print("=== 格式自檢（第一筆未 packing）===")
        print(format_preview(tokenizer, train_pairs[0][0], train_pairs[0][1]))
        labeled = [lab for lab in train_pairs[0][1] if lab != -100]
        if not labeled:
            print("錯誤：第一筆沒有可訓練 label", file=sys.stderr)
            sys.exit(1)
        decoded = tokenizer.decode(labeled, skip_special_tokens=False)
        if "<|turn>user" in decoded or "<|start_header_id|>user" in decoded:
            print("錯誤：label 區段含 user turn，masking 失效", file=sys.stderr)
            sys.exit(1)

    if args.format_check:
        if is_main():
            print("format-check 通過，不載入模型。")
        return

    attn = pick_attn_implementation()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )
    except Exception as exc:
        if attn == "sdpa":
            raise
        if is_main():
            print(f"attn={attn} 載入失敗，改 sdpa：{exc}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model.config.use_cache = False
    model.enable_input_require_grads()

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = resolve_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        ),
    )
    if is_main():
        model.print_trainable_parameters()

    ckpt_dir = args.adapter_dir / "checkpoints"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=args.seed,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer_kw = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    try:
        trainer = Trainer(**trainer_kw, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kw, tokenizer=tokenizer)
    trainer.train()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.adapter_dir))

    if is_main():
        tokenizer.save_pretrained(args.adapter_dir)
        meta = {
            "base_model": args.model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "global_batch": global_bs,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "target_modules_count": len(target_modules),
        }
        with open(args.adapter_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n訓練完成。適配器：{args.adapter_dir}")
        print(
            "推理：python scripts/train/generate_sft.py "
            f"--adapter-path {args.adapter_dir} --prompt '請說明焚風現象'"
        )


if __name__ == "__main__":
    main()
