"""QLoRA на Qwen2.5: KL (teacher_answers) или CE (hard_labels), lr по умолчанию 5e-5."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Запуск из корня репозитория: uv run python scripts/train_distill.py
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from openjev.prompter import option_rows

from distill_common import (
    build_training_prompt,
    hard_label_class_index,
    hard_label_probability_vector,
    letter_token_ids,
    parse_question,
    teacher_probability_vector,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install train extras: uv sync --extra train") from exc


@dataclass(frozen=True, slots=True)
class _TrainRow:
    prompt: str
    num_opts: int
    letter_ids: list[int]
    loss: str  # "kl" | "ce"
    target_probs: list[float] | None = None
    target_idx: int | None = None


class DistillDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_length: int = 1024,
        prompt_format: str = "instruct",
        loss_mode: str = "auto",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_mode = loss_mode
        self.items: list[_TrainRow] = []

        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = row["state"]
                questions = row["questions"]
                teacher_answers = row.get("teacher_answers") or {}
                hard_labels = row.get("hard_labels") or {}

                for q_key, q_raw in questions.items():
                    try:
                        question = parse_question(q_raw)
                    except Exception:
                        continue

                    teacher = teacher_answers.get(q_key)
                    hard = hard_labels.get(q_key)
                    use_kl = False
                    use_ce = False
                    probs: list[float] | None = None
                    target_idx: int | None = None

                    if teacher and self.loss_mode in ("auto", "kl"):
                        probs = teacher_probability_vector(question, teacher)
                        use_kl = probs is not None
                    if not use_kl and hard and self.loss_mode in ("auto", "ce"):
                        target_idx = hard_label_class_index(question, hard)
                        use_ce = target_idx is not None
                        if use_ce:
                            probs = hard_label_probability_vector(question, hard)

                    if not use_kl and not use_ce:
                        continue

                    num_opts = len(option_rows(question))
                    prompt = build_training_prompt(
                        state,
                        question,
                        tokenizer,
                        prompt_format=prompt_format,
                    )
                    self.items.append(
                        _TrainRow(
                            prompt=prompt,
                            num_opts=num_opts,
                            letter_ids=letter_token_ids(tokenizer, num_opts),
                            loss="kl" if use_kl else "ce",
                            target_probs=probs if (use_kl or use_ce) else None,
                            target_idx=target_idx if use_ce else None,
                        )
                    )

        if not self.items:
            raise ValueError(f"no training rows loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        tokens = self.tokenizer(
            item.prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        target_probs = (
            torch.tensor(item.target_probs, dtype=torch.float32)
            if item.target_probs is not None
            else torch.zeros(item.num_opts, dtype=torch.float32)
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "num_opts": item.num_opts,
            "target_probs": target_probs,
            "letter_ids": torch.tensor(item.letter_ids, dtype=torch.long),
            "loss": item.loss,
            "target_idx": item.target_idx if item.target_idx is not None else -1,
        }


def _collate_batch(batch: list[dict]) -> dict:
    max_opts = max(int(b["num_opts"]) for b in batch)
    padded_targets = []
    padded_letters = []
    for b in batch:
        t = b["target_probs"]
        pad = max_opts - t.shape[0]
        if pad:
            t = F.pad(t, (0, pad))
        padded_targets.append(t)
        lid = b["letter_ids"]
        if pad:
            lid = F.pad(lid, (0, pad), value=int(lid[0]))
        padded_letters.append(lid)
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "num_opts": torch.tensor([b["num_opts"] for b in batch], dtype=torch.long),
        "target_probs": torch.stack(padded_targets),
        "letter_ids": torch.stack(padded_letters),
        "max_opts": max_opts,
        "loss": [b["loss"] for b in batch],
        "target_idx": torch.tensor([b["target_idx"] for b in batch], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA distillation on Jev soft labels")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="scripts/distill_dataset.jsonl")
    parser.add_argument("--output-dir", default="./distilled_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Default 5e-5 (Kev): 2e-4 often hurts base MMLU.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["auto", "kl", "ce"],
        default="auto",
        help="auto: teacher_answers→KL, hard_labels→CE; kl/ce force one mode.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--prompt-format", default="instruct", choices=["instruct", "chatml", "auto"])
    parser.add_argument("--no-4bit", action="store_true", help="Disable QLoRA (CPU / debug).")
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=0.1,
        help="CE only: add λ·mean((P−Y)²) on option softmax vs one-hot hard label.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_4bit = device.type == "cuda" and not args.no_4bit

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            print("bitsandbytes not available; falling back to full precision.", file=sys.stderr)
            use_4bit = False

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" and not use_4bit else None,
    )
    if use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = DistillDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        loss_mode=args.loss_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        micro_steps = 0

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_probs = batch["target_probs"].to(device)
            letter_ids = batch["letter_ids"].to(device)
            num_opts = batch["num_opts"].to(device)
            max_opts = int(batch["max_opts"])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            last_logits = outputs.logits[:, -1, :]

            batch_loss = torch.tensor(0.0, device=device)
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                n = int(num_opts[i].item())
                ids = letter_ids[i, :n]
                student_sub = last_logits[i, ids]
                if batch["loss"][i] == "ce":
                    idx = int(batch["target_idx"][i].item())
                    if idx < 0 or idx >= n:
                        continue
                    ce = F.cross_entropy(
                        student_sub.unsqueeze(0),
                        torch.tensor([idx], device=device),
                    )
                    batch_loss = batch_loss + ce
                    if args.brier_weight > 0:
                        student_p = F.softmax(student_sub, dim=-1)
                        target_y = target_probs[i, :n]
                        brier = ((student_p - target_y) ** 2).mean()
                        batch_loss = batch_loss + args.brier_weight * brier
                else:
                    student_log_probs = F.log_softmax(student_sub, dim=-1)
                    teacher_p = target_probs[i, :n]
                    batch_loss = batch_loss + kl_loss_fn(
                        student_log_probs.unsqueeze(0),
                        teacher_p.unsqueeze(0),
                    )

            loss = batch_loss / max(batch_size, 1) / args.grad_accum
            loss.backward()
            running_loss += float(loss.item()) * args.grad_accum
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loop.set_postfix({"loss": f"{running_loss / micro_steps:.4f}"})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Merging LoRA and exporting...")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Exported distilled weights to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
