"""
aste_model.py
Unified T5 Sequence-to-Sequence Generative ASTE Model Pipeline.

Usage:
  # To train:
  python aste_model.py --mode train --data merged_train.jsonl --epochs 5 --batch_size 8

  # To predict:
  python aste_model.py --mode predict --test test_data.jsonl --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import json
import time
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# ── 1. Constants ─────────────────────────────────────────────────────────────
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 256
PREFIX = "extract aspect opinion sentiment: "

# ── 2. Data Processing & Formatting ──────────────────────────────────────────

def format_triplets(quadruplets: list) -> str:
    """Convert list of quadruplet dicts to linearised target string."""
    parts = []
    for q in quadruplets:
        asp = q.get("Aspect", "NULL") or "NULL"
        opn = q.get("Opinion", "NULL") or "NULL"
        va  = q["VA"]
        v, a = va.split("#")
        parts.append(f"( {asp} | {opn} | {v} | {a} )")
    return " ; ".join(parts)


def parse_triplets(text: str) -> list:
    """Parse linearised triplet string back into list of dicts.
    Returns list of {"Aspect": ..., "Opinion": ..., "VA": "V#A"}
    """
    triplets = []
    raw_parts = text.split(";")
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        part = part.strip("() ")
        fields = [f.strip() for f in part.split("|")]
        if len(fields) >= 4:
            asp = fields[0] if fields[0] else "NULL"
            opn = fields[1] if fields[1] else "NULL"
            try:
                v = float(fields[2])
                a = float(fields[3])
                v = max(1.0, min(9.0, round(v, 2)))
                a = max(1.0, min(9.0, round(a, 2)))
            except (ValueError, IndexError):
                v, a = 5.0, 5.0
            triplets.append({
                "Aspect":  asp,
                "Opinion": opn,
                "VA":      f"{v:.2f}#{a:.2f}",
            })
        elif len(fields) >= 2:
            asp = fields[0] if fields[0] else "NULL"
            opn = fields[1] if fields[1] else "NULL"
            triplets.append({
                "Aspect":  asp,
                "Opinion": opn,
                "VA":      "5.00#5.00",
            })
    return triplets


class ASTEDataset(Dataset):
    """Dataset mapping sentences to linearized target sequences."""
    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "google-t5/t5-small",
        max_input_len: int = MAX_INPUT_LEN,
        max_target_len: int = MAX_TARGET_LEN,
        is_test: bool = False,
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.is_test = is_test

        self.samples = []
        self._load(jsonl_path)

    def _load(self, path: str):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record["Text"]

                if self.is_test:
                    self.samples.append({"id": record["ID"], "text": text, "target": ""})
                else:
                    target = format_triplets(record.get("Quadruplet", []))
                    self.samples.append({"id": record["ID"], "text": text, "target": target})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = PREFIX + s["text"]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids":      input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
        }

        if not self.is_test and s["target"]:
            target_enc = self.tokenizer(
                s["target"],
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = target_enc["input_ids"].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels

        return result


def get_dataloaders(
    train_path: str,
    val_split: float = 0.1,
    tokenizer_name: str = "google-t5/t5-small",
    batch_size: int = 8,
    num_workers: int = 2,
):
    full_ds = ASTEDataset(train_path, tokenizer_name=tokenizer_name)

    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ── 3. Model Architecture ────────────────────────────────────────────────────

class ASTEModel:
    """Thin wrapper around T5ForConditionalGeneration."""
    def __init__(self, model_name: str = "google-t5/t5-small", max_target_len: int = 256):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_target_len = max_target_len

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self): self.model.train()
    def eval(self): self.model.eval()
    def parameters(self): return self.model.parameters()
    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, state_dict): self.model.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, num_beams=4):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=self.max_target_len, num_beams=num_beams,
            early_stopping=True, no_repeat_ngram_size=0,
        )

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


# ── 4. Training Eval Metrics ─────────────────────────────────────────────────

def normalize_triplet(t):
    return (t.get("Aspect", "NULL").strip().lower(), t.get("Opinion", "NULL").strip().lower())

def compute_triplet_f1(pred_triplets_list, gold_triplets_list):
    total_tp = total_fp = total_fn = 0
    for preds, golds in zip(pred_triplets_list, gold_triplets_list):
        pred_set = Counter([normalize_triplet(t) for t in preds])
        gold_set = Counter([normalize_triplet(t) for t in golds])

        for triplet in pred_set:
            if triplet in gold_set:
                tp = min(pred_set[triplet], gold_set[triplet])
                total_tp += tp
                total_fp += pred_set[triplet] - tp
            else:
                total_fp += pred_set[triplet]

        for triplet in gold_set:
            if triplet in pred_set:
                total_fn += max(0, gold_set[triplet] - pred_set[triplet])
            else:
                total_fn += gold_set[triplet]

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}


# ── 5. Training & Eval Handlers ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, scaler, grad_accum_steps):
    model.train()
    total_loss, num_batches = 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cpu'):
            outputs = model.forward(input_ids, attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        total_loss += outputs.loss.item()
        num_batches += 1

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if num_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {"loss": total_loss / max(num_batches, 1)}

@torch.no_grad()
def eval_epoch(model, loader, device, tokenizer, num_beams=4):
    model.eval()
    total_loss, num_batches = 0, 0
    all_pred_triplets, all_gold_triplets = [], []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model.forward(input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

        gen_ids = model.generate(input_ids, attention_mask, num_beams=num_beams)
        pred_texts = model.batch_decode(gen_ids)

        gold_labels = labels.clone()
        gold_labels[gold_labels == -100] = tokenizer.pad_token_id
        gold_texts = tokenizer.batch_decode(gold_labels, skip_special_tokens=True)

        for pt, gt in zip(pred_texts, gold_texts):
            all_pred_triplets.append(parse_triplets(pt))
            all_gold_triplets.append(parse_triplets(gt))

    metrics = compute_triplet_f1(all_pred_triplets, all_gold_triplets)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics, all_pred_triplets, all_gold_triplets


def run_training(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    Path(args.output_dir).mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        args.data, val_split=args.val_split, tokenizer_name=args.encoder,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = ASTEModel(model_name=args.encoder).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)

    best_val_f1, patience_counter = 0.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, None, args.grad_accum)
        val_metrics, pt, gt = eval_epoch(model, val_loader, device, model.tokenizer, num_beams=args.num_beams)

        print(f"  Loss | Train: {train_metrics['loss']:.4f}  Val: {val_metrics['loss']:.4f}")
        print(f"  Val  | Prec: {val_metrics['precision']:.4f}  Rec: {val_metrics['recall']:.4f}  F1: {val_metrics['f1']:.4f} [TP:{val_metrics['tp']}]")
        
        history.append({"epoch": epoch, "val_f1": val_metrics["f1"]})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt = os.path.join(args.output_dir, "best_model.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": best_val_f1, "args": vars(args)}, ckpt)
            print(f"  ✓ Saved new best model to {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered.")
                break


# ── 6. Inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder_name = ckpt.get("args", {}).get("encoder", args.encoder)

    model = ASTEModel(model_name=encoder_name)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (val_f1={ckpt.get('val_f1', 0):.4f})")

    records = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            if line.strip(): records.append(json.loads(line.strip()))

    results = []
    for i in range(0, len(records), args.batch_size):
        batch_records = records[i:i + args.batch_size]
        input_texts = [PREFIX + r["Text"] for r in batch_records]

        enc = model.tokenizer(input_texts, max_length=MAX_INPUT_LEN, padding="max_length", truncation=True, return_tensors="pt")
        gen_ids = model.generate(enc["input_ids"].to(device), enc["attention_mask"].to(device), num_beams=args.num_beams)
        pred_texts = model.batch_decode(gen_ids)

        for record, pred_text in zip(batch_records, pred_texts):
            triplets = parse_triplets(pred_text)
            if not triplets: triplets = [{"Aspect": "NULL", "Opinion": "NULL", "VA": "5.00#5.00"}]
            results.append({"ID": record["ID"], "Triplet": triplets})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} predictions → {args.output}")


# ── 7. Main Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",          choices=["train", "predict"], required=True)
    p.add_argument("--data",          default="merged_train.jsonl")
    p.add_argument("--test",          default="test_data.jsonl")
    p.add_argument("--encoder",       default="google-t5/t5-small")
    p.add_argument("--output_dir",    default="checkpoints")
    p.add_argument("--checkpoint",    default="checkpoints/best_model.pt")
    p.add_argument("--output",        default="predictions.jsonl")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--val_split",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--num_beams",     type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=2)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    
    args = p.parse_args()
    if args.mode == "train":
        run_training(args)
    else:
        run_prediction(args)
