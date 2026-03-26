"""
train.py
Training loop for T5-based ASTE model.

Usage:
    python train.py                          # defaults
    python train.py --epochs 15 --lr 3e-4
"""

import argparse
import os
import json
import time
import re
from pathlib import Path
from collections import Counter

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, T5Tokenizer
from tqdm import tqdm

from data_loader import get_dataloaders, parse_triplets
from model import ASTEModel


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          default="merged_train.jsonl")
    p.add_argument("--encoder",       default="google-t5/t5-base")
    p.add_argument("--output_dir",    default="checkpoints")
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--val_split",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--num_beams",     type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=2,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    return p.parse_args()


# ── Triplet-level F1 ─────────────────────────────────────────────────────────

def normalize_triplet(t):
    """Normalize a triplet dict for comparison."""
    asp = t.get("Aspect", "NULL").strip().lower()
    opn = t.get("Opinion", "NULL").strip().lower()
    # For F1 we compare aspect+opinion (ignoring VA for span F1)
    return (asp, opn)


def compute_triplet_f1(pred_triplets_list, gold_triplets_list):
    """
    Compute triplet-level precision, recall, F1.
    pred_triplets_list: list of list of triplet dicts (one list per sentence)
    gold_triplets_list: list of list of triplet dicts (one list per sentence)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, golds in zip(pred_triplets_list, gold_triplets_list):
        pred_set = Counter([normalize_triplet(t) for t in preds])
        gold_set = Counter([normalize_triplet(t) for t in golds])

        # TP: min of pred and gold counts for each triplet
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

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, scaler, grad_accum_steps):
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

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

    # Handle remaining gradients
    if num_batches % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def eval_epoch(model, loader, device, tokenizer, num_beams=4):
    model.eval()
    total_loss = 0
    num_batches = 0

    all_pred_triplets = []
    all_gold_triplets = []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Compute loss
        outputs = model.forward(input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

        # Generate predictions
        gen_ids = model.generate(input_ids, attention_mask, num_beams=num_beams)
        pred_texts = model.batch_decode(gen_ids)

        # Decode gold
        gold_labels = labels.clone()
        gold_labels[gold_labels == -100] = tokenizer.pad_token_id
        gold_texts = tokenizer.batch_decode(gold_labels, skip_special_tokens=True)

        for pred_text, gold_text in zip(pred_texts, gold_texts):
            pred_trips = parse_triplets(pred_text)
            gold_trips = parse_triplets(gold_text)
            all_pred_triplets.append(pred_trips)
            all_gold_triplets.append(gold_trips)

    metrics = compute_triplet_f1(all_pred_triplets, all_gold_triplets)
    metrics["loss"] = total_loss / max(num_batches, 1)

    return metrics, all_pred_triplets, all_gold_triplets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    Path(args.output_dir).mkdir(exist_ok=True)

    # Data
    print("Loading data …")
    train_loader, val_loader = get_dataloaders(
        args.data,
        val_split=args.val_split,
        tokenizer_name=args.encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    # Model
    print(f"Loading model: {args.encoder}")
    model = ASTEModel(model_name=args.encoder)
    model.to(device)
    tokenizer = model.tokenizer
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = None  # No AMP on CPU

    # Training
    best_val_f1  = 0.0
    history      = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, args.grad_accum
        )

        val_metrics, pred_trips, gold_trips = eval_epoch(
            model, val_loader, device, tokenizer, num_beams=args.num_beams
        )

        elapsed = time.time() - t0
        print(f"  {'Metric':<14} {'Train':>8} {'Val':>8}")
        print(f"  {'-'*32}")
        print(f"  {'Loss':<14} {train_metrics['loss']:>8.4f} {val_metrics['loss']:>8.4f}")
        print(f"  {'Precision':<14} {'':>8} {val_metrics['precision']:>8.4f}")
        print(f"  {'Recall':<14} {'':>8} {val_metrics['recall']:>8.4f}")
        print(f"  {'F1 (triplet)':<14} {'':>8} {val_metrics['f1']:>8.4f}")
        print(f"  {'TP/FP/FN':<14} {'':>8} {val_metrics['tp']}/{val_metrics['fp']}/{val_metrics['fn']}")

        # Show some predictions
        print(f"  Sample predictions:")
        for i in range(min(3, len(pred_trips))):
            print(f"    Pred: {pred_trips[i]}")
            print(f"    Gold: {gold_trips[i]}")
            print()

        print(f"  Time: {elapsed:.0f}s")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)

        # Save best
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      best_val_f1,
                "args":        vars(args),
            }, ckpt)
            print(f"  ✓ New best val F1={best_val_f1:.4f} — saved to {ckpt}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val triplet F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
