"""
train.py
Full training loop for ModernBERT + GCN ASTE model.

Usage:
    python train.py                          # defaults
    python train.py --epochs 10 --lr 2e-5
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from data_loader import get_dataloaders, ID2LABEL, PAD_LABEL_ID
from model import ASTEModel, ASTELoss


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          default="merged_train.jsonl")
    p.add_argument("--encoder",       default="answerdotai/ModernBERT-base")
    p.add_argument("--output_dir",    default="checkpoints")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--val_split",     type=float, default=0.1)
    p.add_argument("--gcn_layers",    type=int,   default=2)
    p.add_argument("--lambda_span",   type=float, default=1.0)
    p.add_argument("--lambda_va",     type=float, default=0.5)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--no_dep_graph",  action="store_true",
                   help="Disable spaCy dependency graph (faster, lower quality)")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    """
    Compute token-level metrics for span labels (ignores PAD_LABEL_ID=-100).
    Returns overall accuracy, macro precision, recall, F1,
    and per-class precision, recall, F1 for each BIO label.
    """
    LABEL_NAMES = {0: "O", 1: "B-ASP", 2: "I-ASP", 3: "B-OPN", 4: "I-OPN"}
    n_classes = len(LABEL_NAMES)

    # Per-class TP, FP, FN
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    correct = total = 0

    for p_seq, l_seq in zip(preds, labels):
        for pi, li in zip(p_seq, l_seq):
            if li == PAD_LABEL_ID:
                continue
            total += 1
            if pi == li:
                correct += 1
                tp[pi] += 1
            else:
                fp[pi] += 1
                fn[li] += 1

    accuracy = correct / (total + 1e-9)

    per_class = {}
    for c in range(n_classes):
        p  = tp[c] / (tp[c] + fp[c] + 1e-9)
        r  = tp[c] / (tp[c] + fn[c] + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class[LABEL_NAMES[c]] = {"precision": p, "recall": r, "f1": f1,
                                      "support": tp[c] + fn[c]}

    # Macro averages (span labels only, exclude O)
    span_classes = ["B-ASP", "I-ASP", "B-OPN", "I-OPN"]
    macro_prec = sum(per_class[c]["precision"] for c in span_classes) / len(span_classes)
    macro_rec  = sum(per_class[c]["recall"]    for c in span_classes) / len(span_classes)
    macro_f1   = sum(per_class[c]["f1"]        for c in span_classes) / len(span_classes)

    return {
        "accuracy":   accuracy,
        "precision":  macro_prec,
        "recall":     macro_rec,
        "f1":         macro_f1,
        "per_class":  per_class,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = span_loss_sum = va_loss_sum = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        adj            = batch["adj"].to(device)
        span_labels    = batch["span_labels"].to(device)
        valence        = batch["valence"].to(device)
        arousal        = batch["arousal"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            span_logits, va_pred = model(input_ids, attention_mask, adj)
            loss, sl, vl = criterion(span_logits, va_pred, span_labels, valence, arousal)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss    += loss.item()
        span_loss_sum += sl.item()
        va_loss_sum   += vl.item()

        preds = span_logits.argmax(-1).cpu().tolist()
        lbls  = span_labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls)

    n = len(loader)
    m = compute_metrics(all_preds, all_labels)
    return {
        "loss":      total_loss / n,
        "span_loss": span_loss_sum / n,
        "va_loss":   va_loss_sum / n,
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
        "per_class": m["per_class"],
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = span_loss_sum = va_loss_sum = 0
    va_preds, va_targets = [], []
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        adj            = batch["adj"].to(device)
        span_labels    = batch["span_labels"].to(device)
        valence        = batch["valence"].to(device)
        arousal        = batch["arousal"].to(device)

        with torch.cuda.amp.autocast():
            span_logits, va_pred = model(input_ids, attention_mask, adj)
            loss, sl, vl = criterion(span_logits, va_pred, span_labels, valence, arousal)

        total_loss    += loss.item()
        span_loss_sum += sl.item()
        va_loss_sum   += vl.item()

        preds = span_logits.argmax(-1).cpu().tolist()
        lbls  = span_labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls)

        va_preds.extend(va_pred.cpu().tolist())
        va_tgt = torch.stack([valence, arousal], dim=1).cpu().tolist()
        va_targets.extend(va_tgt)

    n = len(loader)
    m = compute_metrics(all_preds, all_labels)

    import numpy as np
    va_preds   = np.array(va_preds)
    va_targets = np.array(va_targets)
    val_mae    = float(np.abs(va_preds[:, 0] - va_targets[:, 0]).mean())
    aro_mae    = float(np.abs(va_preds[:, 1] - va_targets[:, 1]).mean())

    return {
        "loss":      total_loss / n,
        "span_loss": span_loss_sum / n,
        "va_loss":   va_loss_sum / n,
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
        "per_class": m["per_class"],
        "val_mae":   val_mae,
        "aro_mae":   aro_mae,
    }


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
        use_dep_graph=not args.no_dep_graph,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    # Model & loss
    model = ASTEModel(
        encoder_name=args.encoder,
        gcn_layers=args.gcn_layers,
    ).to(device)

    # Upweight B/I span labels — O is ~81% of tokens, causing near-zero span F1
    criterion = ASTELoss(
        lambda_span=args.lambda_span,
        lambda_va=args.lambda_va,
        span_class_weights=[1.0, 5.0, 4.0, 5.0, 4.0],
    )
    criterion.ce_loss.weight = criterion.ce_loss.weight.to(device)

    # Optimiser — lower LR for encoder, higher for new heads
    encoder_params = list(model.encoder.parameters())
    head_params    = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]
    optimizer = AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params,    "lr": args.lr * 10},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training
    best_val_f1  = 0.0
    history      = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_metrics   = eval_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"  {'Metric':<12} {'Train':>8} {'Val':>8}")
        print(f"  {'-'*30}")
        print(f"  {'Loss':<12} {train_metrics['loss']:>8.4f} {val_metrics['loss']:>8.4f}")
        print(f"  {'Accuracy':<12} {train_metrics['accuracy']:>8.4f} {val_metrics['accuracy']:>8.4f}")
        print(f"  {'Precision':<12} {train_metrics['precision']:>8.4f} {val_metrics['precision']:>8.4f}")
        print(f"  {'Recall':<12} {train_metrics['recall']:>8.4f} {val_metrics['recall']:>8.4f}")
        print(f"  {'F1 (macro)':<12} {train_metrics['f1']:>8.4f} {val_metrics['f1']:>8.4f}")
        print(f"  {'Val V-MAE':<12} {'':>8} {val_metrics['val_mae']:>8.3f}")
        print(f"  {'Val A-MAE':<12} {'':>8} {val_metrics['aro_mae']:>8.3f}")
        print(f"  Per-class (val):")
        for cls, s in val_metrics['per_class'].items():
            print(f"    {cls:<10} P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  support={s['support']}")
        print(f"  Time: {elapsed:.0f}s")

        skip = {"per_class"}
        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_metrics.items() if k not in skip},
               **{f"val_{k}": v for k, v in val_metrics.items() if k not in skip}}
        history.append(row)

        # Save best
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            ckpt = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      best_val_f1,
                "args":        vars(args),
            }, ckpt)
            print(f"  ✓ New best val F1={best_val_f1:.4f} — saved to {ckpt}")

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val span F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
