"""
data_loader.py
T5 seq2seq data loader for ASTE.

Input:  "Extract aspect-opinion-sentiment triplets: <sentence>"
Target: "( aspect | opinion | valence | arousal ) ; ( ... )"

Handles NULL aspects/opinions as literal "NULL" strings.
"""

import json
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

# ── constants ───────────────────────────────────────────────────────────────
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 256
PREFIX = "extract aspect opinion sentiment: "

# ── helpers ──────────────────────────────────────────────────────────────────

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
    # Split on ; then parse each ( ... )
    raw_parts = text.split(";")
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Remove parens
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
            # Partial parse — at least aspect and opinion
            asp = fields[0] if fields[0] else "NULL"
            opn = fields[1] if fields[1] else "NULL"
            triplets.append({
                "Aspect":  asp,
                "Opinion": opn,
                "VA":      "5.00#5.00",
            })
    return triplets


# ── Dataset ──────────────────────────────────────────────────────────────────

class ASTEDataset(Dataset):
    """
    Each item is one sentence with ALL its quadruplets as the target.
    This groups quadruplets per sentence (unlike the old per-quadruplet approach).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "google-t5/t5-base",
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
                    self.samples.append({
                        "id":     record["ID"],
                        "text":   text,
                        "target": "",
                    })
                else:
                    target = format_triplets(record["Quadruplet"])
                    self.samples.append({
                        "id":     record["ID"],
                        "text":   text,
                        "target": target,
                    })

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
            # Replace padding token id with -100 so it's ignored in loss
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels

        return result


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_path: str,
    val_split: float = 0.1,
    tokenizer_name: str = "google-t5/t5-base",
    batch_size: int = 8,
    num_workers: int = 2,
    **kwargs,  # accept and ignore extra args for compatibility
):
    full_ds = ASTEDataset(
        train_path,
        tokenizer_name=tokenizer_name,
    )

    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing T5 data loader …")
    tr, vl = get_dataloaders("merged_train.jsonl", batch_size=4)
    batch = next(iter(tr))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
    print(f"Train batches: {len(tr)}, Val batches: {len(vl)}")

    # Show a sample
    ds = tr.dataset.dataset  # unwrap Subset → ASTEDataset
    s = ds.samples[0]
    print(f"\n  Input:  {PREFIX + s['text'][:80]}...")
    print(f"  Target: {s['target'][:120]}...")
