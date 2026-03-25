"""
data_loader.py
Loads merged_train.jsonl, builds token-level BIO labels for aspect & opinion spans,
constructs spaCy dependency graphs, and returns PyTorch DataLoader objects.

Label scheme
------------
B-ASP / I-ASP  – aspect span
B-OPN / I-OPN  – opinion span
O              – outside
NULL_SPAN      – token belongs to a NULL aspect/opinion (no span to extract)
"""

import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import spacy

# ── constants ───────────────────────────────────────────────────────────────
LABEL2ID = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPN": 3, "I-OPN": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_SPAN_LABELS = len(LABEL2ID)

NULL_VALENCE  = 5.0
NULL_AROUSAL  = 5.0
MAX_SEQ_LEN   = 128
PAD_LABEL_ID  = -100      # ignored by CrossEntropyLoss

# ── helpers ──────────────────────────────────────────────────────────────────

def _find_span_positions(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    """Return (start_char, end_char) of first occurrence of phrase in text."""
    if not phrase or phrase == "NULL":
        return None
    m = re.search(re.escape(phrase), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def _char_to_token_span(offset_mapping, start_char, end_char):
    """Map character offsets to token indices using the tokenizer's offset_mapping."""
    tok_start, tok_end = None, None
    for idx, (cs, ce) in enumerate(offset_mapping):
        if cs == 0 and ce == 0:   # special token
            continue
        if tok_start is None and cs >= start_char:
            tok_start = idx
        if ce <= end_char:
            tok_end = idx
    return tok_start, tok_end


def build_bio_labels(offset_mapping, text, aspect, opinion):
    """
    Build per-token BIO label array aligned with tokenizer output.
    Returns list of int label ids.
    """
    n = len(offset_mapping)
    labels = [PAD_LABEL_ID] * n   # default → ignored (special tokens keep this)

    # Mark real tokens as O first
    for idx, (cs, ce) in enumerate(offset_mapping):
        if cs == 0 and ce == 0:
            continue
        labels[idx] = LABEL2ID["O"]

    # Aspect span
    asp_pos = _find_span_positions(text, aspect)
    if asp_pos:
        ts, te = _char_to_token_span(offset_mapping, asp_pos[0], asp_pos[1])
        if ts is not None and te is not None:
            labels[ts] = LABEL2ID["B-ASP"]
            for i in range(ts + 1, te + 1):
                labels[i] = LABEL2ID["I-ASP"]

    # Opinion span
    opn_pos = _find_span_positions(text, opinion)
    if opn_pos:
        ts, te = _char_to_token_span(offset_mapping, opn_pos[0], opn_pos[1])
        if ts is not None and te is not None:
            labels[ts] = LABEL2ID["B-OPN"]
            for i in range(ts + 1, te + 1):
                labels[i] = LABEL2ID["I-OPN"]

    return labels


# ── spaCy dependency graph ───────────────────────────────────────────────────

def build_dep_graph(text: str, nlp, offset_mapping):
    """
    Build adjacency matrix [seq_len × seq_len] from spaCy dependency tree.
    Each spaCy token is mapped to the corresponding sub-word token indices.
    Self-loops included.
    """
    seq_len = len(offset_mapping)
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)

    try:
        doc = nlp(text)
    except Exception:
        # fallback: identity matrix
        np.fill_diagonal(adj, 1.0)
        return adj

    # Map each spaCy token's char span to the FIRST sub-word token index
    spacy_to_tok = {}
    for spacy_tok in doc:
        cs, ce = spacy_tok.idx, spacy_tok.idx + len(spacy_tok.text)
        for t_idx, (ts, te) in enumerate(offset_mapping):
            if ts == 0 and te == 0:
                continue
            if ts >= cs and te <= ce:
                if spacy_tok.i not in spacy_to_tok:
                    spacy_to_tok[spacy_tok.i] = t_idx
                break

    # Add edges (undirected)
    for spacy_tok in doc:
        head = spacy_tok.head
        t1 = spacy_to_tok.get(spacy_tok.i)
        t2 = spacy_to_tok.get(head.i)
        if t1 is not None and t2 is not None:
            adj[t1][t2] = 1.0
            adj[t2][t1] = 1.0

    # Self-loops
    np.fill_diagonal(adj, 1.0)

    # Row-normalize (D^{-1} A)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj = adj / row_sums

    return adj


# ── Dataset ──────────────────────────────────────────────────────────────────

class ASTEDataset(Dataset):
    """
    Each item is ONE quadruplet (not one sentence).
    Multiple quadruplets from the same sentence share the same tokenisation
    but have different span labels and VA targets.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_len: int = MAX_SEQ_LEN,
        spacy_model: str = "en_core_web_sm",
        use_dep_graph: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.use_dep_graph = use_dep_graph
        self.nlp = spacy.load(spacy_model) if use_dep_graph else None

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
                for quad in record["Quadruplet"]:
                    v, a = map(float, quad["VA"].split("#"))
                    self.samples.append({
                        "id":      record["ID"],
                        "text":    text,
                        "aspect":  quad.get("Aspect", "NULL"),
                        "opinion": quad.get("Opinion", "NULL"),
                        "valence": v,
                        "arousal": a,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text    = s["text"]
        aspect  = s["aspect"]
        opinion = s["opinion"]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offset_mapping = enc["offset_mapping"].squeeze(0).tolist()

        # BIO labels
        bio = build_bio_labels(offset_mapping, text, aspect, opinion)
        bio_tensor = torch.tensor(bio, dtype=torch.long)

        # Dependency adjacency matrix
        if self.use_dep_graph and self.nlp is not None:
            adj = build_dep_graph(text, self.nlp, offset_mapping)
        else:
            adj = np.eye(self.max_len, dtype=np.float32)
        adj_tensor = torch.tensor(adj, dtype=torch.float)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "adj":            adj_tensor,
            "span_labels":    bio_tensor,
            "valence":        torch.tensor(s["valence"], dtype=torch.float),
            "arousal":        torch.tensor(s["arousal"], dtype=torch.float),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_path: str,
    val_split: float = 0.1,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 16,
    num_workers: int = 2,
    use_dep_graph: bool = True,
):
    full_ds = ASTEDataset(
        train_path,
        tokenizer_name=tokenizer_name,
        use_dep_graph=use_dep_graph,
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
    print("Testing data loader …")
    tr, vl = get_dataloaders("merged_train.jsonl", batch_size=4)
    batch = next(iter(tr))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
    print(f"Train batches: {len(tr)}, Val batches: {len(vl)}")
