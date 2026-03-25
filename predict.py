"""
predict.py
Run inference on a test JSONL file and produce the required output format.

Usage:
    python predict.py --test test_data.jsonl --checkpoint checkpoints/best_model.pt
"""

import argparse
import json

import torch
from transformers import AutoTokenizer
import spacy

from data_loader import (
    LABEL2ID, ID2LABEL, MAX_SEQ_LEN, PAD_LABEL_ID,
    build_dep_graph, build_bio_labels,
)
from model import ASTEModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",       required=True,  help="Test JSONL file")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--output",     default="predictions.jsonl")
    p.add_argument("--encoder",    default="answerdotai/ModernBERT-base")
    p.add_argument("--threshold",  type=float, default=0.0,
                   help="Min logit threshold for span detection (0 = argmax only)")
    p.add_argument("--no_dep_graph", action="store_true")
    return p.parse_args()


def extract_spans(tokens, labels, offset_mapping, text):
    """
    Given BIO labels per sub-word token, extract the surface string spans.
    Returns list of (aspect_str, opinion_str).
    """
    def _extract(tag_b, tag_i):
        spans = []
        in_span, start = False, None
        for i, lbl in enumerate(labels):
            if lbl == LABEL2ID[tag_b]:
                if in_span and start is not None:
                    spans.append((start, i - 1))
                in_span, start = True, i
            elif lbl == LABEL2ID[tag_i] and in_span:
                pass
            else:
                if in_span and start is not None:
                    spans.append((start, i - 1))
                in_span, start = False, None
        if in_span and start is not None:
            spans.append((start, len(labels) - 1))

        results = []
        for ts, te in spans:
            cs = offset_mapping[ts][0]
            ce = offset_mapping[te][1]
            results.append(text[cs:ce])
        return results

    aspects  = _extract("B-ASP", "I-ASP") or ["NULL"]
    opinions = _extract("B-OPN", "I-OPN") or ["NULL"]
    return aspects, opinions


@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    encoder_name = saved_args.get("encoder", args.encoder)

    model = ASTEModel(
        encoder_name=encoder_name,
        gcn_layers=saved_args.get("gcn_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, val_f1={ckpt.get('val_f1',0):.4f})")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    nlp = spacy.load("en_core_web_sm") if not args.no_dep_graph else None

    results = []

    with open(args.test, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["Text"]

            enc = tokenizer(
                text,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            offset_mapping = enc["offset_mapping"].squeeze(0).tolist()

            if nlp:
                adj = build_dep_graph(text, nlp, offset_mapping)
            else:
                import numpy as np
                adj = np.eye(MAX_SEQ_LEN, dtype="float32")
            adj_tensor = torch.tensor(adj).unsqueeze(0).to(device)

            span_logits, va_pred = model(input_ids, attention_mask, adj_tensor)
            labels = span_logits.squeeze(0).argmax(-1).cpu().tolist()

            aspects, opinions = extract_spans(
                None, labels, offset_mapping, text
            )

            v = round(float(va_pred[0, 0].cpu()), 2)
            a = round(float(va_pred[0, 1].cpu()), 2)
            # Clamp to [1.00, 9.00]
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))

            # Build triplets (pair aspects × opinions, share same VA)
            triplets = []
            for asp in aspects:
                for opn in opinions:
                    triplets.append({
                        "Aspect":  asp,
                        "Opinion": opn,
                        "VA":      f"{v:.2f}#{a:.2f}",
                    })

            results.append({"ID": record["ID"], "Triplet": triplets})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    args = parse_args()
    predict(args)
