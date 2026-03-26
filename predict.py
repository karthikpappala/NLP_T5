"""
predict.py
Run T5 inference on test JSONL and produce predictions.

Usage:
    python predict.py --test test_data.jsonl --checkpoint checkpoints/best_model.pt
"""

import argparse
import json

import torch
from transformers import T5Tokenizer

from data_loader import parse_triplets, PREFIX, MAX_INPUT_LEN
from model import ASTEModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",       required=True,  help="Test JSONL file")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--output",     default="predictions.jsonl")
    p.add_argument("--encoder",    default="google-t5/t5-base")
    p.add_argument("--num_beams",  type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    encoder_name = saved_args.get("encoder", args.encoder)

    model = ASTEModel(model_name=encoder_name)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, val_f1={ckpt.get('val_f1',0):.4f})")

    tokenizer = model.tokenizer

    # Load test data
    records = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Test samples: {len(records)}")

    # Process in batches
    results = []
    for i in range(0, len(records), args.batch_size):
        batch_records = records[i:i + args.batch_size]
        input_texts = [PREFIX + r["Text"] for r in batch_records]

        enc = tokenizer(
            input_texts,
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        gen_ids = model.generate(input_ids, attention_mask, num_beams=args.num_beams)
        pred_texts = model.batch_decode(gen_ids)

        for record, pred_text in zip(batch_records, pred_texts):
            triplets = parse_triplets(pred_text)

            # If no triplets were parsed, add a default NULL triplet
            if not triplets:
                triplets = [{"Aspect": "NULL", "Opinion": "NULL", "VA": "5.00#5.00"}]

            results.append({"ID": record["ID"], "Triplet": triplets})

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} predictions → {args.output}")

    # Show some samples
    print("\nSample predictions:")
    for r in results[:5]:
        print(f"  {r['ID']}: {r['Triplet']}")


if __name__ == "__main__":
    args = parse_args()
    predict(args)
