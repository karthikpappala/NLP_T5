"""
merge_data.py
Merges laptop_train.jsonl and restaurant_train.jsonl into merged_train.jsonl.
Adds a 'domain' field to each record.
"""
import json, os

SOURCES = [
    ("laptop_train.jsonl",     "laptop"),
    ("restaurant_train.jsonl", "restaurant"),
]
OUTPUT = "merged_train.jsonl"

merged = []
for fname, domain in SOURCES:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} not found. Run from the project directory.")
    with open(fname, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["domain"] = domain
            merged.append(record)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r) + "\n")

print(f"Merged {len(merged)} records → {OUTPUT}")
