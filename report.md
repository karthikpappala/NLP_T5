# Model Description Report — ASTE Assignment (T5 Migration)

## 1. Task Overview

This submission addresses the extended Aspect Sentiment Triplet Extraction (ASTE) task, requiring extraction of **(Aspect, Opinion, Valence-Arousal)** triplets from natural language text across two domains: laptop reviews and restaurant reviews.

---

## 2. Dataset

| Split | Sentences | Quadruplets |
|---|---|---|
| Laptop train | 4,076 | 5,773 |
| Restaurant train | 2,284 | 3,659 |
| **Merged total** | **6,360** | **9,432** |
| Training (90%) | ~5,724 | ~8,489 |
| Validation (10%) | ~636 | ~943 |

Key observations:
- ~22.6% of aspects and ~24.2% of opinions are implicit (NULL)
- Because of these implicit nulls and the necessity of binding aspects to opinions, a sequence-to-sequence generative paradigm is a better fit than a token-level extractive paradigm.

---

## 3. Model Architecture

### 3.1 Seq2Seq Generator — T5-small

We use **T5-small** (`google-t5/t5-small`, 60M parameters) as a generative sequence-to-sequence model. This architecture formulates ASTE as a direct text generation problem, avoiding the pitfalls of BIO-tagging such as subword boundary mapping errors and cartesian product false positives.

**Input Format:**
Sentences are prefixed with a task prompt:
`extract aspect opinion sentiment: [Sentence]`

**Output Target Format:**
A linearized string grouping all triplets for the given sentence:
`( aspect | opinion | V | A ) ; ( aspect | opinion | V | A )`

Implicit (NULL) aspects and opinions are explicitly generated as the literal text `"NULL"`.

### 3.2 Decoding and Parsing

During inference, beam search is used with `num_beams=2` or `4`. The generated output string is then aggressively parsed:
- Split by `;` to get individual triplets
- Split by `|` to extract `[Aspect, Opinion, Valence, Arousal]` fields
- Fallback logic ensures that malformed strings yield a default valid output (e.g., `( NULL | NULL | 5.00 | 5.00 )`).

---

## 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Model | google-t5/t5-small |
| Max Input Length | 128 |
| Max Target Length | 256 |
| Batch Size | 8 |
| Gradient Accumulation | 2 (Effective BS = 16) |
| Epochs | 5 |
| Learning rate | 3e-4 |
| Warmup Ratio | 10% |
| Validation Metric | Triplet-level F1 |
| Early Stopping Patience | 5 epochs |

---

## 5. Results

After 5 epochs of training on CPU, the generative **T5-small** model drastically outperformed the initial ModernBERT token-classification baseline.

| Epoch | Val Triplet F1 | Val Precision | Val Recall |
|---|---|---|---|
| Initial ModernBERT Baseline | ~0.5900 | — | — |
| T5-small Epoch 1 | 0.5504 | 0.5681 | 0.5338 |
| T5-small Epoch 2 | 0.6313 | 0.6337 | 0.6290 |
| T5-small Epoch 3 | 0.6397 | 0.6366 | 0.6427 |
| **T5-small Best (Epoch 4)** | **0.6702** | **0.6713** | **0.6691** |

By eliminating BIO span constraints and naturally structuring the output as discrete semantic triplets, the generative sequence-to-sequence approach raised the macro F1 to **0.67**.

---

## 6. Academic Integrity

This work was completed individually in accordance with the course Academic Integrity Policy.
