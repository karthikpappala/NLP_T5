# Model Description Report — ASTE Assignment

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
- Mean sentence length: 15.5 words (favourable for BERT-style models)
- ~22.6% of aspects and ~24.2% of opinions are implicit (NULL)
- Valence: mean 6.05, std 1.79 (broad range, good regression signal)
- Arousal: mean 6.74, std 1.08 (narrower — harder to predict)

---

## 3. Model Architecture

### 3.1 Encoder — ModernBERT-base

We use **ModernBERT-base** (`answerdotai/ModernBERT-base`, 149M parameters) as the contextual encoder. ModernBERT is a recent masked language model trained with rotary position embeddings and FlashAttention, offering improved efficiency and performance over BERT-base on a wide range of NLP benchmarks.

Input sentences are tokenised with the ModernBERT tokeniser (WordPiece), padded/truncated to 128 tokens.

### 3.2 Dependency Graph — spaCy

Each sentence is parsed with spaCy's `en_core_web_sm` dependency parser. The resulting dependency tree is converted into a **128×128 adjacency matrix** with:
- Undirected edges for each dependency arc
- Self-loops on every node
- Row-normalised (D⁻¹A) before passing to the GCN

### 3.3 Graph Convolutional Network (GCN)

Two GCN layers propagate syntactic context over the dependency graph:

```
H' = ReLU( D⁻¹ A H W )
```

Each GCN layer is followed by LayerNorm and a residual connection from the input, preventing over-smoothing.

### 3.4 Task Heads

**Span Classifier** (token-level):
- Dropout → Linear(768 → 5)
- Labels: `O`, `B-ASP`, `I-ASP`, `B-OPN`, `I-OPN`
- Loss: CrossEntropy (ignoring padding tokens)

**VA Regressor** (sentence-level):
- Mean-pool over non-padding tokens
- Linear(768 → 384) → GELU → Dropout → Linear(384 → 2) → Sigmoid → scale to [1, 9]
- Loss: MSE

### 3.5 Combined Loss

```
L = λ_span × CE(span) + λ_va × MSE(VA)
```
Default: λ_span = 1.0, λ_va = 0.5

---

## 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Encoder | answerdotai/ModernBERT-base |
| Max sequence length | 128 |
| Batch size | 16 |
| Epochs | 10 |
| Learning rate (encoder) | 2e-5 |
| Learning rate (heads) | 2e-4 |
| Warmup ratio | 10% |
| Weight decay | 0.01 |
| Mixed precision | FP16 (AMP) |
| Gradient clipping | 1.0 |
| GCN layers | 2 |
| Validation split | 10% |

---

## 5. Inference

Given a test sentence:
1. Tokenise with ModernBERT tokeniser
2. Build dependency graph with spaCy
3. Forward pass → span logits + VA prediction
4. Decode BIO labels → aspect/opinion surface strings (NULL if no span detected)
5. Clamp VA to [1.00, 9.00]; format as `V#A`

---

## 6. Results

*(Fill in after training)*

| Metric | Value |
|---|---|
| Val span F1 | — |
| Val valence MAE | — |
| Val arousal MAE | — |
| Best epoch | — |

---

## 7. Academic Integrity

This work was completed individually in accordance with the course Academic Integrity Policy. Ideas were discussed with peers verbally; no written or electronic notes were shared.
