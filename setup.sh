#!/bin/bash
set -e
echo "=== Installing dependencies ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install transformers>=4.40.0 datasets spacy scipy scikit-learn tqdm -q
python -m spacy download en_core_web_sm
echo "=== All dependencies installed ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
