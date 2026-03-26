"""
model.py
T5-based Aspect Sentiment Triplet Extraction (ASTE) model.

Uses T5ForConditionalGeneration to directly generate structured triplets
from input text in a seq2seq fashion.

Input:  "extract aspect opinion sentiment: <sentence>"
Output: "( aspect | opinion | valence | arousal ) ; ( ... )"
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ASTEModel:
    """
    Thin wrapper around T5ForConditionalGeneration.
    Provides consistent interface for training and inference.
    """

    def __init__(
        self,
        model_name: str = "google-t5/t5-base",
        max_target_len: int = 256,
    ):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_target_len = max_target_len

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for training.
        Returns loss when labels are provided.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, num_beams=4):
        """
        Generate triplet text for inference.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_target_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=0,
        )
        return outputs

    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_decode(self, token_ids):
        """Decode batch of token IDs to list of strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


if __name__ == "__main__":
    model = ASTEModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Smoke test
    tokenizer = model.tokenizer
    text = "extract aspect opinion sentiment: The food was great but service was slow."
    enc = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

    # Training forward pass
    target = "( food | great | 7.50 | 6.00 ) ; ( service | slow | 3.00 | 5.50 )"
    tgt_enc = tokenizer(target, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    labels = tgt_enc["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model.forward(enc["input_ids"], enc["attention_mask"], labels=labels)
    print(f"Loss: {outputs.loss.item():.4f}")

    # Generation
    gen_ids = model.generate(enc["input_ids"], enc["attention_mask"])
    print(f"Generated: {model.decode(gen_ids[0])}")
