"""Shared Qwen3-Embedding scorer for RWML rewards.

The RWML reward (paper Sec 2.2) is r = 1.0 if cosine_distance(E(pred), E(gold)) < tau_d
else 0.0, with E = Qwen3-Embedding-8B (paper Appendix B.1). Last-token pooling with
left padding follows the Qwen3-Embedding model card.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"


class EmbeddingScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        max_length: int = 2048,
        batch_size: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            embeddings = hidden[:, -1]  # left padding puts the last real token at -1
            chunks.append(torch.nn.functional.normalize(embeddings, p=2, dim=1).float().cpu())
        return torch.cat(chunks, dim=0)

    def cosine_distances(self, predictions: list[str], references: list[str]) -> list[float]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have equal length")
        if not predictions:
            return []
        embedded = self.embed(list(predictions) + list(references))
        pred, ref = embedded[: len(predictions)], embedded[len(predictions) :]
        return (1.0 - (pred * ref).sum(dim=1)).tolist()
