"""Embedding reward server for RWML GRPO.

Runs on a GPU that the veRL trainer is not using; the custom reward function
(rwml_reward.py) calls /distances during training. Batched per request.

Usage:
    .venv-verl/bin/python scripts/rwml/embed_server.py --device cuda:0 --port 8901
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_scorer import DEFAULT_MODEL, EmbeddingScorer

app = FastAPI()
scorer: EmbeddingScorer | None = None


class DistanceRequest(BaseModel):
    predictions: list[str]
    references: list[str]


@app.post("/distances")
def distances(request: DistanceRequest) -> dict:
    return {"distances": scorer.cosine_distances(request.predictions, request.references)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    global scorer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    scorer = EmbeddingScorer(args.model, device=args.device, max_length=args.max_length)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
