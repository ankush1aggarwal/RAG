"""Learned / configured weights for dense + BM25 RRF fusion."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class FusionWeights(BaseModel):
    """Non-negative weights for [dense, BM25] ranked lists in weighted RRF."""

    dense: float = Field(default=0.5, ge=0.0)
    bm25: float = Field(default=0.5, ge=0.0)
    metric_optimized: str = "ndcg@10"
    qrels_split: str = "validation"
    num_queries: int = 0
    score_at_selection: float = 0.0

    def normalized(self) -> tuple[float, float]:
        total = self.dense + self.bm25
        if total <= 0.0:
            return 1.0, 1.0
        return self.dense / total, self.bm25 / total

    def as_list(self) -> list[float]:
        dense, bm25 = self.normalized()
        return [dense, bm25]


def save_fusion_weights(path: str | Path, weights: FusionWeights) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(weights.model_dump_json(indent=2), encoding="utf-8")


def load_fusion_weights(path: str | Path) -> FusionWeights:
    return FusionWeights.model_validate_json(Path(path).read_text(encoding="utf-8"))
