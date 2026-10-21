"""Per-document chunk caps for rerank candidate pools (E3)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from app.ingestion.schemas import SearchHit

E3ArmMode = Literal["control", "dedup_no_backfill", "overfetch_cap"]


@dataclass(frozen=True)
class E3Arm:
    """One E3 experiment arm — how to build the pre-rerank chunk pool."""

    arm_id: str
    label: str
    mode: E3ArmMode
    rerank_budget: int = 50
    overfetch_limit: int = 150
    per_doc_cap: int | None = None

    def first_stage_retrieve_limit(self) -> int:
        if self.mode == "overfetch_cap":
            return self.overfetch_limit
        return self.rerank_budget


def duplicate_slot_count(hits: list[SearchHit]) -> int:
    """Chunks beyond the first per ``doc_id`` in pool order."""
    counts = Counter(hit.doc_id for hit in hits)
    return sum(count - 1 for count in counts.values() if count > 1)


@dataclass(frozen=True)
class CandidatePoolDiagnostics:
    pool_size: int
    unique_doc_count: int
    duplicate_slots: int

    @property
    def duplicate_slot_rate(self) -> float:
        if self.pool_size <= 0:
            return 0.0
        return self.duplicate_slots / self.pool_size


def pool_diagnostics(hits: list[SearchHit]) -> CandidatePoolDiagnostics:
    return CandidatePoolDiagnostics(
        pool_size=len(hits),
        unique_doc_count=len({hit.doc_id for hit in hits}),
        duplicate_slots=duplicate_slot_count(hits),
    )


def _dedup_no_backfill(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out: list[SearchHit] = []
    for hit in hits:
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        out.append(hit)
    return out


def _overfetch_with_cap(
    hits: list[SearchHit],
    *,
    budget: int,
    per_doc_cap: int,
) -> list[SearchHit]:
    if per_doc_cap < 1:
        raise ValueError("per_doc_cap must be at least 1")
    counts: dict[str, int] = {}
    out: list[SearchHit] = []
    for hit in hits:
        if len(out) >= budget:
            break
        n = counts.get(hit.doc_id, 0)
        if n >= per_doc_cap:
            continue
        counts[hit.doc_id] = n + 1
        out.append(hit)
    return out


def shape_candidate_pool(hits: list[SearchHit], arm: E3Arm) -> list[SearchHit]:
    """Apply E3 pool policy after dense (or other) first-stage retrieval."""
    if arm.mode == "control":
        return hits[: arm.rerank_budget]
    if arm.mode == "dedup_no_backfill":
        return _dedup_no_backfill(hits[: arm.rerank_budget])
    if arm.mode == "overfetch_cap":
        if arm.per_doc_cap is None:
            raise ValueError("overfetch_cap requires per_doc_cap")
        return _overfetch_with_cap(
            hits,
            budget=arm.rerank_budget,
            per_doc_cap=arm.per_doc_cap,
        )
    raise ValueError(f"Unknown E3 arm mode: {arm.mode}")


E3_ARMS: tuple[E3Arm, ...] = (
    E3Arm(
        arm_id="control",
        label="Control — top 50 dense, no cap",
        mode="control",
    ),
    E3Arm(
        arm_id="dedup_top50",
        label="Diagnostic — dedup top 50 (no backfill)",
        mode="dedup_no_backfill",
    ),
    E3Arm(
        arm_id="overfetch_cap1",
        label="Overfetch 150, cap 1 per doc_id, fill to 50",
        mode="overfetch_cap",
        per_doc_cap=1,
    ),
    E3Arm(
        arm_id="overfetch_cap2",
        label="Overfetch 150, cap 2 per doc_id, fill to 50",
        mode="overfetch_cap",
        per_doc_cap=2,
    ),
)


def e3_arm_by_id(arm_id: str) -> E3Arm:
    for arm in E3_ARMS:
        if arm.arm_id == arm_id:
            return arm
    known = ", ".join(a.arm_id for a in E3_ARMS)
    raise ValueError(f"Unknown E3 arm_id '{arm_id}'. Choose from: {known}")
