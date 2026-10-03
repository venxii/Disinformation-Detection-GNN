"""Independent contradiction/disagreement gate.

The score is intentionally supplied by an external stance/semantic module.
This module never treats a changed model prediction as contradictory evidence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContradictionConfig:
    score_threshold: float = 0.70
    min_new_evidence: int = 2
    min_contradictory_evidence: int = 2
    min_proportion: float = 0.50


def contradiction_gate(*, score: float, new_evidence: int, contradictory_evidence: int, config: ContradictionConfig = ContradictionConfig()) -> bool:
    """Return true only for material independent disagreement evidence."""
    if new_evidence < config.min_new_evidence or contradictory_evidence < config.min_contradictory_evidence:
        return False
    if contradictory_evidence / max(new_evidence, 1) < config.min_proportion:
        return False
    return score >= config.score_threshold
