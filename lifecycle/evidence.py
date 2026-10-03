"""Phase 4 (b): per-checkpoint evidence store and conflict detector.

Evidence store
    Every observed reply becomes an evidence item (post id, chronological index,
    time, stance probabilities, normalised text, unit-norm embedding). Items are
    grouped into near-duplicate clusters (same normalised text, or cosine >= dup_cos
    on BERTweet vectors). A cluster counts once; its stance is the member mean.
    The store snapshots a summary at every checkpoint.

Evidence model (fitted on the training fold only)
    P(stance | label) from predicted stance over training replies. For a committed
    label L, the per-stance log-likelihood ratio LLR(s, L) = log P(s|not L) - log P(s|L)
    says how much stance s argues *against* L (positive) or *for* L (negative).

Conflict detector (relative to a committed label L and the commitment point)
    Only clusters first seen after the commitment are "new"; a repost of evidence
    that already existed at commit time is not new.
      conflict_j = sum_s p_j(s) * max(0,  LLR(s, L))     (nats)
      support_j  = sum_s p_j(s) * max(0, -LLR(s, L))     (nats)
      strength   = max_j conflict_j / max_s LLR(s, L)     (semantic strength in [0, 1])
      volume     = #{ j : conflict_j / max_s LLR >= weak }  (distinct contradicting clusters)
      net        = sum of conflict_j over clusters with normalised conflict >= weak
                   - beta * (sum of support_j over clusters with normalised support >= weak)
    "Meaningful contradiction" := semantic path OR volume path
        semantic path: strength >= theta_strong AND net >= 0
                       (one confident contradicting post suffices unless support outweighs it)
        volume path:   volume >= v_min AND net >= theta_net * scale
                       (many *distinct* weaker posts; duplicates cannot satisfy it)
    where scale = kappa for CONFIRMED (hysteresis) and 1 for PROVISIONAL.
    Design note: an earlier version also required net >= theta_net on the semantic
    path; on PHEME the largest per-post LLR is ~0.2-0.5 nats, which made a single
    strong post unable to trigger, contradicting edge case 4. Changed after one
    smoke-test fold, before the full runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .data import STANCES

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def normalise_text(text: str) -> str:
    text = _MENTION.sub(" ", _URL.sub(" ", text.lower()))
    text = re.sub(r"\brt\b", " ", text)
    return " ".join(_NONWORD.sub(" ", text).split())


@dataclass
class Cluster:
    first_index: int
    key: str
    vector: np.ndarray
    members: list[int] = field(default_factory=list)
    stance_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(STANCES)))

    @property
    def stance(self) -> np.ndarray:
        return self.stance_sum / max(len(self.members), 1)


class EvidenceStore:
    """Accumulated evidence for one thread. ``add`` must be called in chronological order."""

    def __init__(self, dedup: bool = True, dup_cos: float = 0.95, min_key_tokens: int = 3):
        self.dedup, self.dup_cos, self.min_key_tokens = dedup, dup_cos, min_key_tokens
        self.clusters: list[Cluster] = []
        self.items: list[dict] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)   # cluster representative vectors
        self._by_key: dict[str, int] = {}
        self.source_seen = False
        self.history: list[dict] = []

    def add(self, index: int, text: str, vector: np.ndarray, stance: np.ndarray, *, is_source: bool = False, time: float = 0.0):
        if is_source:
            self.source_seen = True
            return None
        vector = np.asarray(vector, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
        key = normalise_text(text)
        position = None
        if self.dedup and self.clusters:
            if len(key.split()) >= self.min_key_tokens:
                position = self._by_key.get(key)
            if position is None:
                cosine = self._matrix @ vector
                best = int(np.argmax(cosine))
                position = best if cosine[best] >= self.dup_cos else None
        if position is None:
            position = len(self.clusters)
            self.clusters.append(Cluster(first_index=index, key=key, vector=vector))
            self._matrix = np.vstack([self._matrix.reshape(-1, len(vector)), vector[None]])
            if len(key.split()) >= self.min_key_tokens:
                self._by_key.setdefault(key, position)
        target = self.clusters[position]
        target.members.append(index)
        target.stance_sum = target.stance_sum + np.asarray(stance, dtype=np.float64)
        self.items.append({"index": index, "time": time, "cluster": position})
        return target

    @property
    def evidence_units(self) -> int:
        """Distinct pieces of evidence: the source post plus distinct reply clusters."""
        return int(self.source_seen) + len(self.clusters)

    def stance_counts(self, deduplicated=True) -> np.ndarray:
        if deduplicated:
            return np.sum([c.stance for c in self.clusters], axis=0) if self.clusters else np.zeros(len(STANCES))
        return np.sum([c.stance_sum for c in self.clusters], axis=0) if self.clusters else np.zeros(len(STANCES))

    def snapshot(self, checkpoint: int):
        record = {"checkpoint": checkpoint, "items": len(self.items), "clusters": len(self.clusters),
                  "evidence_units": self.evidence_units, "stance_dedup": self.stance_counts(True).round(3).tolist()}
        self.history.append(record)
        return record


class EvidenceModel:
    """P(stance | label) and the per-label contradiction LLR table, fitted on training threads."""

    def __init__(self, stance_given_label: np.ndarray, prior: np.ndarray):
        self.p = stance_given_label / stance_given_label.sum(1, keepdims=True)  # [C, S]
        self.prior = prior / prior.sum()
        C = self.p.shape[0]
        self.llr = np.zeros_like(self.p)
        for L in range(C):
            others = [y for y in range(C) if y != L]
            w = self.prior[others] / self.prior[others].sum()
            self.llr[L] = np.log(w @ self.p[others]) - np.log(self.p[L])

    @classmethod
    def fit(cls, stance_probs_by_thread, labels, num_classes, smoothing=1.0):
        counts = np.full((num_classes, len(STANCES)), smoothing)
        prior = np.bincount(labels, minlength=num_classes).astype(float) + 1
        for probs, y in zip(stance_probs_by_thread, labels):
            counts[y] += probs.sum(0)
        return cls(counts, prior)

    def against(self, L: int) -> np.ndarray:
        return np.maximum(self.llr[L], 0.0)

    def towards(self, L: int) -> np.ndarray:
        return np.maximum(-self.llr[L], 0.0)

    def log_likelihood(self, stance: np.ndarray) -> np.ndarray:
        """log P(evidence | y) for one stance distribution, per label."""
        return np.log(self.p) @ stance


@dataclass(frozen=True)
class ConflictConfig:
    theta_net: float = 0.5
    theta_strong: float = 0.7
    weak: float = 0.3
    v_min: int = 3
    beta_support: float = 1.0
    kappa_confirmed: float = 1.5
    use_strength: bool = True
    use_volume: bool = True


@dataclass
class ConflictAssessment:
    strength: float
    volume: int
    conflict_nats: float
    support_nats: float
    net: float
    new_clusters: int
    triggered: bool

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class ConflictDetector:
    def __init__(self, model: EvidenceModel, config: ConflictConfig = ConflictConfig()):
        self.model, self.config = model, config

    def assess(self, store: EvidenceStore, label: int, since_index: int, *, confirmed: bool = False) -> ConflictAssessment:
        c = self.config
        against, towards = self.model.against(label), self.model.towards(label)
        max_against = max(float(against.max()), 1e-8)
        max_towards = max(float(towards.max()), 1e-8)
        new = [cl for cl in store.clusters if cl.first_index >= since_index]
        conflict = np.array([float(cl.stance @ against) for cl in new]) if new else np.zeros(0)
        support = np.array([float(cl.stance @ towards) for cl in new]) if new else np.zeros(0)
        # Only stance-bearing clusters enter the sums, so a mass of near-neutral comments
        # cannot accumulate into conflict (or into support that masks a strong denial).
        conflict_on = conflict / max_against >= c.weak
        support_on = support / max_towards >= c.weak
        strength = float(conflict.max() / max_against) if len(new) else 0.0
        volume = int(conflict_on.sum())
        net = float(conflict[conflict_on].sum() - c.beta_support * support[support_on].sum())
        scale = c.kappa_confirmed if confirmed else 1.0
        semantic = c.use_strength and strength >= c.theta_strong and net >= 0.0
        by_volume = c.use_volume and volume >= c.v_min and net >= c.theta_net * scale
        triggered = bool(semantic or by_volume)
        return ConflictAssessment(strength, volume, float(conflict[conflict_on].sum()), float(support[support_on].sum()), net, len(new), triggered)


def reverify(verifier_probs: np.ndarray, store: EvidenceStore, model: EvidenceModel, gamma: float) -> np.ndarray:
    """Evidence-aware re-verification: verifier posterior times deduplicated stance likelihood.

    log q(y) = log p_verifier(y) + gamma * sum_clusters log P(stance_j | y)."""
    log_q = np.log(np.clip(verifier_probs, 1e-8, 1.0))
    for cl in store.clusters:
        log_q = log_q + gamma * model.log_likelihood(cl.stance)
    log_q -= log_q.max()
    q = np.exp(log_q)
    return q / q.sum()
