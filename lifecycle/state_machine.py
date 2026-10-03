"""Phase 5: decision lifecycle WAIT -> PROVISIONAL -> CONFIRMED with conflict-gated REOPEN.

Per checkpoint k the controller receives the calibrated verifier distribution p_k and
the evidence store after adding the posts that arrived since checkpoint k-1.

    WAIT         -> PROVISIONAL(y)  if max signal >= tau_prov, evidence_units >= min_units_prov,
                                    and argmax stable for `stable` consecutive checkpoints
    PROVISIONAL  -> CONFIRMED       if p(y) >= tau_conf, evidence_units >= min_units_conf, stable,
                                    and new conflicting evidence is below half the reopen bar
                                    (consistency check)
    PROVISIONAL / CONFIRMED -> REOPENED   on the gate (see ``gate``); the commitment is
                                    invalidated and re-verification runs immediately on the
                                    current snapshot, so even a contradiction at the last
                                    checkpoint gets a re-verified outcome
    REOPENED     -> PROVISIONAL(y') if the re-verified max >= tau_prov and evidence suffices,
                    else WAIT

``gate`` selects the revision trigger:
    "conflict"           meaningful contradiction from lifecycle.evidence.ConflictDetector (full system)
    "prediction_change"  argmax of the verifier differs from the committed label
                         (revision without conflict gating; the old decision/controller.py rule)
    "conflict+change"    either of the above, the latter only when the new label is stable
                         and confident (ablation)
    "none"               commitments are never revisited (early decision, no revision)

After the first reopen, the decision signal becomes the evidence-aware re-verified
posterior (``reverify="bayes"``) instead of the raw verifier output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .evidence import ConflictConfig, ConflictDetector, EvidenceModel, EvidenceStore, reverify

WAIT, PROVISIONAL, CONFIRMED, REOPENED = "WAIT", "PROVISIONAL", "CONFIRMED", "REOPENED"


@dataclass(frozen=True)
class LifecycleConfig:
    tau_prov: float = 0.6
    tau_conf: float = 0.75
    min_units_prov: int = 1
    min_units_conf: int = 3
    stable: int = 2
    gate: str = "conflict"
    reopen_confirmed: bool = True
    reverify: str = "bayes"           # "bayes" | "verifier"
    gamma: float = 0.3
    final_only: bool = False          # baseline (a): decide once, at the last checkpoint
    dedup: bool = True
    conflict: ConflictConfig = ConflictConfig()

    def with_(self, **kwargs):
        conflict_keys = {k: kwargs.pop(k) for k in list(kwargs) if k in ConflictConfig.__dataclass_fields__}
        cfg = replace(self, **kwargs)
        return replace(cfg, conflict=replace(cfg.conflict, **conflict_keys)) if conflict_keys else cfg


class Lifecycle:
    """Runs one thread. Call ``observe`` for each new post, then ``decide`` once per checkpoint."""

    def __init__(self, config: LifecycleConfig, evidence_model: EvidenceModel | None):
        self.config = config
        self.model = evidence_model
        self.detector = ConflictDetector(evidence_model, config.conflict) if evidence_model is not None else None
        self.store = EvidenceStore(dedup=config.dedup)
        self.state, self.label = WAIT, None
        self.commit_since = 0       # evidence index from which clusters count as "new"
        self.reopened = False
        self.last_argmax, self.stable_count = None, 0
        self.log: list[dict] = []

    def observe(self, index, text, vector, stance, *, is_source=False, time=0.0):
        self.store.add(index, text, vector, stance, is_source=is_source, time=time)

    def _signal(self, p):
        if self.reopened and self.config.reverify == "bayes" and self.model is not None:
            return reverify(p, self.store, self.model, self.config.gamma)
        return p

    def _commit(self, label, next_index):
        self.state, self.label, self.commit_since = PROVISIONAL, int(label), next_index

    def decide(self, k: int, p: np.ndarray, next_index: int, *, last: bool = False, meta: dict | None = None):
        """``next_index``: chronological index of the first post *not* yet observed."""
        c = self.config
        events, assessment = [], None
        units = self.store.evidence_units
        if c.final_only:
            if last:
                self.state, self.label = CONFIRMED, int(np.argmax(p))
            return self._record(k, p, p, events, None, meta)
        signal = self._signal(p)
        top = int(np.argmax(signal))
        self.stable_count = self.stable_count + 1 if top == self.last_argmax else 1
        self.last_argmax = top

        if self.state in (PROVISIONAL, CONFIRMED):
            trigger = False
            if c.gate in ("conflict", "conflict+change") and self.detector is not None:
                assessment = self.detector.assess(self.store, self.label, self.commit_since, confirmed=self.state == CONFIRMED)
                trigger = assessment.triggered
                if c.gate == "conflict+change":
                    trigger = trigger or (top != self.label and signal[top] >= c.tau_prov and self.stable_count >= c.stable)
                trigger = trigger and (self.state == PROVISIONAL or c.reopen_confirmed)
            elif c.gate == "prediction_change":
                trigger = top != self.label and (self.state == PROVISIONAL or c.reopen_confirmed)
            if trigger:
                events.append({"type": "reopen", "from": self.state, "label": self.label})
                self.state, self.label, self.reopened = REOPENED, None, True
                signal = self._signal(p)
                top = int(np.argmax(signal))
                self.last_argmax, self.stable_count = top, 1
                if signal[top] >= c.tau_prov and units >= c.min_units_prov:
                    self._commit(top, next_index)
                    events.append({"type": "reverified", "label": top})
                else:
                    self.state = WAIT
                    events.append({"type": "reverified", "label": None})
                return self._record(k, p, signal, events, assessment, meta)

        if self.state == WAIT:
            if signal[top] >= c.tau_prov and units >= c.min_units_prov and self.stable_count >= c.stable:
                self._commit(top, next_index)
                events.append({"type": "commit", "label": top})
        elif self.state == PROVISIONAL:
            consistent = assessment is None or assessment.net < 0.5 * c.conflict.theta_net
            if (top == self.label and signal[top] >= c.tau_conf and units >= c.min_units_conf
                    and self.stable_count >= c.stable and consistent):
                self.state = CONFIRMED
                events.append({"type": "confirm", "label": self.label})
        return self._record(k, p, signal, events, assessment, meta)

    def _record(self, k, p, signal, events, assessment, meta):
        row = {"checkpoint": k, "state": self.state, "label": self.label, "verifier_pred": int(np.argmax(p)),
               "verifier_probs": np.round(p, 4).tolist(), "signal_probs": np.round(signal, 4).tolist(),
               "evidence_units": self.store.evidence_units, "events": events,
               "conflict": assessment.as_dict() if assessment is not None else None}
        if meta:
            row.update(meta)
        self.log.append(row)
        return row


def run_thread(config, evidence_model, thread, embeddings, stance, verifier_probs, checkpoint_lengths, override_posts=None):
    """Replay one thread through the lifecycle. ``stance``: [n, 4] stance probabilities.

    ``override_posts`` (optional) maps checkpoint index -> list of extra evidence items
    (dicts with text, vector, stance) injected *after* that checkpoint's real posts."""
    life = Lifecycle(config, evidence_model)
    seen = 0
    K = len(checkpoint_lengths)
    for k, m in enumerate(checkpoint_lengths):
        for i in range(seen, m):
            life.observe(i, thread.texts[i], embeddings[thread.emb_rows[i]], stance[i], is_source=bool(thread.is_source[i]), time=float(thread.rel_time[i]))
        seen = max(seen, m)
        next_index = m
        for j, item in enumerate((override_posts or {}).get(k, [])):
            life.observe(thread.n + 1000 * k + j, item["text"], item["vector"], item["stance"])
            next_index = thread.n + 1000 * k + j + 1
        life.decide(k, verifier_probs[k], next_index, last=k == K - 1,
                    meta={"num_posts": m, "elapsed_min": round(float(thread.rel_time[m - 1]) / 60.0, 2)})
    return life.log
