"""Phase 5: required edge cases of the decision lifecycle, one test (or more) each."""

import unittest

import numpy as np

from lifecycle.data import Thread
from lifecycle.evidence import ConflictConfig
from lifecycle.state_machine import CONFIRMED, PROVISIONAL, REOPENED, WAIT, LifecycleConfig, run_thread
from tests.test_evidence import COMMENT, STRONG_DENY, WEAK_DENY, toy_model

FALSE, TRUE = 0, 1
CHECKPOINTS = [1, 2, 3, 4, 5, 6]   # one checkpoint per arriving post block (prefix lengths)


def make_thread(posts):
    """posts: list of (text, stance) replies after a source post. Returns (thread, embeddings, stance)."""
    n = len(posts) + 1
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(n, 64)).astype(np.float32)
    texts = ["source claim"] + [t for t, _ in posts]
    for i, (text, _) in enumerate(posts, start=1):  # identical texts get identical vectors
        first = texts.index(text)
        embeddings[i] = embeddings[first]
    stance = np.vstack([np.zeros(4)] + [s for _, s in posts]).astype(np.float32)
    thread = Thread("r", "e", "true", [str(i) for i in range(n)], texts, np.arange(n, dtype=np.float32) * 60,
                    np.array([-1] + [0] * (n - 1)), np.array([True] + [False] * (n - 1)), np.arange(n))
    return thread, embeddings, stance


BASE = LifecycleConfig(tau_prov=0.6, tau_conf=0.8, min_units_prov=1, min_units_conf=2, stable=2, gamma=1.0,
                       conflict=ConflictConfig(theta_net=1.0, theta_strong=0.8, weak=0.25, v_min=3, kappa_confirmed=1.5))
CONFIDENT_TRUE = np.tile([0.1, 0.9], (6, 1))


def run(config, posts, probs=CONFIDENT_TRUE, lengths=CHECKPOINTS):
    thread, emb, stance = make_thread(posts)
    return run_thread(config, toy_model(), thread, emb, stance, probs, lengths)


def comments(n, offset=0):
    return [(f"neutral remark {offset + i}", COMMENT) for i in range(n)]


class EdgeCaseTests(unittest.TestCase):
    # 1. Very late contradiction after CONFIRMED
    def test_very_late_contradiction_after_confirmed_reopens_and_reverifies(self):
        log = run(BASE, comments(4) + [("police say the report is a hoax", STRONG_DENY)])
        self.assertEqual(log[-2]["state"], CONFIRMED)
        self.assertEqual(log[-1]["events"][0], {"type": "reopen", "from": CONFIRMED, "label": TRUE})
        self.assertEqual(log[-1]["events"][1]["type"], "reverified")
        self.assertNotEqual(log[-1]["state"], REOPENED)  # re-verified on the same (last) snapshot

    def test_late_contradiction_ignored_when_confirmed_is_locked(self):
        log = run(BASE.with_(reopen_confirmed=False), comments(4) + [("police say the report is a hoax", STRONG_DENY)])
        self.assertEqual((log[-1]["state"], log[-1]["label"]), (CONFIRMED, TRUE))

    # 2. Repeated / near-duplicate contradictory posts
    def test_near_duplicate_contradictions_are_not_double_counted(self):
        posts = comments(1) + [("not true", WEAK_DENY)] * 4
        log = run(BASE, posts)
        self.assertFalse(any(e["type"] == "reopen" for row in log for e in row["events"]))
        self.assertLessEqual(log[-1]["conflict"]["new_clusters"], 2)

    def test_without_dedup_duplicates_would_reopen(self):
        posts = comments(1) + [("not true", WEAK_DENY)] * 4
        log = run(BASE.with_(dedup=False, kappa_confirmed=1.0), posts)
        self.assertTrue(any(e["type"] == "reopen" for row in log for e in row["events"]))

    # 3. Insufficient evidence -> stay WAIT instead of forcing a verdict
    def test_insufficient_evidence_stays_wait(self):
        thread, emb, stance = make_thread([])
        log = run_thread(BASE.with_(min_units_prov=2), toy_model(), thread, emb, stance, CONFIDENT_TRUE, [1] * 6)
        self.assertTrue(all(row["state"] == WAIT for row in log))

    def test_low_confidence_stays_wait(self):
        log = run(BASE, comments(5), probs=np.tile([0.45, 0.55], (6, 1)))
        self.assertTrue(all(row["state"] == WAIT for row in log))

    # 4. Semantically strong contradiction from very few posts
    def test_single_strong_contradiction_reopens_provisional(self):
        probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])  # never confirms
        log = run(BASE, comments(2) + [("this was debunked, it is fake", STRONG_DENY)] + comments(2, 10), probs)
        reopen = [row for row in log if any(e["type"] == "reopen" for e in row["events"])]
        self.assertEqual(len(reopen), 1)
        self.assertEqual(reopen[0]["events"][0]["from"], PROVISIONAL)
        self.assertEqual(reopen[0]["conflict"]["volume"], 1)

    def test_few_weak_contradictions_do_not_reopen(self):
        probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
        log = run(BASE, comments(2) + [("maybe not", WEAK_DENY), ("doubt it", WEAK_DENY)] + comments(1, 10), probs)
        self.assertFalse(any(e["type"] == "reopen" for row in log for e in row["events"]))

    # Prediction instability alone is not contradiction under the conflict gate
    def test_prediction_flip_without_conflict_does_not_reopen(self):
        probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])
        conflict_log = run(BASE.with_(tau_conf=0.95), comments(5), probs)
        change_log = run(BASE.with_(tau_conf=0.95, gate="prediction_change", reverify="verifier"), comments(5), probs)
        self.assertFalse(any(e["type"] == "reopen" for row in conflict_log for e in row["events"]))
        self.assertTrue(any(e["type"] == "reopen" for row in change_log for e in row["events"]))
        self.assertEqual(change_log[-1]["label"], FALSE)

    def test_final_only_baseline_decides_once(self):
        log = run(BASE.with_(final_only=True), comments(5))
        self.assertTrue(all(row["state"] == WAIT for row in log[:-1]))
        self.assertEqual((log[-1]["state"], log[-1]["label"]), (CONFIRMED, TRUE))

    def test_no_revision_baseline_never_changes(self):
        log = run(BASE.with_(gate="none"), comments(4) + [("police say the report is a hoax", STRONG_DENY)])
        labels = {row["label"] for row in log if row["label"] is not None}
        self.assertEqual(labels, {TRUE})


if __name__ == "__main__":
    unittest.main()
