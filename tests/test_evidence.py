"""Phase 4 unit tests on hand-built toy evidence (no trained models involved)."""

import unittest

import numpy as np

from lifecycle.evidence import ConflictConfig, ConflictDetector, EvidenceModel, EvidenceStore, normalise_text, reverify

FALSE, TRUE = 0, 1
STRONG_DENY = np.array([0.0, 0.95, 0.05, 0.0])
WEAK_DENY = np.array([0.10, 0.30, 0.20, 0.40])
SUPPORT = np.array([0.90, 0.0, 0.0, 0.10])
COMMENT = np.array([0.05, 0.05, 0.05, 0.85])


def toy_model():
    # P(stance | label) over (support, deny, query, comment)
    return EvidenceModel(np.array([[0.10, 0.30, 0.15, 0.45],    # false
                                   [0.32, 0.03, 0.15, 0.50]]),  # true
                         prior=np.array([1.0, 1.0]))


def vec(seed, dim=64):
    return np.random.default_rng(seed).normal(size=dim)


CONFIG = ConflictConfig(theta_net=1.0, theta_strong=0.8, weak=0.25, v_min=3, kappa_confirmed=1.5)


class EvidenceStoreTests(unittest.TestCase):
    def test_exact_and_near_duplicates_collapse(self):
        store = EvidenceStore()
        base = vec(1)
        store.add(1, "This is FAKE!!! http://t.co/x @bob", base, WEAK_DENY)
        store.add(2, "this is fake @alice", vec(2), WEAK_DENY)              # same normalised text
        store.add(3, "different words entirely here", base + 0.01 * vec(3), WEAK_DENY)  # near-identical vector
        store.add(4, "a genuinely new reply", vec(4), COMMENT)
        self.assertEqual(len(store.clusters), 2)
        self.assertEqual(store.evidence_units, 2)  # no source seen yet
        self.assertEqual(len(store.items), 4)

    def test_short_texts_do_not_merge_by_text(self):
        store = EvidenceStore()
        store.add(1, "@a @b", vec(1), COMMENT)
        store.add(2, "@c", vec(2), COMMENT)   # both normalise to "", must not be merged on text
        self.assertEqual(len(store.clusters), 2)

    def test_dedup_can_be_disabled(self):
        store = EvidenceStore(dedup=False)
        for i in range(3):
            store.add(i, "same text every time", vec(0), WEAK_DENY)
        self.assertEqual(len(store.clusters), 3)

    def test_normalise_text(self):
        self.assertEqual(normalise_text("RT @x: Hoax!! http://t.co/a"), "hoax")


class ConflictDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = ConflictDetector(toy_model(), CONFIG)

    def store_with(self, posts, start=10):
        store = EvidenceStore()
        store.add(0, "source", vec(99), np.zeros(4), is_source=True)
        for j, (text, v, stance) in enumerate(posts):
            store.add(start + j, text, v, stance)
        return store

    def test_llr_direction(self):
        model = toy_model()
        self.assertGreater(model.llr[TRUE][1], 0)   # deny argues against TRUE
        self.assertLess(model.llr[TRUE][0], 0)      # support argues for TRUE
        self.assertGreater(model.llr[FALSE][0], 0)  # support argues against FALSE

    def test_single_strong_post_triggers_on_semantic_path(self):
        a = self.detector.assess(self.store_with([("it is a hoax, police denied", vec(1), STRONG_DENY)]), TRUE, since_index=5)
        self.assertTrue(a.triggered)
        self.assertGreaterEqual(a.strength, 0.8)
        self.assertEqual(a.volume, 1)

    def test_single_weak_post_does_not_trigger(self):
        a = self.detector.assess(self.store_with([("hmm not sure", vec(1), WEAK_DENY)]), TRUE, since_index=5)
        self.assertFalse(a.triggered)

    def test_duplicates_count_once(self):
        dup = [("not true at all", vec(1), WEAK_DENY)] * 8
        a = self.detector.assess(self.store_with(dup), TRUE, since_index=5)
        self.assertEqual(a.new_clusters, 1)
        self.assertFalse(a.triggered)

    def test_many_distinct_weak_posts_trigger_on_volume_path(self):
        posts = [(f"distinct doubt number {i}", vec(i), WEAK_DENY) for i in range(4)]
        a = self.detector.assess(self.store_with(posts), TRUE, since_index=5)
        self.assertLess(a.strength, 0.8)
        self.assertGreaterEqual(a.volume, 3)
        self.assertTrue(a.triggered)

    def test_supporting_evidence_offsets_conflict(self):
        posts = [(f"distinct doubt number {i}", vec(i), WEAK_DENY) for i in range(4)]
        posts += [(f"confirmed by source {i}", vec(100 + i), SUPPORT) for i in range(6)]
        a = self.detector.assess(self.store_with(posts), TRUE, since_index=5)
        self.assertLess(a.net, 1.0)
        self.assertFalse(a.triggered)

    def test_evidence_before_commit_is_not_new(self):
        store = self.store_with([("it is a hoax, police denied", vec(1), STRONG_DENY)], start=1)
        store.add(20, "it is a hoax, police denied", vec(1), STRONG_DENY)  # repost after commit at index 5
        a = self.detector.assess(store, TRUE, since_index=5)
        self.assertEqual(a.new_clusters, 0)
        self.assertFalse(a.triggered)

    def test_confirmed_needs_more_net_evidence(self):
        detector = ConflictDetector(toy_model(), ConflictConfig(**{**CONFIG.__dict__, "v_min": 2}))
        store = self.store_with([(f"distinct doubt number {i}", vec(i), WEAK_DENY) for i in range(2)])
        self.assertTrue(detector.assess(store, TRUE, 5, confirmed=False).triggered)
        self.assertFalse(detector.assess(store, TRUE, 5, confirmed=True).triggered)

    def test_many_neutral_comments_do_not_mask_a_strong_denial(self):
        posts = [(f"neutral remark {i}", vec(200 + i), COMMENT) for i in range(30)]
        posts.append(("it is a hoax, police denied", vec(1), STRONG_DENY))
        a = self.detector.assess(self.store_with(posts), TRUE, since_index=5, confirmed=True)
        self.assertEqual(a.support_nats, 0.0)
        self.assertTrue(a.triggered)

    def test_single_strong_post_outweighed_by_support_does_not_trigger(self):
        posts = [("it is a hoax, police denied", vec(1), STRONG_DENY)] + [(f"confirmed by source {i}", vec(100 + i), SUPPORT) for i in range(3)]
        a = self.detector.assess(self.store_with(posts), TRUE, since_index=5)
        self.assertLess(a.net, 0.0)
        self.assertFalse(a.triggered)

    def test_strength_and_volume_paths_can_be_ablated(self):
        strong = self.store_with([("it is a hoax, police denied", vec(1), STRONG_DENY)])
        no_strength = ConflictDetector(toy_model(), ConflictConfig(**{**CONFIG.__dict__, "use_strength": False}))
        self.assertFalse(no_strength.assess(strong, TRUE, 5).triggered)
        volume = self.store_with([(f"distinct doubt number {i}", vec(i), WEAK_DENY) for i in range(4)])
        no_volume = ConflictDetector(toy_model(), ConflictConfig(**{**CONFIG.__dict__, "use_volume": False}))
        self.assertFalse(no_volume.assess(volume, TRUE, 5).triggered)

    def test_reverify_moves_towards_evidence(self):
        store = self.store_with([("it is a hoax, police denied", vec(1), STRONG_DENY), ("fake fake", vec(2), STRONG_DENY)])
        q = reverify(np.array([0.3, 0.7]), store, toy_model(), gamma=1.0)
        self.assertGreater(q[FALSE], 0.5)
        self.assertAlmostEqual(q.sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
