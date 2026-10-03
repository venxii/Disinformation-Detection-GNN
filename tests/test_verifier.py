import unittest

import numpy as np
import torch

from lifecycle.data import Thread
from lifecycle.verifier import ThreadCache, TemporalVerifier, VerifierConfig, build_batch


def toy_thread(n=8, seed=0):
    rng = np.random.default_rng(seed)
    parent = np.array([-1] + [int(rng.integers(0, i)) for i in range(1, n)])
    return Thread(root_id="r", event="e", label="false", post_ids=[str(i) for i in range(n)], texts=[""] * n,
                  rel_time=np.sort(rng.uniform(0, 7200, n)).astype(np.float32) - 0, parent=parent,
                  is_source=np.array([True] + [False] * (n - 1)), emb_rows=np.arange(n))


class VerifierLeakageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TemporalVerifier(VerifierConfig(num_classes=4, embed_dim=16, hidden=16, heads=2)).eval()

    def logits(self, thread, embeddings, m):
        thread.rel_time[0] = 0.0
        return self.model(build_batch(ThreadCache([thread]), [(0, m)], embeddings))

    def test_future_posts_do_not_change_prefix_prediction(self):
        thread = toy_thread()
        embeddings = torch.randn(8, 16)
        for m in range(1, 8):
            before = self.logits(thread, embeddings, m)
            perturbed_emb = embeddings.clone(); perturbed_emb[m:] = torch.randn(8 - m, 16) * 10
            perturbed = toy_thread()
            perturbed.rel_time[m:] = perturbed.rel_time[m - 1] + 1  # future posts arrive at a different time
            perturbed.parent[m:] = 0                                # and attach elsewhere
            after = self.logits(perturbed, perturbed_emb, m)
            self.assertTrue(torch.allclose(before, after, atol=1e-5), f"prefix {m} leaked future rows")

    def test_late_arriving_parent_is_not_used(self):
        thread = toy_thread(4)
        thread.parent = np.array([-1, 3, 0, 0])  # node 1 replies to a post that only appears later
        cache = ThreadCache([thread])
        self.assertEqual(cache.valid_parent[0][1], -1)
        self.assertEqual(cache.depth[0][1], 0)


if __name__ == "__main__":
    unittest.main()
