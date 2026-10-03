import unittest

try:
    import torch
    from torch_geometric.data import Batch
    from models.data import graph_from_json, snapshot_graph
    from models.gnn import GraphClassifier, ModelConfig, set_seed
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch/PyG are not installed")
class GNNTests(unittest.TestCase):
    def graph(self, n=1, edges=None):
        edges = edges or []
        nodes = [{"post_id": str(i), "post_type": "source" if i == 0 else "reaction", "timestamp": f"Wed Jan 07 11:{i:02d}:00 +0000 2015"} for i in range(n)]
        return {"root_id": "0", "event": "event", "label": "false", "nodes": nodes, "edges":[{"source":str(a),"target":str(b)} for a,b in edges]}

    def test_graph_shapes_and_edge_cases(self):
        for graph in [self.graph(), self.graph(2, [(0, 1)]), self.graph(4, [(0, 1), (0, 2), (2, 3)])]:
            data = graph_from_json(graph)
            self.assertEqual(data.x.shape, (len(graph["nodes"]), 5))
            model = GraphClassifier(ModelConfig(input_dim=5, hidden_dim=8))
            output = model(data)
            self.assertEqual(output["logits"].shape, (1, 4))

    def test_batch_different_sizes(self):
        batch = Batch.from_data_list([graph_from_json(self.graph()), graph_from_json(self.graph(3, [(0, 1), (1, 2)]))])
        output = GraphClassifier(ModelConfig(input_dim=5, hidden_dim=8))(batch)
        self.assertEqual(output["logits"].shape, (2, 4))

    def test_snapshot_is_causal_and_deterministic(self):
        graph = self.graph(4, [(0, 1), (1, 2), (2, 3)])
        snap = snapshot_graph(graph, fraction=0.5)
        self.assertEqual([n["post_id"] for n in snap["nodes"]], ["0", "1"])
        self.assertEqual(snap["edges"], [{"source":"0", "target":"1"}])
        set_seed(11); first = GraphClassifier(ModelConfig(input_dim=5, hidden_dim=8))(graph_from_json(graph))["logits"]
        set_seed(11); second = GraphClassifier(ModelConfig(input_dim=5, hidden_dim=8))(graph_from_json(graph))["logits"]
        self.assertTrue(torch.equal(first, second))

    def test_id_aligned_embedding_concatenation(self):
        graph = self.graph(2, [(0, 1)])
        embeddings = {"0": [1.0, 2.0], "1": [3.0, 4.0]}
        data = graph_from_json(graph, node_embeddings=embeddings)
        self.assertEqual(data.x.shape, (2, 7))
        self.assertTrue(torch.equal(data.x[:, :2], torch.tensor([[1., 2.], [3., 4.]])))

    def test_missing_embedding_id_is_rejected(self):
        with self.assertRaises(KeyError):
            graph_from_json(self.graph(2), node_embeddings={"0": [1.0, 2.0]})

    def test_embedding_store_preserves_id_lookup(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from models.embedding_store import load_embedding_store
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "embeddings.npy", np.asarray([[1., 2.], [3., 4.]], dtype=np.float32))
            (path / "ids.csv").write_text("embedding_index,post_id\n0,20\n1,10\n", encoding="utf-8")
            store = load_embedding_store(path)
            self.assertTrue(np.array_equal(store.lookup(["10", "20"]), [[3., 4.], [1., 2.]]))


if __name__ == "__main__":
    unittest.main()
