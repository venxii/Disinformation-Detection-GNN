# GNN model interface

Person 1's exported graph JSON contains `root_id`, `label`, `event`, `thread_type`, `nodes`, `edges`, and full-thread `temporal_features`. Each node contains `post_id`, `text`, `timestamp`, `parent_id`, and `post_type`; each edge contains `source` and `target`. The model adapter uses `post_id` as the stable node key, creates `edge_index` only from valid node IDs, and maps labels explicitly as `false=0`, `true=1`, `unverified=2`, `non-rumour=3`.

`models.data.snapshot_graph` must be called before `graph_from_json` for early evaluation. It chronologically selects nodes, induces edges among selected nodes, and leaves full-thread aggregates unused. Its output accepts either a node fraction or an absolute timestamp cutoff, never both.

`GraphClassifier` supports `gcn` and `gat`, configurable layer count/width/dropout/pooling, one-node and zero-edge graphs, and PyG batching. Forward output is a dictionary containing `logits`, `probabilities`, and `prediction`. Trajectory metadata remains on the `Data` object (`event`, `root_id`, `num_nodes`, and `observation_time`) for Person 4's evaluator.

The current implementation uses five snapshot-local placeholder features: source flag, relative timestamp, induced-tree depth, in-degree, and out-degree. `graph_from_json` now optionally accepts `node_embeddings` as either `{post_id: vector}` or a tensor already ordered exactly like `node_id`; with concatenation enabled, the model receives `[embedding, structural]` without GNN changes. Missing or misaligned IDs raise an error.

`models.features.FeatureNormalizer` can standardize concatenated features, but its statistics must be fitted on training-fold graphs only and then frozen for validation/test snapshots. The LOEO training path uses class-weighted cross entropy from training-fold label counts; it does not duplicate graphs or use held-out events.

## Commands

```bash
python -m unittest discover -s tests -v
python -m models.train --model gcn --epochs 2 --graphs data/graphs
python -m models.train --model gat --epochs 2 --graphs data/graphs
python -m decision.evaluate --checkpoint artifacts/gnn_smoke.pt --graph data/graphs/graph_000001.json
```

The decision layer consumes GNN outputs and remains independent of the model architecture. Contradiction scores must come from an external stance/semantic component; a changed prediction is recorded as instability, never as contradiction.
