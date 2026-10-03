"""Leave-one-event-out graph-only evaluation with event-level inner validation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from decision.calibration import TemperatureScaler
from decision.metrics import brier_score, classification_metrics
from decision.trajectory import build_trajectory
from models.data import LABELS, graph_from_json
from models.embedding_store import load_embedding_store
from models.gnn import GraphClassifier, ModelConfig, set_seed
from experiments.run_graph_baseline import collect_logits, read_graphs, train

CHECKPOINTS = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)


def loader(records, batch_size, shuffle=False, embedding_store=None):
    data = []
    for _, graph in records:
        node_embeddings = None
        if embedding_store is not None:
            node_embeddings = embedding_store.lookup([node["post_id"] for node in graph.get("nodes", [])])
        data.append(graph_from_json(graph, node_embeddings=node_embeddings))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def fold_split(items, held_out):
    events = sorted({str(graph["event"]) for _, graph in items})
    remaining = [event for event in events if event != held_out]
    # Deterministic, test-independent inner validation: the next event in the
    # sorted cycle. The held-out event is never consulted for this choice.
    validation_event = remaining[(events.index(held_out)) % len(remaining)]
    split = {"test_event": held_out, "validation_event": validation_event, "train_events": [event for event in remaining if event != validation_event]}
    groups = {"train": [], "validation": [], "test": []}
    for path, graph in items:
        event = str(graph["event"])
        group = "test" if event == held_out else "validation" if event == validation_event else "train"
        groups[group].append((path, graph))
    return split, groups


def metrics_for_rows(rows):
    return {"classification": classification_metrics(rows), "brier": brier_score(rows)}


def run_model(groups, model_type, args, output):
    set_seed(args.seed)
    embedding_store = getattr(args, "embedding_store", None)
    input_dim = 5 if embedding_store is None else 5 + embedding_store.dimension
    model = GraphClassifier(ModelConfig(model_type=model_type, input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=2, dropout=args.dropout))
    class_weights = None
    if args.class_weighting:
        train_labels = [LABELS[graph["label"]] for _, graph in groups["train"]]
        counts = torch.bincount(torch.tensor(train_labels), minlength=len(LABELS)).float()
        class_weights = (counts.sum() / counts.clamp_min(1.0)).div(len(LABELS))
    history = train(model, loader(groups["train"], args.batch_size, True, embedding_store), args.epochs, args.lr, args.weight_decay, class_weights=class_weights)
    val_logits, val_labels = collect_logits(model, loader(groups["validation"], args.batch_size, embedding_store=embedding_store))
    calibrator = TemperatureScaler().fit(val_logits, val_labels)
    rows_by_checkpoint = {}
    trajectories = []
    for _, graph in groups["test"]:
        trajectories.append(build_trajectory(model, graph, calibrator=calibrator, embedding_store=embedding_store))
    for index, checkpoint in enumerate(CHECKPOINTS):
        rows_by_checkpoint[str(checkpoint)] = metrics_for_rows([trajectory[index] for trajectory in trajectories])
    return {"history": history, "temperature": float(calibrator.temperature.detach()), "class_weights": None if class_weights is None else class_weights.tolist(), "metrics": rows_by_checkpoint, "trajectories": trajectories, "test_support": dict(Counter(graph["label"] for _, graph in groups["test"]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="data/graphs")
    parser.add_argument("--output", default="artifacts/loeo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-events", type=int, default=0, help="debug only; 0 runs all nine folds")
    parser.add_argument("--embedding-dir", default=None, help="directory containing embeddings.npy and ids.csv; enables text + graph mode")
    parser.add_argument("--class-weighting", action="store_true", help="opt in to training-fold class-weighted cross entropy")
    args = parser.parse_args()
    args.embedding_store = load_embedding_store(args.embedding_dir) if args.embedding_dir else None
    set_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    items = read_graphs(args.graphs)
    events = sorted({str(graph["event"]) for _, graph in items})
    if args.limit_events:
        events = events[:args.limit_events]
    report_config = {key: value for key, value in vars(args).items() if key != "embedding_store"}
    report = {"config": report_config, "folds": {}}
    for held_out in events:
        split, groups = fold_split(items, held_out)
        fold = {"split": split, "class_distribution": {name: dict(Counter(graph["label"] for _, graph in records)) for name, records in groups.items()}, "models": {}}
        for model_type in ("gcn", "gat"):
            result = run_model(groups, model_type, args, output)
            trajectory_path = output / f"{held_out}_{model_type}_trajectories.json"
            trajectory_path.write_text(json.dumps(result.pop("trajectories")), encoding="utf-8")
            fold["models"][model_type] = {**result, "trajectory_file": str(trajectory_path)}
        report["folds"][held_out] = fold
        (output / "partial_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"completed held_out={held_out}")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={output / 'report.json'}")


if __name__ == "__main__":
    main()
