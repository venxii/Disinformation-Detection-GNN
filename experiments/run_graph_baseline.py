"""Run the first reproducible event-held-out graph-only PHEME baseline."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from decision.calibration import TemperatureScaler
from decision.controller import ControllerConfig, DecisionController, DecisionState
from decision.metrics import brier_score, classification_metrics, expected_calibration_error
from decision.trajectory import build_trajectory
from models.data import LABELS, graph_from_json
from models.gnn import GraphClassifier, ModelConfig, save_checkpoint, set_seed


def read_graphs(graph_dir):
    items = []
    for path in sorted(Path(graph_dir).glob("graph_*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        if graph.get("label") in LABELS and graph.get("label") != "unknown":
            items.append((path, graph))
    return items


def make_split(items, seed):
    events = sorted({str(graph["event"]) for _, graph in items})
    rng = random.Random(seed)
    rng.shuffle(events)
    n_test = max(1, round(len(events) * 0.20))
    n_val = max(1, round(len(events) * 0.20))
    test_events = sorted(events[:n_test])
    val_events = sorted(events[n_test:n_test + n_val])
    train_events = sorted(events[n_test + n_val:])
    event_to_split = {event: "test" for event in test_events} | {event: "validation" for event in val_events} | {event: "train" for event in train_events}
    split = {"seed": seed, "train_events": train_events, "validation_events": val_events, "test_events": test_events, "threads": {str(graph["root_id"]): event_to_split[str(graph["event"])] for _, graph in items}}
    return split


def datasets(items, split):
    result = {name: [] for name in ("train", "validation", "test")}
    for path, graph in items:
        result[split["threads"][str(graph["root_id"])]].append((path, graph))
    return result


def loader(records, batch_size, shuffle=False):
    return DataLoader([graph_from_json(graph) for _, graph in records], batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def collect_logits(model, data_loader):
    model.eval()
    logits, labels = [], []
    for batch in data_loader:
        output = model(batch)
        logits.append(output["logits"].cpu())
        labels.append(batch.y.view(-1).cpu())
    return torch.cat(logits), torch.cat(labels)


def train(model, data_loader, epochs, lr, weight_decay, class_weights=None):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in data_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = nn.functional.cross_entropy(output["logits"], batch.y.view(-1), weight=class_weights)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * batch.num_graphs
        history.append({"epoch": epoch, "loss": loss_total / max(len(data_loader.dataset), 1)})
    return history


def evaluate_snapshot(model, records, calibrator, checkpoint):
    rows = []
    for _, graph in records:
        trajectory = build_trajectory(model, graph, checkpoints=[checkpoint], calibrator=calibrator)
        rows.extend(trajectory)
    return rows


def summary(rows):
    return {"classification": classification_metrics(rows), "calibrated_brier": brier_score(rows), "calibrated_ece": expected_calibration_error(rows), "raw_brier": brier_score([{**row, "probability": row["raw_probability"]} for row in rows]), "raw_ece": expected_calibration_error(rows, probability_key="raw_probability")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="data/graphs")
    parser.add_argument("--output", default="artifacts/graph_baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    items = read_graphs(args.graphs)
    split = make_split(items, args.seed)
    groups = datasets(items, split)
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    config = {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "layers": args.layers, "dropout": args.dropout, "learning_rate": args.lr, "weight_decay": args.weight_decay, "feature_set": "snapshot-local [source, relative time, induced depth, in-degree, out-degree]", "train_graphs": len(groups["train"]), "validation_graphs": len(groups["validation"]), "test_graphs": len(groups["test"]), "train_events": split["train_events"], "validation_events": split["validation_events"], "test_events": split["test_events"]}
    results = {"config": config, "models": {}, "class_distribution": {name: dict(Counter(graph["label"] for _, graph in records)) for name, records in groups.items()}}
    for model_type in ("gcn", "gat"):
        set_seed(args.seed)
        model = GraphClassifier(ModelConfig(model_type=model_type, input_dim=5, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout))
        history = train(model, loader(groups["train"], args.batch_size, True), args.epochs, args.lr, args.weight_decay)
        val_logits, val_labels = collect_logits(model, loader(groups["validation"], args.batch_size))
        calibrator = TemperatureScaler().fit(val_logits, val_labels)
        calibrator.save(out / f"{model_type}_temperature.pt")
        save_checkpoint(out / f"{model_type}.pt", model, None, args.epochs, {"history": history, "temperature": float(calibrator.temperature.detach())})
        model_results = {"history": history, "temperature": float(calibrator.temperature.detach()), "snapshots": {}}
        all_rows = []
        for checkpoint in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            rows = evaluate_snapshot(model, groups["test"], calibrator, checkpoint)
            model_results["snapshots"][str(checkpoint)] = summary(rows)
            all_rows.extend(rows)
        trajectories = []
        for _, graph in groups["test"]:
            rows = build_trajectory(model, graph, calibrator=calibrator)
            state = DecisionState()
            controller = DecisionController(ControllerConfig())
            stable_count = 0
            for row in rows:
                stable_count = stable_count + 1 if not row["prediction_changed"] else 1
                controller.update(state, prediction=row["prediction"], confidence=max(row["probability"]), num_nodes=row["num_nodes"], stable_count=stable_count)
                row["decision_state"], row["decision_label"] = state.status, state.label
            trajectories.append(rows)
        (out / f"{model_type}_trajectories.json").write_text(json.dumps(trajectories), encoding="utf-8")
        model_results["trajectory_count"] = len(trajectories)
        results["models"][model_type] = model_results
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
