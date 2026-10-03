"""Minimal reproducible training entry point for exported graph JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from .data import graph_from_json
from .gnn import GraphClassifier, ModelConfig, save_checkpoint, set_seed, train_epoch


def load_graphs(paths):
    with_paths = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            with_paths.append(graph_from_json(json.load(file)))
    return with_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="data/graphs")
    parser.add_argument("--model", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="artifacts/gnn_smoke.pt")
    args = parser.parse_args()
    set_seed(args.seed)
    paths = sorted(Path(args.graphs).glob("graph_*.json"))[:64]
    dataset = load_graphs(paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = GraphClassifier(ModelConfig(model_type=args.model, input_dim=dataset[0].num_node_features, hidden_dim=args.hidden_dim))
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    metrics = {}
    for epoch in range(1, args.epochs + 1):
        metrics[f"epoch_{epoch}"] = {"loss": train_epoch(model, loader, optimizer, criterion)}
        print(f"epoch={epoch} loss={metrics[f'epoch_{epoch}']['loss']:.4f}")
    save_checkpoint(args.checkpoint, model, optimizer, args.epochs, metrics)
    print(f"saved={args.checkpoint}")


if __name__ == "__main__":
    main()

