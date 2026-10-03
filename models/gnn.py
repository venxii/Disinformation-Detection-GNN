"""Small configurable graph-level GCN/GAT classifiers using PyG layers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_add_pool, global_max_pool, global_mean_pool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class ModelConfig:
    model_type: str = "gcn"
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    num_classes: int = 4
    dropout: float = 0.2
    pooling: str = "mean"
    feature_normalization: str = "none"


class GraphClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.model_type not in {"gcn", "gat"}:
            raise ValueError("model_type must be 'gcn' or 'gat'")
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if config.pooling not in {"mean", "max", "add"}:
            raise ValueError("pooling must be mean, max, or add")
        self.config = config
        layers = []
        in_dim = config.input_dim
        for _ in range(config.num_layers):
            layer = GCNConv(in_dim, config.hidden_dim) if config.model_type == "gcn" else GATConv(in_dim, config.hidden_dim, heads=1, concat=False)
            layers.append(layer)
            in_dim = config.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.num_classes))

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "max":
            return global_max_pool(x, batch)
        if self.config.pooling == "add":
            return global_add_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index))
            x = torch.dropout(x, self.config.dropout, self.training)
        logits = self.classifier(self._pool(x, batch))
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities, "prediction": probabilities.argmax(dim=-1)}


def save_checkpoint(path: str | Path, model: GraphClassifier, optimizer, epoch: int, metrics: dict) -> None:
    payload = {"config": asdict(model.config), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None, "epoch": epoch, "metrics": metrics}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = GraphClassifier(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"])
    return model, payload


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output["logits"], data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    records = []
    for data in loader:
        output = model(data)
        for i in range(output["logits"].size(0)):
            records.append({"event_id": data.event[i] if isinstance(data.event, list) else data.event, "thread_id": data.root_id[i] if isinstance(data.root_id, list) else data.root_id, "num_nodes": int(data.ptr[i + 1] - data.ptr[i]) if hasattr(data, "ptr") else int(data.num_nodes), "prediction": int(output["prediction"][i]), "probability": output["probabilities"][i].cpu().tolist(), "logits": output["logits"][i].cpu().tolist()})
    return records
