"""Causal prediction trajectories for count or time snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch_geometric.loader import DataLoader

from models.data import graph_from_json, snapshot_graph

COUNT_CHECKPOINTS = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)


def trajectory_snapshots(graph: Mapping[str, Any], checkpoints: Iterable[float] = COUNT_CHECKPOINTS):
    """Yield chronologically valid node-count snapshots."""
    return [("node_fraction", float(value), snapshot_graph(graph, fraction=float(value))) for value in checkpoints]


def _observation_time(snapshot):
    values = [node.get("timestamp") for node in snapshot.get("nodes", []) if node.get("timestamp")]
    return max(values) if values else None


@torch.no_grad()
def build_trajectory(model, graph: Mapping[str, Any], *, checkpoints=COUNT_CHECKPOINTS, calibrator=None, device="cpu", contradiction_by_checkpoint=None, embedding_store=None):
    """Run a model over one graph's causal trajectory and return JSON rows."""
    model = model.to(device).eval()
    rows = []
    previous_prediction = None
    previous_probability = None
    for checkpoint_index, (snapshot_type, snapshot_value, snapshot) in enumerate(trajectory_snapshots(graph, checkpoints)):
        node_embeddings = None
        if embedding_store is not None:
            node_embeddings = embedding_store.lookup(
                [node["post_id"] for node in snapshot.get("nodes", [])]
            )
        data = graph_from_json(snapshot, node_embeddings=node_embeddings).to(device)
        output = model(data)
        raw_logits = output["logits"][0].detach().cpu()
        calibrated_logits = calibrator(raw_logits.unsqueeze(0))[0].detach().cpu() if calibrator else raw_logits
        probabilities = torch.softmax(calibrated_logits, dim=-1)
        prediction = int(probabilities.argmax())
        probability = probabilities.tolist()
        max_probability = float(max(probability))
        row = {
            "event_id": str(graph.get("event", "")),
            "thread_id": str(graph.get("root_id", "")),
            "snapshot_type": snapshot_type,
            "snapshot_value": snapshot_value,
            "snapshot_index": checkpoint_index,
            "observation_time": _observation_time(snapshot),
            "num_nodes": len(snapshot.get("nodes", [])),
            "prediction": prediction,
            "raw_probability": torch.softmax(raw_logits, dim=-1).tolist(),
            "probability": probability,
            "raw_logits": raw_logits.tolist(),
            "logits": calibrated_logits.tolist(),
            "previous_prediction": previous_prediction,
            "prediction_changed": previous_prediction is not None and prediction != previous_prediction,
            "probability_change": None if previous_probability is None else max_probability - previous_probability,
            "prediction_stability": 1.0 if previous_prediction is None or prediction == previous_prediction else 0.0,
            "contradiction_score": float((contradiction_by_checkpoint or {}).get(checkpoint_index, 0.0)),
            "label": str(graph.get("label", "")),
        }
        rows.append(row)
        previous_prediction, previous_probability = prediction, max_probability
    return rows


def write_trajectories(rows, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
