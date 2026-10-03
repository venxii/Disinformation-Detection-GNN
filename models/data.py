"""Adapters from Person 1 graph JSON exports to PyG graph objects.

The adapter deliberately computes snapshot-local features only. It does not
read the full-thread temporal aggregate when making a partial graph.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch
from torch_geometric.data import Data

LABELS = {"false": 0, "true": 1, "unverified": 2, "non-rumour": 3}


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%a %b %d %H:%M:%S %z %Y")
    except (TypeError, ValueError):
        return None


def _features(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Return [is_source, relative_seconds, depth, in_degree, out_degree]."""
    ids = [str(node["post_id"]) for node in nodes]
    index = {node_id: i for i, node_id in enumerate(ids)}
    parents = {str(edge["target"]): str(edge["source"]) for edge in edges}
    children: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        if source in children and target in index:
            children[source].append(target)
    times = [_timestamp(node.get("timestamp")) for node in nodes]
    valid = [time for time in times if time is not None]
    origin = min(valid) if valid else None
    values = []
    for node, node_id, time in zip(nodes, ids, times):
        depth = 0
        parent = parents.get(node_id)
        seen = set()
        while parent in index and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = parents.get(parent)
        relative = (time - origin).total_seconds() if time and origin else 0.0
        values.append([
            float(node.get("post_type") == "source"),
            float(max(relative, 0.0)),
            float(depth),
            float(sum(1 for edge in edges if str(edge["target"]) == node_id)),
            float(len(children.get(node_id, []))),
        ])
    return torch.tensor(values, dtype=torch.float32)


def graph_from_json(
    graph: Mapping[str, Any],
    *,
    node_features: torch.Tensor | None = None,
    node_embeddings: Mapping[str, Sequence[float]] | torch.Tensor | None = None,
    concatenate_embeddings: bool = True,
) -> Data:
    """Convert an exported graph into a graph-level PyG ``Data`` object."""
    nodes = list(graph.get("nodes", []))
    node_ids = [str(node["post_id"]) for node in nodes]
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = []
    for edge in graph.get("edges", []):
        source, target = str(edge["source"]), str(edge["target"])
        if source in index and target in index:
            edges.append((index[source], index[target]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    structural = node_features if node_features is not None else _features(nodes, graph.get("edges", []))
    if node_embeddings is None:
        x = structural
    else:
        if isinstance(node_embeddings, Mapping):
            missing = [node_id for node_id in node_ids if node_id not in node_embeddings]
            if missing:
                raise KeyError(f"Missing embeddings for {len(missing)} node IDs; first missing ID: {missing[0]}")
            embedding_tensor = torch.tensor([node_embeddings[node_id] for node_id in node_ids], dtype=torch.float32)
        else:
            embedding_tensor = torch.as_tensor(node_embeddings, dtype=torch.float32)
            if embedding_tensor.ndim != 2 or embedding_tensor.size(0) != len(node_ids):
                raise ValueError("node_embeddings tensor must have one row per node in node_id order")
        if embedding_tensor.ndim != 2 or embedding_tensor.size(0) != structural.size(0):
            raise ValueError("node embeddings must align one-to-one with node IDs")
        x = torch.cat([embedding_tensor, structural], dim=1) if concatenate_embeddings else embedding_tensor
    if x.ndim != 2 or x.size(0) != len(nodes):
        raise ValueError("node_features must have one row per exported node")
    label = str(graph.get("label", ""))
    if label not in LABELS:
        raise ValueError(f"Unsupported label {label!r}; expected one of {sorted(LABELS)}")
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([LABELS[label]], dtype=torch.long),
        node_id=node_ids,
        root_id=str(graph.get("root_id", "")),
        event=str(graph.get("event", "")),
        label=label,
        num_nodes=len(nodes),
        observation_time=max((_timestamp(n.get("timestamp")) for n in nodes), default=None),
        structural_dim=structural.size(1),
        embedding_dim=0 if node_embeddings is None else embedding_tensor.size(1),
    )


def snapshot_graph(graph: Mapping[str, Any], *, fraction: float | None = None, cutoff: datetime | None = None) -> dict[str, Any]:
    """Create a chronologically induced snapshot without future nodes/edges."""
    if (fraction is None) == (cutoff is None):
        raise ValueError("provide exactly one of fraction or cutoff")
    def sort_key(node):
        time = _timestamp(node.get("timestamp"))
        return (time.timestamp() if time is not None else float("-inf"), str(node.get("post_id")))

    nodes = sorted(graph.get("nodes", []), key=sort_key)
    if cutoff is not None:
        selected = [node for node in nodes if (_timestamp(node.get("timestamp")) is not None and _timestamp(node.get("timestamp")) <= cutoff)]
    else:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        count = max(1, int(len(nodes) * fraction + 0.999999))
        selected = nodes[:count]
    ids = {str(node["post_id"]) for node in selected}
    edges = [edge for edge in graph.get("edges", []) if str(edge["source"]) in ids and str(edge["target"]) in ids]
    result = copy.deepcopy(dict(graph))
    result["nodes"], result["edges"] = selected, edges
    result["num_nodes"], result["num_edges"] = len(selected), len(edges)
    return result
