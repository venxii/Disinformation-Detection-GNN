"""Chronologically ordered thread tensors shared by every lifecycle component.

Each thread is stored once with its nodes sorted by (timestamp, post_id), the
same order ``models.data.snapshot_graph`` uses. A checkpoint is then just a
prefix length, so "no future posts" reduces to "only rows < n_k are read".
Full-thread aggregates from the graph JSON (``temporal_features``) are never
loaded.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

CHECKPOINTS = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
LABELS_4 = {"false": 0, "true": 1, "unverified": 2, "non-rumour": 3}
LABELS_VERACITY = {"false": 0, "true": 1, "unverified": 2}
TASKS = {"4class": LABELS_4, "veracity": LABELS_VERACITY}
STANCES = ("support", "deny", "query", "comment")


def parse_time(value) -> float | None:
    try:
        return datetime.strptime(str(value), "%a %b %d %H:%M:%S %z %Y").timestamp()
    except (TypeError, ValueError):
        return None


def prefix_length(num_nodes: int, fraction: float) -> int:
    """Identical rounding to ``models.data.snapshot_graph``."""
    return max(1, int(num_nodes * fraction + 0.999999))


@dataclass
class Thread:
    root_id: str
    event: str
    label: str
    post_ids: list[str]          # chronological
    texts: list[str]
    rel_time: np.ndarray         # seconds since first post, chronological, >= 0
    parent: np.ndarray           # index of parent in chronological order, -1 if none/unavailable
    is_source: np.ndarray        # bool
    emb_rows: np.ndarray         # rows into the global embedding matrix

    @property
    def n(self) -> int:
        return len(self.post_ids)

    def checkpoint_lengths(self, checkpoints=CHECKPOINTS) -> list[int]:
        return [prefix_length(self.n, f) for f in checkpoints]


def load_threads(graph_dir="data/graphs", embedding_dir="data/nlp/embeddings/full"):
    """Return (threads, embeddings) with embeddings memory-mapped float32 [N, 768]."""
    embeddings = np.load(Path(embedding_dir) / "embeddings.npy", mmap_mode="r")
    with open(Path(embedding_dir) / "ids.csv", newline="", encoding="utf-8") as file:
        index = {row["post_id"]: int(row["embedding_index"]) for row in csv.DictReader(file)}
    threads = []
    for path in sorted(Path(graph_dir).glob("graph_*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        if graph.get("label") not in LABELS_4:
            continue  # drops the single 'unknown' thread
        nodes = sorted(graph["nodes"], key=lambda n: (parse_time(n.get("timestamp")) or float("-inf"), str(n["post_id"])))
        ids = [str(n["post_id"]) for n in nodes]
        pos = {pid: i for i, pid in enumerate(ids)}
        times = np.array([parse_time(n.get("timestamp")) or 0.0 for n in nodes], dtype=np.float64)
        rel = np.maximum(times - times.min(), 0.0) if len(times) else times
        parents = {str(e["target"]): str(e["source"]) for e in graph.get("edges", [])}
        parent = np.array([pos.get(parents.get(pid, ""), -1) for pid in ids], dtype=np.int64)
        threads.append(Thread(
            root_id=str(graph["root_id"]), event=str(graph["event"]), label=str(graph["label"]),
            post_ids=ids, texts=[str(n.get("text", "")) for n in nodes], rel_time=rel.astype(np.float32),
            parent=parent, is_source=np.array([n.get("post_type") == "source" for n in nodes]),
            emb_rows=np.array([index[pid] for pid in ids], dtype=np.int64),
        ))
    return threads, embeddings


def task_threads(threads, task: str):
    labels = TASKS[task]
    return [t for t in threads if t.label in labels]


def rich_events(threads, min_threads=200, min_classes=3, min_per_class=30):
    """Events usable for validation: enough threads and >= min_classes classes with >= min_per_class each."""
    from collections import Counter
    by_event: dict[str, Counter] = {}
    for t in threads:
        by_event.setdefault(t.event, Counter())[t.label] += 1
    return sorted(e for e, c in by_event.items()
                  if sum(c.values()) >= min_threads and sum(v >= min_per_class for v in c.values()) >= min_classes)


def loeo_folds(threads):
    """Leave-one-event-out folds. Inner validation = next *rich* event in the sorted cycle.

    experiments/run_loeo.py used the plain next event, which makes e.g. ebola-essien
    (14 threads, one class) the validation set for the charliehebdo fold; calibration
    and threshold tuning on it are degenerate. The held-out event never influences the choice."""
    events = sorted({t.event for t in threads})
    rich = rich_events(threads)
    for i, held_out in enumerate(events):
        cycle = events[i + 1:] + events[:i]
        validation = next(e for e in cycle if e in rich)
        yield held_out, validation, [e for e in events if e not in (held_out, validation)]
