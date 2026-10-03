"""Small trajectory/controller evaluation CLI and baseline helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.gnn import load_checkpoint
from .controller import DecisionController, DecisionState
from .trajectory import build_trajectory


def apply_controller(rows, controller=None):
    controller = controller or DecisionController()
    state = DecisionState()
    stable_count = 0
    for row in rows:
        stable_count = stable_count + 1 if not row["prediction_changed"] else 1
        confidence = max(row["probability"])
        controller.update(state, prediction=row["prediction"], confidence=confidence, num_nodes=row["num_nodes"], stable_count=stable_count, contradiction_score=row.get("contradiction_score", 0.0), new_evidence=row.get("new_evidence", 0), contradictory_evidence=row.get("contradictory_evidence", 0))
        row["decision_state"] = state.status
        row["decision_label"] = state.label
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", default="artifacts/trajectory.json")
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint)
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    rows = apply_controller(build_trajectory(model, graph))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved={args.output} checkpoints={len(rows)} final_state={rows[-1]['decision_state']}")


if __name__ == "__main__":
    main()
