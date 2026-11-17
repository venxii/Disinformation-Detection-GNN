"""Does held-out accuracy rise across snapshots? Pooled LOEO accuracy / macro-F1 at each checkpoint,
mean ± std over seeds, and the per-seed slope (100% − 10%), for every configuration root.

    .venv/bin/python -m experiments.generalization.snapshot_curves --roots baseline,masked
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lifecycle.data import CHECKPOINTS, TASKS

from .compare import load

BASE = Path("artifacts/generalization")


def curves(root, task, model):
    C = len(set(TASKS[task].values()))
    out = {}
    for seed, folds in load(root, task, model).items():
        g = np.concatenate([v[0] for v in folds.values()]); p = np.concatenate([v[1] for v in folds.values()])
        out[seed] = {"acc": [float((p[:, k] == g).mean()) for k in range(len(CHECKPOINTS))],
                     "mf1": [float(f1_score(g, p[:, k], labels=range(C), average="macro", zero_division=0)) for k in range(len(CHECKPOINTS))]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", default="baseline,masked")
    parser.add_argument("--models", default="dgtr,gat_full,prefix_lr")
    args = parser.parse_args()
    res = {}
    print("| task | config | model | acc 10% → 100% (mean over seeds) | Δacc 100−10 (mean ± std) | seeds Δacc>0 | Δ macro-F1 100−10 (mean ± std) |")
    print("|---|---|---|---|---|---|---|")
    for task in ("4class", "binary"):
        for r in args.roots.split(","):
            for m in args.models.split(","):
                c = curves(BASE / r, task, m)
                if not c:
                    continue
                res[f"{task}/{r}/{m}"] = c
                acc = np.array([v["acc"] for v in c.values()]); mf1 = np.array([v["mf1"] for v in c.values()])
                da, dm = acc[:, -1] - acc[:, 0], mf1[:, -1] - mf1[:, 0]
                print(f"| {task} | {r} | {m} | {' / '.join(f'{x:.3f}' for x in acc.mean(0))} | {da.mean():+.4f} ± {da.std(ddof=1):.4f} | {(da > 0).sum()}/{len(da)} | {dm.mean():+.4f} ± {dm.std(ddof=1):.4f} |")
    (BASE / "snapshot_curves.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
