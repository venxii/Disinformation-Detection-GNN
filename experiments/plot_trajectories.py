"""Phase 6 figures: example decision trajectories and coverage/accuracy over checkpoints.

    .venv/bin/python -m experiments.plot_trajectories --task veracity
Writes artifacts/lifecycle/<task>/figures/*.png
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lifecycle.data import CHECKPOINTS, TASKS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # reference categorical slots 1-4
STATE = {"WAIT": "#e4e3df", "PROVISIONAL": "#9ec3ee", "CONFIRMED": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SYSTEMS = {"b_early_no_revision": "(b) early, no revision", "c_revision_no_conflict_gate": "(c) revise on prediction change", "d_full": "(d) conflict-gated (full)"}


def load(out, name):
    rows = {}
    for path in sorted((out / "trajectories").glob(f"*_{name}.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                rows[r["root_id"]] = r
    return rows


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pick_examples(d, c):
    picks = []
    def reopens(r):
        return [(k, e) for k, row in enumerate(r["log"]) for e in row["events"] if e["type"] == "reopen"]
    corrected = [r for r in d.values() if any(e["label"] != r["gold"] for _, e in reopens(r)) and r["log"][-1]["label"] == r["gold"]]
    false_reopen = [r for r in d.values() if any(e["label"] == r["gold"] for _, e in reopens(r))]
    disagree = [r for r in d.values() if r["root_id"] in c and reopens(c[r["root_id"]]) and not reopens(r)]
    for title, pool in (("Justified reopen that corrected the decision", corrected), ("False reopen (committed label was right)", false_reopen),
                        ("(c) reopens on a prediction flip; (d) holds", disagree)):
        if pool:
            pool = sorted(pool, key=lambda r: -len(r["log"][-1].get("events", [])) - r["log"][-1]["num_posts"] / 1000)
            picks.append((title, pool[0]))
    return picks


def plot_example(ax_p, ax_s, title, record, logs, classes):
    x = np.arange(len(CHECKPOINTS))
    probs = np.array([row["verifier_probs"] for row in record["log"]])
    for j, name in enumerate(classes):
        ax_p.plot(x, probs[:, j], color=SERIES[j], linewidth=2, marker="o", markersize=5, label=name + (" (gold)" if j == record["gold"] else ""))
    style(ax_p)
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("calibrated P(label)", color=MUTED, fontsize=9)
    ax_p.set_title(f"{title}\n{record['event'].replace('-all-rnr-threads', '')} · thread {record['root_id']} · gold = {classes[record['gold']]}", fontsize=10, color=INK, loc="left")
    ax_p.legend(frameon=False, fontsize=8, ncol=len(classes), loc="upper left", bbox_to_anchor=(0, -0.02))
    ax_p.set_xticks(x, [""] * len(x))
    for i, (name, label) in enumerate(SYSTEMS.items()):
        r = logs[name][record["root_id"]]
        for k, row in enumerate(r["log"]):
            ax_s.barh(i, 0.96, left=k - 0.48, color=STATE[row["state"]], height=0.7, edgecolor="white", linewidth=2)
            text = "·" if row["label"] is None else classes[row["label"]][:5]
            ax_s.text(k, i, text, ha="center", va="center", fontsize=7.5, color=INK if row["state"] != "CONFIRMED" else "white")
            if any(e["type"] == "reopen" for e in row["events"]):
                ax_s.plot(k - 0.42, i + 0.25, marker="v", color="#e34948", markersize=7)
    ax_s.set_yticks(range(len(SYSTEMS)), list(SYSTEMS.values()), fontsize=8, color=MUTED)
    ax_s.set_xticks(x, [f"{int(c * 100)}%\n{row['num_posts']} posts" for c, row in zip(CHECKPOINTS, record["log"])], fontsize=8, color=MUTED)
    ax_s.invert_yaxis()
    for side in ("top", "right", "left"):
        ax_s.spines[side].set_visible(False)
    ax_s.spines["bottom"].set_color(MUTED)
    ax_s.tick_params(length=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="veracity")
    parser.add_argument("--root", default="artifacts/lifecycle")
    args = parser.parse_args()
    out = Path(args.root) / args.task
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    classes = list(TASKS[args.task])
    logs = {name: load(out, name) for name in SYSTEMS}
    for n, (title, record) in enumerate(pick_examples(logs["d_full"], logs["c_revision_no_conflict_gate"]), start=1):
        fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(8, 5.4), gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.35})
        plot_example(ax_p, ax_s, title, record, logs, classes)
        handles = [matplotlib.patches.Patch(color=STATE[s], label=s) for s in STATE] + [matplotlib.lines.Line2D([], [], marker="v", color="#e34948", linestyle="", label="REOPEN")]
        ax_s.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.38))
        fig.savefig(figs / f"trajectory_example_{n}.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    summary = json.loads((out / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = [int(c * 100) for c in CHECKPOINTS]
    names = {"a_final_only": "(a) final-only", **SYSTEMS}
    for j, (name, label) in enumerate(names.items()):
        pc = summary["systems"][name]["per_checkpoint"]
        axes[0].plot(x, [p["coverage"] for p in pc], color=SERIES[j], linewidth=2, marker="o", markersize=5, label=label)
        axes[1].plot(x, [p["decision_acc"] if p["coverage"] > 0 else np.nan for p in pc], color=SERIES[j], linewidth=2, marker="o", markersize=5, label=label)
    for ax, t in zip(axes, ("Share of threads with a committed decision", "Accuracy of committed decisions")):
        style(ax); ax.set_title(t, fontsize=10, color=INK, loc="left"); ax.set_xlabel("checkpoint (% of thread observed)", fontsize=9, color=MUTED)
        ax.set_xticks(x)
    axes[1].axhline(1 / len(classes), color=MUTED, linestyle="--", linewidth=1)
    axes[1].annotate("chance (1/C)", (x[-1], 1 / len(classes)), xytext=(0, -12), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(figs / "coverage_accuracy.png", dpi=160, bbox_inches="tight", facecolor="white")
    print(f"figures in {figs}")


if __name__ == "__main__":
    main()
