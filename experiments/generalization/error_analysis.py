"""Step 6: per-event results and three example decision trajectories (clean protocol, system d).

    .venv/bin/python -m experiments.generalization.error_analysis --root artifacts/generalization/rq/<name>/seed7

Examples are picked by a fixed rule, not by eye: among held-out test threads of the full system (d),
  * correct early commit: first commit at checkpoint <= 1 (20%), committed label == gold, no reopen,
    a rumour class and an 8–60-post thread preferred; then highest committed probability;
  * wrong early commit: same, committed label != gold;
  * reopen: a thread with a reopen event, preferring one whose final label is correct.
Writes <root>/4class/error_analysis.md.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lifecycle.data import CHECKPOINTS, LABELS_4, load_threads, task_threads

NAMES = {v: k for k, v in LABELS_4.items()}
SYSTEM = "d_full"


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_traj(root, event, system=SYSTEM):
    with gzip.open(root / "4class" / "variants" / "clean" / "trajectories" / f"{event}_{system}.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def first_commit(log):
    for row in log:
        for e in row["events"]:
            if e["type"] == "commit":
                return row["checkpoint"], e["label"], max(row["signal_probs"])
    return None, None, None


def per_event(root, model="dgtr"):
    vdir = root / "4class" / "verifiers"
    rows = []
    for j in sorted(vdir.glob("*-threads.json")):
        meta = json.loads(j.read_text()); f = np.load(j.with_suffix(".npz"))
        gold = f["test_labels"]; p = softmax(f[f"{model}_test"] / meta["temperatures"][model]).argmax(-1)
        train_majority = json.loads((root / "4class" / "variants" / "clean" / "folds" / f"{j.stem}.json").read_text()).get("majority_class")
        traj = load_traj(root, j.stem)
        final = np.array([r["log"][-1]["label"] if r["log"][-1]["label"] is not None else -1 for r in traj])
        g = np.array([r["gold"] for r in traj])
        commits = [first_commit(r["log"])[0] for r in traj]
        rows.append({
            "event": j.stem.replace("-all-rnr-threads", ""), "n": len(gold),
            "class_counts": {NAMES[c]: int((gold == c).sum()) for c in range(4)},
            "acc_10": float((p[:, 0] == gold).mean()), "acc_100": float((p[:, -1] == gold).mean()),
            "mf1_10": float(f1_score(gold, p[:, 0], labels=range(4), average="macro", zero_division=0)),
            "mf1_100": float(f1_score(gold, p[:, -1], labels=range(4), average="macro", zero_division=0)),
            "nonrumour_share": float((gold == 3).mean()),
            "d_final_coverage": float((final >= 0).mean()), "d_final_acc_committed": float((final[final >= 0] == g[final >= 0]).mean()) if (final >= 0).any() else None,
            "d_median_commit_checkpoint": float(np.median([c for c in commits if c is not None])) if any(c is not None for c in commits) else None,
            "d_reopens": int(sum(e["type"] == "reopen" for r in traj for row in r["log"] for e in row["events"])),
            "majority_class_train": train_majority,
        })
    return rows


def pick_examples(root, threads_by_root):
    cands = {"correct_early": [], "wrong_early": [], "reopen": []}
    for j in sorted((root / "4class" / "verifiers").glob("*-threads.json")):
        for r in load_traj(root, j.stem):
            k, lab, conf = first_commit(r["log"])
            reopened = any(e["type"] == "reopen" for row in r["log"] for e in row["events"])
            if reopened:
                cands["reopen"].append((r["log"][-1]["label"] == r["gold"], r["gold"] != 3, threads_by_root[r["root_id"]].n, r))
            elif k is not None and k <= 1:
                key = "correct_early" if lab == r["gold"] else "wrong_early"
                cands[key].append((r["gold"] != 3, 8 <= threads_by_root[r["root_id"]].n <= 60, conf, r))
    out = {}
    for key in ("correct_early", "wrong_early"):
        if cands[key]:
            out[key] = sorted(cands[key], key=lambda x: x[:3], reverse=True)[0][-1]
    if cands["reopen"]:
        # prefer corrected, then rumour gold, then a mid-sized thread (readable)
        out["reopen"] = sorted(cands["reopen"], key=lambda x: (x[0], x[1], -abs(x[2] - 30)), reverse=True)[0][-1]
    return out


def render(r, t, title):
    lines = [f"### {title}", "",
             f"`{r['event'].replace('-all-rnr-threads', '')}` · root {r['root_id']} · gold **{NAMES[r['gold']]}** · {t.n} posts", "",
             f"> **source:** {t.texts[0][:280]}", ""]
    shown = 0
    for i in range(1, min(t.n, 40)):
        if shown == 4:
            break
        lines.append(f"> reply +{t.rel_time[i] / 60:.0f} min: {t.texts[i][:200]}"); shown += 1
    lines += ["", "| ckpt | posts | min | state | label | p(false/true/unver/non-rum) | conflict (strength / volume / net) | events |", "|---|---|---|---|---|---|---|---|"]
    for row in r["log"]:
        c = row.get("conflict") or {}
        conf = f"{c.get('strength', 0):.2f} / {c.get('volume', 0)} / {c.get('net', 0):.2f}" if c else "–"
        ev = ", ".join(f"{e['type']}" + (f"→{NAMES.get(e.get('label'), e.get('label'))}" if e.get("label") is not None else "") for e in row["events"]) or ""
        lines.append(f"| {int(CHECKPOINTS[row['checkpoint']] * 100)}% | {row['num_posts']} | {row['elapsed_min']:.0f} | {row['state']} | "
                     f"{NAMES.get(row['label'], '–') if row['label'] is not None else '–'} | {' / '.join(f'{x:.2f}' for x in row['signal_probs'])} | {conf} | {ev} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    threads = {t.root_id: t for t in task_threads(load_threads()[0], "4class")}
    rows = per_event(root)
    md = [f"# Error analysis — {root}", "", "## Per held-out event (4-class, DGTR calibrated; lifecycle = clean system d)", "",
          "| event | n | non-rumour share | acc @10% | acc @100% | mF1 @10% | mF1 @100% | d: final coverage | d: acc of final commits | d: median first-commit ckpt | d: reopens |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mc = r["d_median_commit_checkpoint"]
        md.append(f"| {r['event']} | {r['n']} | {r['nonrumour_share']:.2f} | {r['acc_10']:.3f} | {r['acc_100']:.3f} | {r['mf1_10']:.3f} | {r['mf1_100']:.3f} | "
                  f"{r['d_final_coverage']:.2f} | {r['d_final_acc_committed'] if r['d_final_acc_committed'] is None else round(r['d_final_acc_committed'], 3)} | "
                  f"{'–' if mc is None else f'{int(CHECKPOINTS[int(mc)] * 100)}%'} | {r['d_reopens']} |")
    md += ["", "## Example trajectories (system d, picked by rule; see module docstring)", ""]
    titles = {"correct_early": "1. Correct early commit", "wrong_early": "2. Wrong early commit", "reopen": "3. A reopen"}
    for key, r in pick_examples(root, threads).items():
        md.append(render(r, threads[r["root_id"]], titles[key]))
    (root / "4class" / "error_analysis.md").write_text("\n".join(md))
    (root / "4class" / "per_event.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
