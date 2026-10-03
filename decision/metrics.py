"""Standard trajectory metrics; no new research metric is introduced."""

from __future__ import annotations

import math
from collections import Counter

LABEL_IDS = {"false": 0, "true": 1, "unverified": 2, "non-rumour": 3}


def _label_id(value):
    return LABEL_IDS.get(value, value) if isinstance(value, str) else int(value)


def classification_metrics(rows):
    labels = [_label_id(row["label"]) for row in rows]
    predictions = [_label_id(row["prediction"]) for row in rows]
    accuracy = sum(a == b for a, b in zip(labels, predictions)) / max(len(labels), 1)
    classes = sorted(set(labels) | set(predictions))
    f1 = []
    per_class = {}
    for label in classes:
        tp = sum(p == label and y == label for p, y in zip(predictions, labels))
        fp = sum(p == label and y != label for p, y in zip(predictions, labels))
        fn = sum(p != label and y == label for p, y in zip(predictions, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1[-1], "support": sum(y == label for y in labels)}
    return {"accuracy": accuracy, "macro_f1": sum(f1) / max(len(f1), 1), "support": dict(Counter(map(str, labels))), "per_class": per_class}


def brier_score(rows):
    total = 0.0
    for row in rows:
        total += sum((prob - float(index == _label_id(row["label"]))) ** 2 for index, prob in enumerate(row["probability"]))
    return total / max(len(rows), 1)


def expected_calibration_error(rows, bins=10, probability_key="probability"):
    buckets = [[] for _ in range(bins)]
    for row in rows:
        probabilities = row[probability_key]
        confidence = max(probabilities)
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((confidence, int(row["prediction"]) == _label_id(row["label"])))
    total = max(len(rows), 1)
    return sum(len(bucket) / total * abs(sum(conf for conf, _ in bucket) / len(bucket) - sum(correct for _, correct in bucket) / len(bucket)) for bucket in buckets if bucket)


def selective_metrics(rows):
    decided = [row for row in rows if row.get("decision_state") in {"PROVISIONAL", "CONFIRMED"}]
    correct = sum(int(row["prediction"]) == int(row["label"]) for row in decided)
    return {"coverage": len(decided) / max(len(rows), 1), "selective_risk": 1 - correct / max(len(decided), 1), "unnecessary_wait_rate": sum(row.get("decision_state") == "WAIT" and int(row["prediction"]) == int(row["label"]) for row in rows) / max(len(rows), 1)}
