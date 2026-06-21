import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


UNKNOWN_LABEL = "unknown"


def set_random_seed(seed: Optional[int]):
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _candidate_labels(text: str, label_names: Sequence[str]) -> List[str]:
    out = []
    for label in label_names:
        pattern = rf"(?<![a-z0-9_]){re.escape(label.lower())}(?![a-z0-9_])"
        if re.search(pattern, text):
            out.append(label)
    return out


def parse_label_response(raw_response: str, label_names: Sequence[str]) -> Tuple[str, bool]:
    """Parse a generative VLM response into exactly one known class label."""
    if raw_response is None:
        return UNKNOWN_LABEL, False

    labels = [label.lower() for label in label_names]
    canonical = {label.lower(): label for label in label_names}
    text = str(raw_response).strip()
    lowered = text.lower()
    cleaned = lowered.strip(" \t\r\n`'\".,;:!?()[]{}")
    if cleaned in canonical:
        return canonical[cleaned], True

    final_patterns = [
        r"(?:final\s+answer|answer|label|class|classification)\s*(?:is|=|:|-)?\s*([a-z_]+)",
        r"(?:I\s+choose|I\s+select|classified\s+as)\s+([a-z_]+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, lowered)
        for match in reversed(matches):
            match = match.strip(" \t\r\n`'\".,;:!?()[]{}")
            if match in canonical:
                return canonical[match], True

    for line in reversed(text.splitlines()):
        line_clean = line.lower().strip(" \t\r\n`'\".,;:!?()[]{}")
        if line_clean in canonical:
            return canonical[line_clean], True
        line_candidates = _candidate_labels(line_clean, labels)
        if len(line_candidates) == 1:
            return canonical[line_candidates[0]], True

    candidates = _candidate_labels(lowered, labels)
    if len(candidates) == 1:
        return canonical[candidates[0]], True

    return UNKNOWN_LABEL, False


def config_to_text(config) -> str:
    if config is None:
        return "{}\n"
    if hasattr(config, "dump"):
        return config.dump()
    try:
        import yaml

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(config, ensure_ascii=False, indent=2) + "\n"


def _compute_metrics_fallback(true_labels, pred_labels, class_names):
    labels_for_cm = list(class_names)
    if any(pred not in class_names for pred in pred_labels):
        labels_for_cm.append(UNKNOWN_LABEL)
    label_to_idx = {label: idx for idx, label in enumerate(labels_for_cm)}
    cm = np.zeros((len(labels_for_cm), len(labels_for_cm)), dtype=np.int64)
    for true, pred in zip(true_labels, pred_labels):
        if true not in label_to_idx:
            continue
        pred = pred if pred in label_to_idx else UNKNOWN_LABEL
        cm[label_to_idx[true], label_to_idx[pred]] += 1

    per_class = {}
    f1_values = []
    supports = []
    for name in class_names:
        idx = label_to_idx[name]
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
            "support": support,
        }
        f1_values.append(f1)
        supports.append(support)

    total = len(true_labels)
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    weighted_den = sum(supports)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1_values, supports)) / weighted_den if weighted_den else 0.0
    return {
        "accuracy": 100.0 * correct / total if total else 0.0,
        "macro_f1": 100.0 * (sum(f1_values) / len(f1_values) if f1_values else 0.0),
        "weighted_f1": 100.0 * weighted_f1,
        "worst_class_recall": float(min((per_class[name]["recall"] for name in class_names), default=0.0)),
        "worst_class_accuracy": float(min((per_class[name]["recall"] for name in class_names), default=0.0)),
        "per_class": per_class,
        "confusion_matrix_labels": labels_for_cm,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(total),
        "num_parse_failures": int(sum(1 for pred in pred_labels if pred == UNKNOWN_LABEL)),
    }


def compute_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict:
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except Exception:
        return _compute_metrics_fallback(true_labels, pred_labels, class_names)

    labels_for_cm = list(class_names)
    if any(pred not in class_names for pred in pred_labels):
        labels_for_cm.append(UNKNOWN_LABEL)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=list(class_names),
        zero_division=0,
    )
    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx] * 100.0),
            "recall": float(recall[idx] * 100.0),
            "f1": float(f1[idx] * 100.0),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(true_labels, pred_labels, labels=labels_for_cm)
    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels) * 100.0),
        "macro_f1": float(f1_score(true_labels, pred_labels, labels=list(class_names), average="macro", zero_division=0) * 100.0),
        "weighted_f1": float(f1_score(true_labels, pred_labels, labels=list(class_names), average="weighted", zero_division=0) * 100.0),
        "worst_class_recall": float(min((per_class[name]["recall"] for name in class_names), default=0.0)),
        "worst_class_accuracy": float(min((per_class[name]["recall"] for name in class_names), default=0.0)),
        "per_class": per_class,
        "confusion_matrix_labels": labels_for_cm,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(true_labels)),
        "num_parse_failures": int(sum(1 for pred in pred_labels if pred == UNKNOWN_LABEL)),
    }
    return metrics

def save_confusion_matrix_csv(path: os.PathLike, labels: Sequence[str], matrix: Sequence[Sequence[int]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def save_predictions_csv(
    path: os.PathLike,
    rows: Iterable[Dict],
):
    fieldnames = [
        "photo_id",
        "image_path",
        "caption",
        "true_label",
        "pred_label",
        "raw_response",
        "success",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_evaluation_outputs(
    output_dir: os.PathLike,
    records,
    pred_labels: Sequence[str],
    class_names: Sequence[str],
    raw_responses: Optional[Sequence[str]] = None,
    successes: Optional[Sequence[bool]] = None,
    logits: Optional[np.ndarray] = None,
    config=None,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_labels = [record.label_name for record in records]
    raw_responses = raw_responses if raw_responses is not None else [""] * len(records)
    successes = successes if successes is not None else [True] * len(records)

    rows = []
    for record, pred_label, raw_response, success in zip(records, pred_labels, raw_responses, successes):
        rows.append(
            {
                "photo_id": record.photo_id,
                "image_path": record.image_path,
                "caption": record.caption,
                "true_label": record.label_name,
                "pred_label": pred_label,
                "raw_response": raw_response,
                "success": bool(success),
            }
        )

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_to_text(config))

    save_predictions_csv(output_dir / "predictions.csv", rows)
    metrics = compute_metrics(true_labels, pred_labels, class_names)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    try:
        from sklearn.metrics import classification_report

        report = classification_report(
            true_labels,
            pred_labels,
            labels=list(class_names),
            zero_division=0,
        )
    except Exception:
        lines = ["label precision recall f1 support"]
        for name in class_names:
            item = metrics["per_class"][name]
            lines.append(
                f"{name} {item['precision']:.4f} {item['recall']:.4f} "
                f"{item['f1']:.4f} {item['support']}"
            )
        report = "\n".join(lines) + "\n"
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix_csv(
        output_dir / "confusion_matrix.csv",
        metrics["confusion_matrix_labels"],
        metrics["confusion_matrix"],
    )

    failures = [row for row in rows if not row["success"]]
    save_predictions_csv(output_dir / "parse_failures.csv", failures)

    if logits is not None:
        np.save(output_dir / "logits.npy", logits)

    return metrics
