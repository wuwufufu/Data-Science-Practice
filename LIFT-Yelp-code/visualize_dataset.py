import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Yelp dataset statistics.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/Yelp"),
        help="Directory containing Yelp txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot"),
        help="Directory to save visualization images.",
    )
    return parser.parse_args()


def load_classnames(classnames_path: Path):
    idx2name = {}
    with classnames_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(" ", 1)
            idx2name[int(idx_str)] = name
    return idx2name


def load_text_split(text_path: Path):
    records = []
    with text_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            photo_id = parts[0]
            caption = parts[1]
            label = int(parts[2])
            records.append({"photo_id": photo_id, "caption": caption, "label": label})
    return records


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def annotate_bars(ax, bars, labels=None, fmt="{:.0f}", fontsize=8):
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        text = fmt.format(height) if labels is None else labels[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_overall_class_distribution(all_records, idx2name, output_dir: Path):
    labels = [rec["label"] for rec in all_records]
    counter = Counter(labels)
    class_ids = sorted(idx2name.keys())
    class_names = [idx2name[i] for i in class_ids]
    counts = [counter.get(i, 0) for i in class_ids]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, counts, color="#4C72B0")
    ax.set_title("Yelp Class Distribution (All Splits)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax, bars, labels=[str(c) for c in counts], fontsize=9)
    save_figure(output_dir / "yelp_class_distribution_overall.png")


def plot_split_class_distribution(split_records, idx2name, output_dir: Path):
    class_ids = sorted(idx2name.keys())
    class_names = [idx2name[i] for i in class_ids]
    splits = ["train", "val", "test"]
    x = np.arange(len(class_ids))
    width = 0.24
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, split in enumerate(splits):
        counter = Counter(rec["label"] for rec in split_records[split])
        y = [counter.get(cid, 0) for cid in class_ids]
        bars = ax.bar(
            x + (i - 1) * width,
            y,
            width=width,
            label=split,
            color=colors[split],
        )
        annotate_bars(ax, bars, labels=[str(v) for v in y], fontsize=7)

    ax.set_title("Yelp Class Distribution by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.legend()
    save_figure(output_dir / "yelp_class_distribution_by_split.png")


def plot_caption_missing_by_split(split_records, output_dir: Path):
    splits = ["train", "val", "test"]
    missing_rates = []
    totals = []
    missings = []
    for split in splits:
        records = split_records[split]
        total = len(records)
        missing = sum(1 for r in records if not r["caption"].strip())
        totals.append(total)
        missings.append(missing)
        missing_rates.append(100.0 * missing / total if total else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(splits, missing_rates, color="#8172B3")
    ax.set_title("Caption Missing Rate by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Missing Rate (%)")
    labels = [
        f"{rate:.2f}%\n({missing}/{total})"
        for rate, missing, total in zip(missing_rates, missings, totals)
    ]
    annotate_bars(ax, bars, labels=labels, fontsize=9)
    save_figure(output_dir / "yelp_caption_missing_by_split.png")


def plot_caption_missing_by_class(all_records, idx2name, output_dir: Path):
    class_ids = sorted(idx2name.keys())
    class_names = [idx2name[i] for i in class_ids]
    class_total = Counter(rec["label"] for rec in all_records)
    class_missing = Counter(
        rec["label"] for rec in all_records if not rec["caption"].strip()
    )

    missing_rates = []
    total_counts = []
    missing_counts = []
    for cid in class_ids:
        total = class_total.get(cid, 0)
        missing = class_missing.get(cid, 0)
        rate = 100.0 * missing / total if total else 0.0
        total_counts.append(total)
        missing_counts.append(missing)
        missing_rates.append(rate)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_ids))
    bars = ax1.bar(x, missing_rates, color="#CCB974", width=0.6)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Missing Rate (%)")
    ax1.set_title("Caption Missing Rate by Class (All Splits)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=20)
    labels = [
        f"{rate:.2f}%\n({missing}/{total})"
        for rate, missing, total in zip(missing_rates, missing_counts, total_counts)
    ]
    annotate_bars(ax1, bars, labels=labels, fontsize=8)
    save_figure(output_dir / "yelp_caption_missing_by_class.png")


def plot_caption_missing_by_class_and_split(split_records, idx2name, output_dir: Path):
    class_ids = sorted(idx2name.keys())
    class_names = [idx2name[i] for i in class_ids]
    splits = ["train", "val", "test"]
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}

    all_records = split_records["train"] + split_records["val"] + split_records["test"]
    overall_total = Counter(rec["label"] for rec in all_records)
    overall_missing = Counter(
        rec["label"] for rec in all_records if not rec["caption"].strip()
    )
    overall_rates = [
        100.0 * overall_missing.get(cid, 0) / overall_total.get(cid, 1) for cid in class_ids
    ]

    x = np.arange(len(class_ids))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, split in enumerate(splits):
        records = split_records[split]
        split_total = Counter(rec["label"] for rec in records)
        split_missing = Counter(rec["label"] for rec in records if not rec["caption"].strip())
        rates = [
            100.0 * split_missing.get(cid, 0) / split_total.get(cid, 1) for cid in class_ids
        ]
        bars = ax.bar(
            x + (i - 1) * width,
            rates,
            width=width,
            label=split,
            color=colors[split],
        )
        annotate_bars(ax, bars, labels=[f"{r:.2f}%" for r in rates], fontsize=7)

    ax.plot(
        x,
        overall_rates,
        color="black",
        marker="o",
        linestyle="--",
        linewidth=1.2,
        label="overall",
    )
    for xi, rate in zip(x, overall_rates):
        ax.text(xi, rate, f"{rate:.2f}%", ha="center", va="bottom", fontsize=8, color="black")

    ax.set_title("Caption Missing Rate by Class and Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Missing Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.legend()
    save_figure(output_dir / "yelp_caption_missing_by_class_split.png")


def plot_caption_length_distribution(all_records, output_dir: Path):
    lengths = [len(rec["caption"].split()) for rec in all_records if rec["caption"].strip()]
    if not lengths:
        print("No non-empty captions, skip caption length plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    counts, bins, _ = ax.hist(lengths, bins=40, color="#64B5CD", edgecolor="white")
    ax.set_title("Caption Length Distribution (Non-empty Captions)")
    ax.set_xlabel("Caption Length (word count)")
    ax.set_ylabel("Number of Samples")
    stats_text = (
        f"count={len(lengths)}\n"
        f"mean={np.mean(lengths):.2f}\n"
        f"median={np.median(lengths):.2f}\n"
        f"max={np.max(lengths)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )

    # Mark the peak bin value to provide explicit numeric reference.
    peak_idx = int(np.argmax(counts))
    peak_x = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    peak_y = counts[peak_idx]
    ax.text(peak_x, peak_y, f"peak={int(peak_y)}", ha="center", va="bottom", fontsize=8)
    save_figure(output_dir / "yelp_caption_length_distribution.png")


def print_stats(split_records, idx2name):
    print("=== Summary ===")
    all_records = split_records["train"] + split_records["val"] + split_records["test"]
    total = len(all_records)
    missing = sum(1 for rec in all_records if not rec["caption"].strip())
    print(f"Total samples: {total}")
    print(f"Total missing captions: {missing} ({100.0 * missing / total:.2f}%)")
    print("")

    for split in ["train", "val", "test"]:
        records = split_records[split]
        n = len(records)
        miss = sum(1 for rec in records if not rec["caption"].strip())
        print(f"[{split}] samples={n}, missing={miss}, missing_rate={100.0 * miss / n:.2f}%")

    print("")
    miss_by_class = defaultdict(int)
    count_by_class = defaultdict(int)
    for rec in all_records:
        count_by_class[rec["label"]] += 1
        if not rec["caption"].strip():
            miss_by_class[rec["label"]] += 1

    print("=== Missing by class ===")
    for cid in sorted(idx2name.keys()):
        cname = idx2name[cid]
        total_c = count_by_class[cid]
        miss_c = miss_by_class[cid]
        rate_c = 100.0 * miss_c / total_c if total_c else 0.0
        print(f"{cname:>8s}: {miss_c:4d}/{total_c:<5d} ({rate_c:.2f}%)")

    print("")
    print("=== Missing by class and split ===")
    for cid in sorted(idx2name.keys()):
        cname = idx2name[cid]
        total_c = count_by_class[cid]
        overall_rate = 100.0 * miss_by_class[cid] / total_c if total_c else 0.0
        detail = []
        for split in ["train", "val", "test"]:
            split_records_c = [rec for rec in split_records[split] if rec["label"] == cid]
            split_total = len(split_records_c)
            split_missing = sum(1 for rec in split_records_c if not rec["caption"].strip())
            split_rate = 100.0 * split_missing / split_total if split_total else 0.0
            detail.append(f"{split}:{split_rate:.2f}%({split_missing}/{split_total})")
        print(f"{cname:>8s} overall={overall_rate:.2f}% | " + " | ".join(detail))


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    idx2name = load_classnames(dataset_dir / "classnames.txt")
    split_records = {
        "train": load_text_split(dataset_dir / "Yelp_train_text.txt"),
        "val": load_text_split(dataset_dir / "Yelp_val_text.txt"),
        "test": load_text_split(dataset_dir / "Yelp_test_text.txt"),
    }

    all_records = split_records["train"] + split_records["val"] + split_records["test"]

    print_stats(split_records, idx2name)
    plot_overall_class_distribution(all_records, idx2name, output_dir)
    plot_split_class_distribution(split_records, idx2name, output_dir)
    plot_caption_missing_by_split(split_records, output_dir)
    plot_caption_missing_by_class(all_records, idx2name, output_dir)
    plot_caption_missing_by_class_and_split(split_records, idx2name, output_dir)
    plot_caption_length_distribution(all_records, output_dir)


if __name__ == "__main__":
    main()
