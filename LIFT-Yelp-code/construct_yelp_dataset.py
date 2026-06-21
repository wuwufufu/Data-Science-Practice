import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from PIL import Image, UnidentifiedImageError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct Yelp train/val/test txt files with image and text modalities."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("/data00/zhiyuan_huang/datasets/Yelp/filtered_photos.json"),
        help="Path to filtered_photos.json.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/data00/zhiyuan_huang/datasets/Yelp/yelp_filtered_image"),
        help="Directory containing Yelp filtered images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/Yelp"),
        help="Directory to save generated txt files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for class-wise shuffle and split.",
    )
    return parser.parse_args()


def sanitize_caption(caption: str) -> str:
    """Replace tabs/newlines to keep one-sample-per-line text format stable."""
    return " ".join(caption.replace("\t", " ").splitlines()).strip()


def allocate_counts(n: int, ratios: List[float]):
    """
    Allocate n samples into multiple splits by ratios.
    Uses largest-remainder method to keep total exact and close to target ratios.
    """
    raw = [n * r for r in ratios]
    base = [int(math.floor(v)) for v in raw]
    remain = n - sum(base)
    remainders = [v - b for v, b in zip(raw, base)]
    order = sorted(range(len(ratios)), key=lambda i: remainders[i], reverse=True)
    for i in order[:remain]:
        base[i] += 1
    return base


def classwise_split_with_missing_rate(
    items: List[Dict],
    rng: random.Random,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    split_names = ["train", "val", "test"]
    ratios = [train_ratio, val_ratio, 1.0 - train_ratio - val_ratio]
    split_data = {"train": [], "val": [], "test": []}
    grouped = defaultdict(list)
    for item in items:
        grouped[item["label_name"]].append(item)

    for label_name, samples in grouped.items():
        # Split within each class and caption-missing group separately to preserve
        # class-wise missing-caption rate across train/val/test.
        missing_groups = {True: [], False: []}
        for sample in samples:
            missing_groups[sample["caption_missing"]].append(sample)

        class_split_counter = {name: 0 for name in split_names}
        class_split_missing_counter = {name: 0 for name in split_names}

        for is_missing in [True, False]:
            bucket = missing_groups[is_missing]
            rng.shuffle(bucket)
            counts = allocate_counts(len(bucket), ratios)

            start = 0
            for split_name, count in zip(split_names, counts):
                end = start + count
                chunk = bucket[start:end]
                split_data[split_name].extend(chunk)
                class_split_counter[split_name] += len(chunk)
                if is_missing:
                    class_split_missing_counter[split_name] += len(chunk)
                start = end

        total = len(samples)
        total_missing = len(missing_groups[True])
        total_missing_rate = 100.0 * total_missing / total if total else 0.0
        print(
            f"[{label_name}] total={total}, missing={total_missing}/{total} "
            f"({total_missing_rate:.2f}%)"
        )
        for split_name in split_names:
            s_total = class_split_counter[split_name]
            s_missing = class_split_missing_counter[split_name]
            s_rate = 100.0 * s_missing / s_total if s_total else 0.0
            print(
                f"  - {split_name}: {s_total} samples, missing={s_missing}/{s_total} "
                f"({s_rate:.2f}%)"
            )

    return split_data


def write_image_txt(path: Path, records: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec['image_rel_path']} {rec['label_idx']}\n")


def write_text_txt(path: Path, records: List[Dict]):
    # Format: photo_id \t caption \t label_idx
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            caption = sanitize_caption(rec["caption"])
            f.write(f"{rec['photo_id']}\t{caption}\t{rec['label_idx']}\n")


def is_valid_image(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            img = Image.open(f)
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def main():
    args = parse_args()
    json_path = args.json_path.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Expected JSON top-level to be a list.")

    labels = sorted({item["label"] for item in raw_data})
    label2idx = {label: idx for idx, label in enumerate(labels)}

    dataset_root = image_dir.parent
    records = []
    missing_images = 0
    invalid_images = 0
    invalid_image_paths = []

    for item in raw_data:
        photo_id = item["photo_id"]
        label_name = item["label"]
        caption = item.get("caption", "")

        image_path = image_dir / f"{photo_id}.jpg"
        if not image_path.exists():
            missing_images += 1
            continue
        if not is_valid_image(image_path):
            invalid_images += 1
            invalid_image_paths.append(str(image_path))
            continue

        image_rel_path = image_path.relative_to(dataset_root).as_posix()
        records.append(
            {
                "photo_id": photo_id,
                "caption": caption,
                "caption_missing": sanitize_caption(caption) == "",
                "label_name": label_name,
                "label_idx": label2idx[label_name],
                "image_rel_path": image_rel_path,
            }
        )

    if missing_images > 0:
        print(f"Warning: skipped {missing_images} samples due to missing images.")
    if invalid_images > 0:
        print(f"Warning: skipped {invalid_images} samples due to invalid images.")
        invalid_txt = output_dir / "invalid_images.txt"
        with invalid_txt.open("w", encoding="utf-8") as f:
            for p in invalid_image_paths:
                f.write(f"{p}\n")
        print(f"Saved invalid image list to {invalid_txt}")

    rng = random.Random(args.seed)
    split_data = classwise_split_with_missing_rate(records, rng)

    # Shuffle each split globally to avoid class blocks.
    for split in split_data:
        rng.shuffle(split_data[split])

    for split in ("train", "val", "test"):
        image_txt = output_dir / f"Yelp_{split}.txt"
        text_txt = output_dir / f"Yelp_{split}_text.txt"
        write_image_txt(image_txt, split_data[split])
        write_text_txt(text_txt, split_data[split])
        print(f"Wrote {len(split_data[split])} samples to {image_txt}")
        print(f"Wrote {len(split_data[split])} samples to {text_txt}")

    classnames_txt = output_dir / "classnames.txt"
    with classnames_txt.open("w", encoding="utf-8") as f:
        for label_name, idx in label2idx.items():
            f.write(f"{idx} {label_name}\n")
    print(f"Wrote class names to {classnames_txt}")

    stat_path = output_dir / "split_stats.json"
    stats = {
        "seed": args.seed,
        "num_classes": len(labels),
        "num_samples_total": len(records),
        "num_missing_images": missing_images,
        "num_invalid_images": invalid_images,
        "num_samples_per_split": {k: len(v) for k, v in split_data.items()},
        "caption_missing_rate_per_split": {
            split: (
                sum(1 for rec in split_data[split] if rec["caption_missing"])
                / len(split_data[split])
            )
            for split in ("train", "val", "test")
        },
        "label2idx": label2idx,
    }
    with stat_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote split stats to {stat_path}")


if __name__ == "__main__":
    main()
