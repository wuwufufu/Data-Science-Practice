import argparse
import json
from pathlib import Path

from utils.eval_outputs import config_to_text, set_random_seed
from utils.yelp_data import (
    CANONICAL_LABELS,
    CLASS_DEFINITIONS,
    infer_yelp_image_root,
    read_yelp_records,
)


def build_instruction(caption: str) -> str:
    caption = caption if isinstance(caption, str) and caption.strip() else "<empty>"
    definitions = "\n".join(f"- {name}: {CLASS_DEFINITIONS[name]}" for name in CANONICAL_LABELS)
    labels = ", ".join(CANONICAL_LABELS)
    return (
        "Classify this Yelp restaurant photo into exactly one of the following labels:\n"
        f"{labels}.\n\n"
        "Definitions:\n"
        f"{definitions}\n\n"
        "Caption:\n"
        f"{caption}\n\n"
        "Return only one label."
    )


def hf_record(record):
    return {
        "id": record.photo_id,
        "image": record.image_path,
        "caption": record.caption,
        "question": build_instruction(record.caption),
        "answer": record.label_name,
        "label": record.label,
    }


def llava_record(record):
    return {
        "id": record.photo_id,
        "image": record.image_path,
        "conversations": [
            {"from": "human", "value": "<image>\n" + build_instruction(record.caption)},
            {"from": "gpt", "value": record.label_name},
        ],
        "label": record.label,
        "caption": record.caption,
    }


def qwen_record(record):
    return {
        "id": record.photo_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record.image_path},
                    {"type": "text", "text": build_instruction(record.caption)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": record.label_name}]},
        ],
        "answer": record.label_name,
        "label": record.label,
        "caption": record.caption,
    }


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv_arg(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Build Yelp VLM-LoRA instruction tuning files.")
    parser.add_argument("--data_dir", type=str, default="datasets/Yelp")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_lora_dataset")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--formats", type=str, default="hf_jsonl,llava_json,qwen_jsonl")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_root = args.image_root or infer_yelp_image_root()
    if image_root is None:
        image_root = "."

    splits = parse_csv_arg(args.splits)
    formats = parse_csv_arg(args.formats)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_to_text(vars(args)))

    for split in splits:
        records = read_yelp_records(split, data_dir=args.data_dir, image_root=image_root, include_caption=True)
        if "hf_jsonl" in formats:
            write_jsonl(output_dir / f"{split}_hf.jsonl", [hf_record(record) for record in records])
        if "llava_json" in formats:
            with open(output_dir / f"{split}_llava.json", "w", encoding="utf-8") as f:
                json.dump([llava_record(record) for record in records], f, ensure_ascii=False, indent=2)
        if "qwen_jsonl" in formats:
            write_jsonl(output_dir / f"{split}_qwen.jsonl", [qwen_record(record) for record in records])
        print(f"{split}: wrote {len(records)} samples")

    print(f"Saved VLM-LoRA instruction data to {output_dir}")


if __name__ == "__main__":
    main()
