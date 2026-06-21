import argparse
import os
from pathlib import Path
from typing import Optional

from PIL import Image

from utils.eval_outputs import parse_label_response, save_evaluation_outputs, set_random_seed
from utils.yelp_data import (
    CANONICAL_LABELS,
    CLASS_DEFINITIONS,
    infer_yelp_image_root,
    read_classnames,
    read_yelp_records,
)


def build_prompt(mode: str, caption: str) -> str:
    labels = ", ".join(CANONICAL_LABELS)
    caption = caption if isinstance(caption, str) and caption.strip() else "<empty>"

    if mode == "image_only":
        definitions = "\n".join(f"- {name}: {CLASS_DEFINITIONS[name]}" for name in CANONICAL_LABELS)
        return (
            "You are classifying Yelp restaurant photos.\n\n"
            "The possible labels are exactly:\n"
            f"{definitions}\n\n"
            "Classify the image into exactly one of the following labels:\n"
            f"{labels}.\n\n"
            "Return only one label. Do not explain."
        )

    if mode == "image_caption":
        return (
            "You are classifying Yelp restaurant photos.\n\n"
            "Caption:\n"
            f"{caption}\n\n"
            "Classify the image into exactly one of the following labels:\n"
            f"{labels}.\n\n"
            "Return only one label. Do not explain."
        )

    if mode == "image_caption_definition":
        definitions = "\n".join(f"- {name}: {CLASS_DEFINITIONS[name]}" for name in CANONICAL_LABELS)
        return (
            "You are classifying Yelp restaurant photos.\n\n"
            "The possible labels are exactly:\n"
            f"{definitions}\n\n"
            "Caption:\n"
            f"{caption}\n\n"
            "Classify the image into exactly one of the following labels:\n"
            f"{labels}.\n\n"
            "Return only one label. Do not explain."
        )

    raise ValueError(f"Unsupported mode: {mode}")


class BaseVLMClient:
    def classify(self, image_path: str, prompt: str) -> str:
        raise NotImplementedError


class MockVLMClient(BaseVLMClient):
    """Small deterministic backend for parser/output smoke tests."""

    def classify(self, image_path: str, prompt: str) -> str:
        lowered = prompt.lower()
        if any(word in lowered for word in ["cocktail", "coffee", "beer", "wine", "beverage"]):
            return "drink"
        if any(word in lowered for word in ["menu", "price list", "ordering board"]):
            return "menu"
        if any(word in lowered for word in ["storefront", "street view", "patio", "entrance"]):
            return "outside"
        if any(word in lowered for word in ["dining room", "interior", "decor"]):
            return "inside"
        return "food"


class Qwen25VLClient(BaseVLMClient):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 8,
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "Qwen2.5-VL backend requires torch and a recent transformers build with "
                "Qwen2_5_VLForConditionalGeneration. Install the VLM dependencies first."
            ) from exc

        dtype = torch_dtype
        if torch_dtype == "bf16":
            dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.max_new_tokens = max_new_tokens

    def classify(self, image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def build_client(args) -> BaseVLMClient:
    if args.model == "mock":
        return MockVLMClient()
    if args.model == "qwen2_5_vl":
        model_name = args.model_name_or_path or "Qwen/Qwen2.5-VL-7B-Instruct"
        return Qwen25VLClient(
            model_name_or_path=model_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    raise ValueError(
        f"Model backend {args.model!r} is not implemented yet. "
        "Current implemented backends: mock, qwen2_5_vl."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate direct generative VLM classification on Yelp.")
    parser.add_argument("--model", type=str, default="qwen2_5_vl", help="Backend: qwen2_5_vl or mock.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="HF model id or local checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--mode",
        type=str,
        default="image_caption_definition",
        choices=["image_only", "image_caption", "image_caption_definition"],
    )
    parser.add_argument("--data_dir", type=str, default="datasets/Yelp")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    image_root = args.image_root or infer_yelp_image_root()
    if image_root is None:
        image_root = "."

    records = read_yelp_records(args.split, data_dir=args.data_dir, image_root=image_root, include_caption=True)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    classnames = read_classnames(Path(args.data_dir) / "classnames.txt")
    client = build_client(args)

    raw_responses = []
    pred_labels = []
    successes = []
    for idx, record in enumerate(records, start=1):
        prompt = build_prompt(args.mode, record.caption)
        try:
            if args.model != "mock" and not os.path.exists(record.image_path):
                raise FileNotFoundError(record.image_path)
            raw_response = client.classify(record.image_path, prompt)
            parsed_label, success = parse_label_response(raw_response, classnames)
        except Exception as exc:
            raw_response = f"ERROR: {type(exc).__name__}: {exc}"
            parsed_label, success = "unknown", False

        raw_responses.append(raw_response)
        pred_labels.append(parsed_label)
        successes.append(success)
        if idx % 50 == 0 or idx == len(records):
            print(f"[{idx}/{len(records)}] processed")

    metrics = save_evaluation_outputs(
        args.output_dir,
        records=records,
        pred_labels=pred_labels,
        class_names=classnames,
        raw_responses=raw_responses,
        successes=successes,
        config=vars(args),
    )
    print(f"Saved VLM direct outputs to {args.output_dir}")
    print(f"accuracy={metrics['accuracy']:.2f} macro_f1={metrics['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
