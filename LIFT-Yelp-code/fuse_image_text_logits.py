import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

from clip import clip
from trainer import Trainer
from utils.config import _C as base_cfg
from utils.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse image/text logits for Yelp classification.")
    parser.add_argument("--data", "-d", type=str, default="yelp_lt")
    parser.add_argument("--model", "-m", type=str, default="clip_vit_b16")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["val", "test"],
        help="single-split mode when --val-tune-then-test is disabled",
    )
    parser.add_argument(
        "--val-tune-then-test",
        action="store_true",
        help="search fusion alpha on val (macro_f1), then evaluate on test",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=None,
        help="fixed coef: fused=logits_img + alpha*logits_txt. If omitted with "
        "--val-tune-then-test, alpha is searched on val. Single-mode default: 1.0",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.0,
        help="grid search lower bound when tuning",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
        help="grid search upper bound when tuning",
    )
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=601,
        help="number of grid points in [alpha-min, alpha-max]",
    )
    parser.add_argument(
        "--image-model-dir",
        type=str,
        required=True,
        help="image model checkpoint directory",
    )
    parser.add_argument(
        "--text-model-dir",
        type=str,
        required=True,
        help="text model checkpoint directory",
    )
    parser.add_argument(
        "--image-opts",
        nargs="*",
        default=["adaptformer", "True", "loss_type", "CE", "classifier", "CosineClassifier", "gpu", "2"],
        help="opts for image model config",
    )
    parser.add_argument(
        "--text-opts",
        nargs="*",
        default=[
            "text_finetune",
            "True",
            "adaptformer",
            "True",
            "loss_type",
            "CE",
            "classifier",
            "LinearClassifier",
            "gpu",
            "1",
        ],
        help="opts for text model config",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="directory to save fused logits/predictions",
    )
    return parser.parse_args()


def build_cfg(data_name: str, model_name: str, opts: List[str]):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(os.path.join("./configs/data", data_name + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/model", model_name + ".yaml"))
    cfg.merge_from_list(opts)
    if cfg.output_dir is None:
        cfg.output_dir = os.path.join("./output", f"{data_name}_{model_name}_fuse_tmp")
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    return cfg


def read_caption_map(mode: str) -> Dict[str, str]:
    if mode == "val":
        text_file = "./datasets/Yelp/Yelp_val_text.txt"
    else:
        text_file = "./datasets/Yelp/Yelp_test_text.txt"

    caption_map = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            photo_id = parts[0]
            caption = parts[1] if len(parts) > 1 else ""
            caption_map[photo_id] = caption
    return caption_map


def pick_loader(trainer: Trainer, mode: str):
    if mode == "val":
        return trainer.val_loader
    return trainer.test_loader


def macro_f1_percent(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Match Evaluator.evaluate() macro-F1 scaling (trainer picks best by this)."""
    pred = logits.argmax(axis=1)
    mf1 = f1_score(
        y_true,
        pred,
        average="macro",
        labels=np.unique(y_true),
    )
    return 100.0 * float(mf1)


def search_best_fusion_alpha(
    image_logits: np.ndarray,
    text_logits: np.ndarray,
    y_true: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    num_steps: int,
) -> Tuple[float, float, np.ndarray]:
    """Return (best_alpha, best_macro_f1_on_val_percent, alphas_used)."""
    alphas = np.linspace(alpha_min, alpha_max, num_steps, dtype=np.float64)
    best_a = float(alphas[0])
    best_f1 = macro_f1_percent(image_logits, y_true)

    # alpha=0 is pure image baseline
    for a in alphas:
        logits = image_logits + float(a) * text_logits
        mf1 = macro_f1_percent(logits, y_true)
        if mf1 > best_f1:
            best_f1 = mf1
            best_a = float(a)
    return best_a, best_f1, alphas


def evaluator_detailed_report(
    image_trainer: Trainer,
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    title: str,
) -> Dict:
    """Same breakdown as Trainer.test(): use Evaluator.evaluate() formatting."""
    if logits_np.shape[0] == 0:
        print(f"\n===== {title} =====\n(no samples)\n")
        return {}

    print(f"\n===== {title} =====")
    dev = image_trainer.device
    evaluator = Evaluator(
        image_trainer.cfg,
        image_trainer.many_idxs,
        image_trainer.med_idxs,
        image_trainer.few_idxs,
    )
    logits_t = torch.from_numpy(logits_np.astype(np.float32)).to(dev)
    labels_t = torch.from_numpy(labels_np.astype(np.int64)).to(dev)
    batch_size = 64
    n = logits_t.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        evaluator.process(logits_t[start:end], labels_t[start:end])
    results = evaluator.evaluate()
    return dict(results)


@torch.no_grad()
def collect_image_text_logits(
    image_trainer: Trainer,
    text_trainer: Trainer,
    mode: str,
):
    image_model = image_trainer.model.module if hasattr(image_trainer.model, "module") else image_trainer.model
    text_model = text_trainer.model.module if hasattr(text_trainer.model, "module") else text_trainer.model
    image_model.eval()
    text_model.eval()

    loader = pick_loader(image_trainer, mode)
    dataset = loader.dataset
    caption_map = read_caption_map(mode)

    y_true_all = []
    y_pred_image = []
    has_caption_flags = []
    image_logits_all = []
    text_logits_all = []

    sample_cursor = 0
    text_device = text_trainer.device

    for batch in loader:
        image = batch[0].to(image_trainer.device)
        label = batch[1].to(image_trainer.device)

        _bsz, _ncrops, _c, _h, _w = image.size()
        image = image.view(_bsz * _ncrops, _c, _h, _w)

        if _ncrops <= 5:
            image_logits = image_model(image)
            image_logits = image_logits.view(_bsz, _ncrops, -1).mean(dim=1)
        else:
            logits_per_crop = []
            image = image.view(_bsz, _ncrops, _c, _h, _w)
            for k in range(_ncrops):
                logits_per_crop.append(image_model(image[:, k]))
            image_logits = torch.stack(logits_per_crop).mean(dim=0)

        batch_photo_ids = []
        for i in range(_bsz):
            path = dataset.img_path[sample_cursor + i]
            photo_id = os.path.splitext(os.path.basename(path))[0]
            batch_photo_ids.append(photo_id)
        sample_cursor += _bsz

        captions = [caption_map.get(pid, "") for pid in batch_photo_ids]
        valid_idxs = [i for i, text in enumerate(captions) if isinstance(text, str) and text.strip() != ""]
        has_caption = np.zeros(_bsz, dtype=np.bool_)
        if valid_idxs:
            has_caption[valid_idxs] = True
            valid_captions = [captions[i] for i in valid_idxs]
            tokenized = clip.tokenize(valid_captions, truncate=True).to(text_device)
            text_logits_valid = text_model(tokenized)
            text_logits_valid = text_logits_valid.to(image_logits.device)
            full_text_logits = torch.zeros_like(image_logits)
            full_text_logits[valid_idxs] = text_logits_valid
        else:
            full_text_logits = torch.zeros_like(image_logits)

        y_true_all.append(label.cpu().numpy())
        y_pred_image.append(image_logits.argmax(dim=1).cpu().numpy())
        has_caption_flags.append(has_caption)
        image_logits_all.append(image_logits.cpu().numpy())
        text_logits_all.append(full_text_logits.cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_image = np.concatenate(y_pred_image)
    has_caption_flags = np.concatenate(has_caption_flags)
    image_logits_all = np.concatenate(image_logits_all)
    text_logits_all = np.concatenate(text_logits_all)

    return {
        "y_true": y_true_all,
        "y_pred_image": y_pred_image,
        "has_caption": has_caption_flags,
        "image_logits": image_logits_all,
        "text_logits": text_logits_all,
    }


def fused_logits_with_alpha(image_logits: np.ndarray, text_logits: np.ndarray, alpha: float) -> np.ndarray:
    return image_logits + float(alpha) * text_logits


def run_reports_for_split(
    image_trainer: Trainer,
    pack: Dict,
    alpha_used: float,
    split_title: str,
) -> Tuple[Dict, Dict, Dict, Dict]:
    y_true = pack["y_true"]
    has_cap = pack["has_caption"]
    img_logits = pack["image_logits"]
    txt_logits = pack["text_logits"]
    fused_logits = fused_logits_with_alpha(img_logits, txt_logits, alpha_used)

    print(f"\n########## Split: {split_title} ##########")

    img_metrics = evaluator_detailed_report(image_trainer, img_logits, y_true, "image_only")
    fused_metrics = evaluator_detailed_report(
        image_trainer,
        fused_logits,
        y_true,
        f"fused logits: image + ({alpha_used:.6g}) * text",
    )

    if has_cap.any():
        fused_cap = evaluator_detailed_report(
            image_trainer,
            fused_logits[has_cap],
            y_true[has_cap],
            f"fused subset (caption exists), alpha={alpha_used:.6g}",
        )
    else:
        fused_cap = {}

    if (~has_cap).any():
        fused_no_cap = evaluator_detailed_report(
            image_trainer,
            fused_logits[~has_cap],
            y_true[~has_cap],
            "fused subset: caption_missing (≈ image only)",
        )
    else:
        fused_no_cap = {}

    summary_block = {
        "split": split_title,
        "fusion_alpha_used": alpha_used,
        "num_samples": int(len(y_true)),
        "num_caption_available": int(has_cap.sum()),
        "num_caption_missing": int((~has_cap).sum()),
        "image_only": img_metrics,
        "fused": fused_metrics,
        "fused_caption_available": fused_cap,
        "fused_caption_missing": fused_no_cap,
    }
    return summary_block, img_metrics, fused_metrics, {
        "fused_logits": fused_logits,
        "y_pred_image": img_logits.argmax(axis=1),
        "y_pred_fused": fused_logits.argmax(axis=1),
    }


def main():
    args = parse_args()

    image_cfg = build_cfg(args.data, args.model, args.image_opts)
    text_cfg = build_cfg(args.data, args.model, args.text_opts)

    image_trainer = Trainer(image_cfg)
    image_trainer.load_model(args.image_model_dir)

    text_trainer = Trainer(text_cfg)
    text_trainer.load_model(args.text_model_dir)

    if args.save_dir is None:
        save_dir = os.path.join(args.image_model_dir, "fusion_with_text_model")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    if args.val_tune_then_test:
        print("\n===== Step 1/2: gather val logits, search best fusion-alpha (macro-F1) =====")
        pack_val = collect_image_text_logits(image_trainer, text_trainer, "val")
        if args.fusion_alpha is not None:
            alpha_best = args.fusion_alpha
            mf1_best = macro_f1_percent(
                fused_logits_with_alpha(pack_val["image_logits"], pack_val["text_logits"], alpha_best),
                pack_val["y_true"],
            )
            print(
                f"Using fixed --fusion-alpha={alpha_best:g} "
                f"(val macro-F1={mf1_best:.2f}%, not searched)"
            )
        else:
            alpha_best, mf1_best, alphas_used = search_best_fusion_alpha(
                pack_val["image_logits"],
                pack_val["text_logits"],
                pack_val["y_true"],
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                num_steps=max(2, args.alpha_steps),
            )
            mf1_pure_img = macro_f1_percent(pack_val["image_logits"], pack_val["y_true"])
            print(
                f"Grid search [{args.alpha_min}, {args.alpha_max}] ({len(alphas_used)} steps).\n"
                f"Best alpha = {alpha_best:.6g}  | val macro-F1 = {mf1_best:.2f}%"
                f"  | image-only baseline macro-F1 = {mf1_pure_img:.2f}%"
            )

        tune_record = {
            "best_fusion_alpha": alpha_best,
            "val_macro_f1_at_best_alpha_percent": mf1_best,
            "alpha_search": {
                "alpha_min": args.alpha_min,
                "alpha_max": args.alpha_max,
                "alpha_steps": args.alpha_steps,
            },
            "image_model_dir": args.image_model_dir,
            "text_model_dir": args.text_model_dir,
        }
        with open(os.path.join(save_dir, "val_tune_fusion_alpha.json"), "w", encoding="utf-8") as f:
            json.dump(_clean(tune_record), f, ensure_ascii=False, indent=2)

        summary_val, *_ = run_reports_for_split(
            image_trainer, pack_val, alpha_best, "validation"
        )

        print("\n===== Step 2/2: test set with tuned alpha =====")
        pack_te = collect_image_text_logits(image_trainer, text_trainer, "test")
        summary_test, _, _, extras_te = run_reports_for_split(
            image_trainer, pack_te, alpha_best, "test"
        )

        full_summary = {
            "workflow": "val_tune_then_test",
            "fusion_alpha_best_on_val": alpha_best,
            "validation_summary": summary_val,
            "test_summary": summary_test,
            "image_model_dir": args.image_model_dir,
            "text_model_dir": args.text_model_dir,
        }
        with open(os.path.join(save_dir, "tuned_test_fusion_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_clean(full_summary), f, ensure_ascii=False, indent=2)

        np.savez_compressed(
            os.path.join(save_dir, "tuned_test_fusion_outputs.npz"),
            y_true=pack_te["y_true"],
            fusion_alpha=float(alpha_best),
            y_pred_image=extras_te["y_pred_image"],
            y_pred_fused=extras_te["y_pred_fused"],
            has_caption=pack_te["has_caption"].astype(np.int64),
            image_logits=pack_te["image_logits"],
            text_logits=pack_te["text_logits"],
            fused_logits=extras_te["fused_logits"],
        )
        np.savez_compressed(
            os.path.join(save_dir, "val_tune_fusion_logits.npz"),
            y_true=pack_val["y_true"],
            fusion_alpha=float(alpha_best),
            has_caption=pack_val["has_caption"].astype(np.int64),
            image_logits=pack_val["image_logits"],
            text_logits=pack_val["text_logits"],
        )

        print(f"\nSaved val tune json: {os.path.join(save_dir, 'val_tune_fusion_alpha.json')}")
        print(f"Saved tuned full summary: {os.path.join(save_dir, 'tuned_test_fusion_summary.json')}")
        print(f"Saved test logits: {os.path.join(save_dir, 'tuned_test_fusion_outputs.npz')}")
        return

    # Single split (legacy): fixed alpha defaults to 1.0
    alpha_used = args.fusion_alpha if args.fusion_alpha is not None else 1.0
    result = collect_image_text_logits(image_trainer, text_trainer, args.mode)
    y_true = result["y_true"]
    fused_logits_np = fused_logits_with_alpha(result["image_logits"], result["text_logits"], alpha_used)

    summary_block, _, _, _ = run_reports_for_split(
        image_trainer,
        result,
        alpha_used,
        args.mode,
    )
    summary = {
        **summary_block,
        "mode": args.mode,
        "image_model_dir": args.image_model_dir,
        "text_model_dir": args.text_model_dir,
    }

    with open(os.path.join(save_dir, f"{args.mode}_fusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(summary), f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        os.path.join(save_dir, f"{args.mode}_fusion_outputs.npz"),
        y_true=y_true,
        fusion_alpha=float(alpha_used),
        y_pred_image=result["y_pred_image"],
        y_pred_fused=fused_logits_np.argmax(axis=1),
        has_caption=result["has_caption"].astype(np.int64),
        image_logits=result["image_logits"],
        text_logits=result["text_logits"],
        fused_logits=fused_logits_np,
    )

    print(f"\nSaved fusion summary to: {os.path.join(save_dir, f'{args.mode}_fusion_summary.json')}")
    print(f"Saved fusion outputs to: {os.path.join(save_dir, f'{args.mode}_fusion_outputs.npz')}")


if __name__ == "__main__":
    main()
