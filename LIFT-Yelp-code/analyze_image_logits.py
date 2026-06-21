import os
import json
import random
import argparse
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from trainer import Trainer
from utils.config import _C as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze image classification top1-logit distributions."
    )
    parser.add_argument("--data", "-d", type=str, default="yelp_lt", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="clip_vit_b16", help="model config file")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-dir", type=str, required=True, help="checkpoint directory")
    parser.add_argument("--analysis-dir", type=str, default=None, help="output analysis directory")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def setup_cfg(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)

    cfg.text_zeroshot = False
    cfg.text_finetune = False
    cfg.zero_shot = False
    cfg.test_only = False
    cfg.test_train = False

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name + "_image_logit_analysis")
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)

    return cfg


def set_seed(local_cfg):
    if local_cfg.seed is not None:
        seed = local_cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if local_cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def summarize(values: np.ndarray) -> Dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def compute_bin_accuracy(max_logits: np.ndarray, correct_flags: np.ndarray, num_bins: int = 10) -> List[Dict]:
    if max_logits.size == 0:
        return []
    lo, hi = float(max_logits.min()), float(max_logits.max())
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, num_bins + 1)
    out = []
    for i in range(num_bins):
        left, right = bins[i], bins[i + 1]
        if i == num_bins - 1:
            idxs = (max_logits >= left) & (max_logits <= right)
        else:
            idxs = (max_logits >= left) & (max_logits < right)
        cnt = int(idxs.sum())
        acc = float(correct_flags[idxs].mean()) if cnt > 0 else None
        out.append({"bin_id": i, "left": left, "right": right, "count": cnt, "accuracy": acc})
    return out


def pick_loader(trainer: Trainer, mode: str):
    if mode == "train":
        return trainer.train_test_loader
    if mode == "val":
        return trainer.val_loader
    return trainer.test_loader


def plot_hist(correct_logits: np.ndarray, wrong_logits: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 5))
    bins = 60
    if correct_logits.size > 0:
        plt.hist(correct_logits, bins=bins, density=True, alpha=0.6, label="correct top1 logit")
    if wrong_logits.size > 0:
        plt.hist(wrong_logits, bins=bins, density=True, alpha=0.6, label="wrong top1 logit")
    plt.xlabel("Top-1 Logit")
    plt.ylabel("Density")
    plt.title("Image Classification Top-1 Logit Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_box(correct_logits: np.ndarray, wrong_logits: np.ndarray, save_path: str):
    data = []
    labels = []
    if correct_logits.size > 0:
        data.append(correct_logits)
        labels.append("correct")
    if wrong_logits.size > 0:
        data.append(wrong_logits)
        labels.append("wrong")
    if not data:
        return
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Top-1 Logit")
    plt.title("Top-1 Logit Boxplot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def run_analysis(trainer: Trainer, mode: str):
    model = trainer.model
    loader = pick_loader(trainer, mode)

    correct_top1_logits = []
    wrong_top1_logits = []
    all_top1_logits = []
    all_correct_flags = []
    all_true_logits = []

    for batch in loader:
        image = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)

        _bsz, _ncrops, _c, _h, _w = image.size()
        image = image.view(_bsz * _ncrops, _c, _h, _w)

        if _ncrops <= 5:
            logits = model(image)
            logits = logits.view(_bsz, _ncrops, -1).mean(dim=1)
        else:
            logits_per_crop = []
            image = image.view(_bsz, _ncrops, _c, _h, _w)
            for k in range(_ncrops):
                logits_per_crop.append(model(image[:, k]))
            logits = torch.stack(logits_per_crop).mean(dim=0)

        pred = logits.argmax(dim=1)
        correct = pred.eq(labels)
        top1_logits = logits.max(dim=1).values
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        correct_top1_logits.append(top1_logits[correct].detach().cpu().numpy())
        wrong_top1_logits.append(top1_logits[~correct].detach().cpu().numpy())
        all_top1_logits.append(top1_logits.detach().cpu().numpy())
        all_correct_flags.append(correct.detach().cpu().numpy().astype(np.float32))
        all_true_logits.append(true_logits.detach().cpu().numpy())

    correct_top1_logits = np.concatenate(correct_top1_logits) if correct_top1_logits else np.array([])
    wrong_top1_logits = np.concatenate(wrong_top1_logits) if wrong_top1_logits else np.array([])
    all_top1_logits = np.concatenate(all_top1_logits) if all_top1_logits else np.array([])
    all_correct_flags = np.concatenate(all_correct_flags) if all_correct_flags else np.array([])
    all_true_logits = np.concatenate(all_true_logits) if all_true_logits else np.array([])

    return {
        "correct_top1_logits": correct_top1_logits,
        "wrong_top1_logits": wrong_top1_logits,
        "all_top1_logits": all_top1_logits,
        "all_correct_flags": all_correct_flags,
        "all_true_logits": all_true_logits,
    }


def main():
    args = parse_args()
    local_cfg = setup_cfg(args)
    set_seed(local_cfg)

    trainer = Trainer(local_cfg)
    trainer.load_model(args.model_dir)
    result = run_analysis(trainer, args.mode)

    if args.analysis_dir is None:
        analysis_dir = os.path.join(local_cfg.output_dir, "logit_analysis_image", args.mode)
    else:
        analysis_dir = args.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)

    correct = result["correct_top1_logits"]
    wrong = result["wrong_top1_logits"]
    all_top1 = result["all_top1_logits"]
    all_correct = result["all_correct_flags"]
    all_true = result["all_true_logits"]

    summary = {
        "mode": args.mode,
        "task": "image_classification",
        "correct_top1_logit": summarize(correct),
        "wrong_top1_logit": summarize(wrong),
        "all_top1_logit": summarize(all_top1),
        "all_true_class_logit": summarize(all_true),
        "gap_mean_correct_minus_wrong": float(correct.mean() - wrong.mean()) if correct.size and wrong.size else None,
        "top1_logit_bin_accuracy": compute_bin_accuracy(all_top1, all_correct, num_bins=10),
    }

    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        os.path.join(analysis_dir, "raw_logits.npz"),
        correct_top1_logits=correct,
        wrong_top1_logits=wrong,
        all_top1_logits=all_top1,
        all_correct_flags=all_correct,
        all_true_logits=all_true,
    )

    plot_hist(correct, wrong, os.path.join(analysis_dir, "top1_logit_hist.png"))
    plot_box(correct, wrong, os.path.join(analysis_dir, "top1_logit_box.png"))

    print("=== Image Logit Analysis ===")
    print(f"mode: {args.mode}")
    print(f"analysis_dir: {analysis_dir}")
    print(f"correct_count: {summary['correct_top1_logit'].get('count', 0)}")
    print(f"wrong_count: {summary['wrong_top1_logit'].get('count', 0)}")
    print(f"correct_mean_top1_logit: {summary['correct_top1_logit'].get('mean', None)}")
    print(f"wrong_mean_top1_logit: {summary['wrong_top1_logit'].get('mean', None)}")
    print(f"mean_gap(correct-wrong): {summary['gap_mean_correct_minus_wrong']}")


if __name__ == "__main__":
    main()
