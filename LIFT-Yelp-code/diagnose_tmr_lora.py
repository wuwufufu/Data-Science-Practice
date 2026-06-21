import argparse
import csv
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from utils.eval_outputs import save_evaluation_outputs


def summarize(values):
    values = np.asarray(values)
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


def set_seed(cfg):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)


def build_cfg(args):
    from utils.config import _C as base_cfg

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(os.path.join("./configs/data", args.data + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/model", args.model + ".yaml"))
    cfg.merge_from_list(args.opts)
    cfg.output_dir = args.analysis_dir or os.path.join("outputs", "diagnostics", args.tag)
    cfg.model_dir = args.model_dir
    cfg.tmr_lora = args.method == "tmr_lora"
    cfg.lora = args.method == "vanilla_lora"
    cfg.adaptformer = args.method == "adaptformer"
    cfg.tmr_routing = args.routing
    cfg.lora_rank = args.lora_rank
    cfg.lora_alpha = args.lora_alpha
    cfg.loss_type = args.loss_type
    cfg.classifier = args.classifier
    cfg.test_only = False
    if args.root is not None:
        cfg.root = args.root
    if args.gpu is not None:
        cfg.gpu = args.gpu
    return cfg


def pick_loader(trainer, mode):
    if mode == "train":
        return trainer.train_test_loader
    if mode == "val":
        return trainer.val_loader
    return trainer.test_loader


def current_routing_weights(model):
    weights = []
    for module in model.modules():
        if module.__class__.__name__ != "TMRLoRA":
            continue
        if module.last_routing_weights is not None:
            weights.append(module.last_routing_weights.detach().float().cpu())
    if not weights:
        return None
    return torch.stack(weights, dim=0).mean(dim=0).numpy()


@torch.no_grad()
def run_forward_diagnostics(trainer, mode, output_dir):
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model.eval()
    loader = pick_loader(trainer, mode)
    dataset = loader.dataset

    records = []
    pred_labels = []
    logits_all = []
    routing_by_class = defaultdict(list)
    sample_cursor = 0

    for batch in loader:
        image = batch[0].to(trainer.device)
        label = batch[1].to(trainer.device)
        _bsz, _ncrops, _c, _h, _w = image.size()
        image = image.view(_bsz * _ncrops, _c, _h, _w)

        if _ncrops <= 5:
            label_for_forward = label.repeat_interleave(_ncrops)
            logits = model(image, labels=label_for_forward)
            route = current_routing_weights(model)
            if route is not None:
                route = route.reshape(_bsz, _ncrops, -1).mean(axis=1)
            logits = logits.view(_bsz, _ncrops, -1).mean(dim=1)
        else:
            logits_per_crop = []
            route_per_crop = []
            image = image.view(_bsz, _ncrops, _c, _h, _w)
            for k in range(_ncrops):
                logits_per_crop.append(model(image[:, k], labels=label))
                route = current_routing_weights(model)
                if route is not None:
                    route_per_crop.append(route)
            logits = torch.stack(logits_per_crop).mean(dim=0)
            route = np.stack(route_per_crop, axis=0).mean(axis=0) if route_per_crop else None

        preds = logits.argmax(dim=1)
        logits_np = logits.detach().cpu().numpy()
        labels_np = label.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        logits_all.append(logits_np)

        if route is not None:
            for label_id, weights in zip(labels_np, route):
                routing_by_class[trainer.classnames[int(label_id)]].append(weights)

        for i, (label_id, pred_id) in enumerate(zip(labels_np, preds_np)):
            ds_idx = sample_cursor + i
            image_path = dataset.img_path[ds_idx] if hasattr(dataset, "img_path") else ""
            photo_id = os.path.splitext(os.path.basename(image_path))[0] if image_path else str(ds_idx)
            caption = dataset.texts[ds_idx] if hasattr(dataset, "texts") else ""
            records.append(
                SimpleNamespace(
                    photo_id=photo_id,
                    image_path=image_path,
                    caption=caption,
                    label=int(label_id),
                    label_name=trainer.classnames[int(label_id)],
                )
            )
            pred_labels.append(trainer.classnames[int(pred_id)])
        sample_cursor += len(labels_np)

    logits_all = np.concatenate(logits_all, axis=0)
    metrics = save_evaluation_outputs(
        os.path.join(output_dir, f"eval_{mode}"),
        records=records,
        pred_labels=pred_labels,
        class_names=trainer.classnames,
        logits=logits_all,
        config=trainer.cfg,
    )

    pred_ids = logits_all.argmax(axis=1)
    true_ids = np.array([record.label for record in records])
    top1 = logits_all.max(axis=1)
    true_logits = logits_all[np.arange(logits_all.shape[0]), true_ids]
    correct = pred_ids == true_ids
    logit_summary = {
        "top1_correct": summarize(top1[correct]),
        "top1_wrong": summarize(top1[~correct]),
        "true_class_logit": summarize(true_logits),
        "per_class_true_logit": {
            name: summarize(true_logits[true_ids == idx])
            for idx, name in enumerate(trainer.classnames)
        },
    }
    with open(os.path.join(output_dir, "logit_distribution_summary.json"), "w", encoding="utf-8") as f:
        json.dump(logit_summary, f, ensure_ascii=False, indent=2)

    if routing_by_class:
        with open(os.path.join(output_dir, "expert_utilization.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "shared_weight", "head_weight", "tail_weight", "count"])
            for classname in trainer.classnames:
                values = routing_by_class.get(classname, [])
                if values:
                    arr = np.stack(values, axis=0)
                    mean = arr.mean(axis=0)
                    row = [classname, *mean[:3].tolist(), arr.shape[0]]
                else:
                    row = [classname, "", "", "", 0]
                writer.writerow(row)

    return metrics


def grad_norm_for_params(model):
    sq_sum = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        lowered = name.lower()
        if "lora" not in lowered and "router" not in lowered:
            continue
        sq_sum += float(param.grad.detach().float().pow(2).sum().cpu())
    return sq_sum ** 0.5


def run_gradient_norm(trainer, output_dir, max_batches):
    trainer.model.train()
    sums = defaultdict(float)
    counts = defaultdict(int)
    for batch_idx, batch in enumerate(trainer.train_loader):
        if batch_idx >= max_batches:
            break
        images = batch[0].to(trainer.device)
        labels = batch[1].to(trainer.device)
        for label_id in labels.unique().tolist():
            mask = labels == int(label_id)
            if not mask.any():
                continue
            trainer.optim.zero_grad()
            logits = trainer.model(images[mask], labels=labels[mask])
            loss = trainer.criterion(logits, labels[mask])
            loss.backward()
            norm = grad_norm_for_params(trainer.model)
            classname = trainer.classnames[int(label_id)]
            sums[classname] += norm
            counts[classname] += 1
    trainer.optim.zero_grad()

    with open(os.path.join(output_dir, "gradient_norm_by_class.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "mean_lora_router_grad_norm", "count"])
        for classname in trainer.classnames:
            count = counts[classname]
            mean = sums[classname] / count if count else ""
            writer.writerow([classname, mean, count])


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose LoRA/TMR-LoRA logits, metrics, routing, and gradients.")
    parser.add_argument("--data", "-d", type=str, default="yelp_lt")
    parser.add_argument("--model", "-m", type=str, default="clip_vit_b16")
    parser.add_argument("--method", type=str, default="tmr_lora", choices=["vanilla_lora", "tmr_lora", "adaptformer"])
    parser.add_argument("--routing", type=str, default="learned")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--analysis-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="tmr_lora")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--loss-type", type=str, default="CE")
    parser.add_argument("--classifier", type=str, default="CosineClassifier")
    parser.add_argument("--gradient-norm", action="store_true")
    parser.add_argument("--gradient-max-batches", type=int, default=10)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    from trainer import Trainer

    cfg = build_cfg(args)
    set_seed(cfg)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.load_model(args.model_dir)
    metrics = run_forward_diagnostics(trainer, args.mode, output_dir)
    if args.gradient_norm:
        run_gradient_norm(trainer, output_dir, args.gradient_max_batches)
    print(f"Saved diagnostics to {output_dir}")
    print(f"accuracy={metrics['accuracy']:.2f} macro_f1={metrics['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
