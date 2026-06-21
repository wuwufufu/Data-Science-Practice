import argparse
import os
import random


PEFT_FLAGS = [
    "full_tuning",
    "bias_tuning",
    "ln_tuning",
    "vpt_shallow",
    "vpt_deep",
    "adapter",
    "adaptformer",
    "lora",
    "lora_mlp",
    "tmr_lora",
    "tmr_lora_mlp",
    "ssf_attn",
    "ssf_mlp",
    "ssf_ln",
    "mask",
]


def normalize_classifier(name: str) -> str:
    mapping = {
        "cosine": "CosineClassifier",
        "linear": "LinearClassifier",
        "l2": "L2NormedClassifier",
        "layernorm": "LayerNormedClassifier",
    }
    return mapping.get(name.lower(), name)


def normalize_loss(name: str) -> str:
    key = name.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "ce": "CE",
        "cross_entropy": "CE",
        "logit_adjustment": "LA",
        "la": "LA",
        "focal": "Focal",
        "ldam": "LDAM",
        "cb": "CB",
        "grw": "GRW",
        "bs": "BS",
        "lade": "LADE",
    }
    return mapping.get(key, name)


def set_seed(cfg):
    import numpy as np
    import torch

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def build_cfg(args):
    from utils.config import _C as base_cfg

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(os.path.join("./configs/data", args.data + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/model", args.model + ".yaml"))
    cfg.merge_from_list(args.opts)

    for flag in PEFT_FLAGS:
        setattr(cfg, flag, False)

    if args.method == "vanilla_lora":
        cfg.lora = True
    elif args.method == "tmr_lora":
        cfg.tmr_lora = True
        cfg.tmr_routing = args.routing
    elif args.method == "adapter":
        cfg.adapter = True
    elif args.method == "adaptformer":
        cfg.adaptformer = True
    elif args.method == "lora_mlp":
        cfg.lora_mlp = True
    elif args.method == "tmr_lora_mlp":
        cfg.tmr_lora_mlp = True
        cfg.tmr_routing = args.routing
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    cfg.lora_rank = args.lora_rank
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    cfg.classifier = normalize_classifier(args.classifier)
    cfg.loss_type = normalize_loss(args.loss)
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed

    if args.root is not None:
        cfg.root = args.root
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.init_head is not None:
        cfg.init_head = args.init_head
    if args.test_only:
        cfg.test_only = True
    if args.model_dir is not None:
        cfg.model_dir = args.model_dir

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP PEFT baselines, including TMR-LoRA, on Yelp.")
    parser.add_argument("--data", "-d", type=str, default="yelp_lt")
    parser.add_argument("--model", "-m", type=str, default="clip_vit_b16")
    parser.add_argument(
        "--method",
        type=str,
        default="vanilla_lora",
        choices=["vanilla_lora", "tmr_lora", "adapter", "adaptformer", "lora_mlp", "tmr_lora_mlp"],
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="uniform",
        choices=["oracle", "learned", "uniform", "random", "vanilla_lora", "shared"],
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--classifier", type=str, default="cosine")
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--init_head", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    from trainer import Trainer
    from utils.logger import setup_logger

    cfg = build_cfg(args)
    print(f"Output directory: {cfg.output_dir}")
    setup_logger(cfg.output_dir)
    print("** Config **")
    print(cfg)
    print("************")
    set_seed(cfg)

    trainer = Trainer(cfg)
    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir)
    if cfg.test_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
