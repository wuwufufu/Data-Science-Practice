import argparse
import json
import os
from pathlib import Path

from PIL import Image

from utils.eval_outputs import (
    config_to_text,
    parse_label_response,
    save_evaluation_outputs,
    set_random_seed,
)
from utils.yelp_data import infer_yelp_image_root, read_classnames, read_yelp_records


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class JsonlInstructionDataset:
    def __init__(self, path, max_samples=None):
        self.rows = load_jsonl(path)
        if max_samples is not None:
            self.rows = self.rows[: max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class QwenVLCollator:
    def __init__(self, processor, max_length=None, include_class_labels=False):
        self.processor = processor
        self.max_length = max_length
        self.include_class_labels = include_class_labels
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        full_texts = []
        prompt_texts = []
        images = []
        for ex in examples:
            image = Image.open(ex["image"]).convert("RGB")
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": ex["question"]},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.append(image)

        common_kwargs = dict(
            images=images,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = self.processor(text=full_texts, **common_kwargs)
        prompt_batch = self.processor(text=prompt_texts, **common_kwargs)

        labels = batch["input_ids"].clone()
        for row_idx in range(labels.shape[0]):
            prompt_len = int(prompt_batch["attention_mask"][row_idx].sum().item())
            labels[row_idx, :prompt_len] = -100
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        if self.include_class_labels:
            import torch

            batch["class_labels"] = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)
        return batch


def import_training_deps():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
    except Exception as exc:
        raise ImportError(
            "VLM-LoRA training requires torch, transformers with Qwen2.5-VL support, and peft. "
            "Install them before running training, for example: pip install -U transformers peft accelerate."
        ) from exc
    return torch, LoraConfig, get_peft_model, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments


def parse_int_list(value):
    if value is None or value == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    if value is None or value == "":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def target_modules_from_args(args):
    if args.target_modules:
        return [item.strip() for item in args.target_modules.split(",") if item.strip()]
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.projector_lora:
        modules.extend(["visual.merger.mlp.0", "visual.merger.mlp.2"])
    return modules


def resolve_tmr_runtime_args(args):
    if args.adapter_method != "tmr_lora":
        return
    if args.tmr_expert_layout == "shared_class":
        class_labels = parse_int_list(args.tmr_class_labels)
        if not class_labels:
            class_labels = [0, 1, 2, 3, 4]
            args.tmr_class_labels = ",".join(str(x) for x in class_labels)
        expected = 1 + len(class_labels)
        if args.tmr_num_experts != expected:
            print(f"Using shared_class expert layout: overriding tmr_num_experts={args.tmr_num_experts} -> {expected}")
            args.tmr_num_experts = expected
    if args.tmr_attnres and args.tmr_attnres_mode == "none":
        args.tmr_attnres_mode = "full"


def discover_qwen_image_token_ids(processor, model, explicit_value=None):
    explicit = parse_int_list(explicit_value) if explicit_value else []
    if explicit:
        return explicit

    token_ids = set()
    config = getattr(model, "config", None)
    for attr in ("image_token_id", "vision_token_id", "vision_start_token_id", "vision_end_token_id"):
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value >= 0:
            token_ids.add(int(value))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
                token_ids.add(int(token_id))
    return sorted(token_ids)


def tmr_adapter_config_from_args(args, target_modules):
    return {
        "adapter_method": args.adapter_method,
        "target_modules": target_modules,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "tmr_num_experts": args.tmr_num_experts,
        "tmr_routing": args.tmr_routing,
        "tmr_aux_weight": args.tmr_aux_weight,
        "tmr_head_labels": parse_int_list(args.tmr_head_labels),
        "tmr_tail_labels": parse_int_list(args.tmr_tail_labels),
        "tmr_head_weights": parse_float_list(args.tmr_head_weights),
        "tmr_tail_weights": parse_float_list(args.tmr_tail_weights),
        "tmr_oracle_fallback": args.tmr_oracle_fallback,
        "tmr_expert_layout": args.tmr_expert_layout,
        "tmr_class_labels": parse_int_list(args.tmr_class_labels),
        "tmr_class_expert_weight": args.tmr_class_expert_weight,
        "tmr_menu_label": args.tmr_menu_label,
        "tmr_menu_expert_weight": args.tmr_menu_expert_weight,
        "tmr_router_entropy_weight": args.tmr_router_entropy_weight,
        "tmr_router_balance_weight": args.tmr_router_balance_weight,
        "tmr_depth_attention": args.tmr_depth_attention,
        "tmr_depth_block_size": args.tmr_depth_block_size,
        "tmr_depth_max_blocks": args.tmr_depth_max_blocks,
        "tmr_depth_context_scale": args.tmr_depth_context_scale,
        "tmr_attnres": args.tmr_attnres,
        "tmr_attnres_mode": args.tmr_attnres_mode,
        "tmr_attnres_block_size": args.tmr_attnres_block_size,
        "tmr_attnres_max_blocks": args.tmr_attnres_max_blocks,
        "tmr_attnres_context_scale": args.tmr_attnres_context_scale,
        "tmr_attnres_residual_scale": args.tmr_attnres_residual_scale,
        "tmr_image_token_ids": getattr(args, "tmr_resolved_image_token_ids", parse_int_list(args.tmr_image_token_ids)),
        "class_balanced_sampling": args.class_balanced_sampling,
        "init_adapter_dir": args.init_adapter_dir,
    }


def attnres_adapter_config_from_args(args, target_modules):
    return {
        "adapter_method": args.adapter_method,
        "target_modules": target_modules,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "attnres_block_size": args.attnres_block_size,
        "attnres_max_blocks": args.attnres_max_blocks,
        "attnres_scale": args.attnres_scale,
        "attnres_image_token_ids": getattr(
            args,
            "attnres_resolved_image_token_ids",
            parse_int_list(args.attnres_image_token_ids),
        ),
        "class_balanced_sampling": args.class_balanced_sampling,
        "init_adapter_dir": args.init_adapter_dir,
    }


def build_tmr_epoch_save_callback(callback_base, save_adapter_fn, adapter_config):
    class VlmTMREpochSaveCallback(callback_base):
        def __init__(self):
            self.saved_epochs = set()
            self.next_epoch = 1

        def _save_epoch(self, training_args, state, model, epoch_id):
            if epoch_id in self.saved_epochs or model is None:
                return
            epoch_dir = os.path.join(training_args.output_dir, f"epoch-{epoch_id}")
            config = dict(adapter_config)
            config["saved_epoch"] = epoch_id
            config["global_step"] = int(getattr(state, "global_step", 0))
            save_adapter_fn(model, epoch_dir, config=config)
            self.saved_epochs.add(epoch_id)
            print(f"Saved VLM adapter to {epoch_dir}")

        def on_step_end(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            if max_steps <= 0:
                return control
            total_epochs = max(1, int(round(float(args.num_train_epochs))))
            model = kwargs.get("model")
            while self.next_epoch <= total_epochs:
                boundary = max(1, int(round(max_steps * self.next_epoch / total_epochs)))
                if int(getattr(state, "global_step", 0) or 0) < boundary:
                    break
                self._save_epoch(args, state, model, self.next_epoch)
                self.next_epoch += 1
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            # Fallback for trainers that report epoch boundaries more reliably than step boundaries.
            if not getattr(state, "is_world_process_zero", True):
                return control
            if getattr(state, "epoch", None) is None:
                return control
            epoch_id = int(float(state.epoch) + 1e-6)
            model = kwargs.get("model")
            while self.next_epoch <= epoch_id:
                self._save_epoch(args, state, model, self.next_epoch)
                self.next_epoch += 1
            return control

    return VlmTMREpochSaveCallback()


def build_vlm_trainer_class(base_trainer):
    class VlmTMRTrainer(base_trainer):
        def __init__(
            self,
            *trainer_args,
            tmr_aux_weight=0.0,
            tmr_head_label_ids=None,
            tmr_tail_label_ids=None,
            tmr_head_weights=None,
            tmr_tail_weights=None,
            tmr_expert_layout="head_tail",
            tmr_class_label_ids=None,
            tmr_class_expert_weight=0.6,
            tmr_menu_label_id=3,
            tmr_menu_expert_weight=0.7,
            tmr_router_entropy_weight=0.0,
            tmr_router_balance_weight=0.0,
            class_balanced_sampling=False,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.tmr_aux_weight = float(tmr_aux_weight)
            self.tmr_head_label_ids = tmr_head_label_ids
            self.tmr_tail_label_ids = tmr_tail_label_ids
            self.tmr_head_weights = tmr_head_weights
            self.tmr_tail_weights = tmr_tail_weights
            self.tmr_expert_layout = tmr_expert_layout
            self.tmr_class_label_ids = tmr_class_label_ids
            self.tmr_class_expert_weight = float(tmr_class_expert_weight)
            self.tmr_menu_label_id = int(tmr_menu_label_id)
            self.tmr_menu_expert_weight = float(tmr_menu_expert_weight)
            self.tmr_router_entropy_weight = float(tmr_router_entropy_weight)
            self.tmr_router_balance_weight = float(tmr_router_balance_weight)
            self.class_balanced_sampling = class_balanced_sampling

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            from vlm_tmr_lora import (
                clear_vlm_tmr_class_labels,
                clear_vlm_tmr_token_context,
                collect_vlm_tmr_routing_loss,
                set_vlm_tmr_class_labels,
                set_vlm_tmr_token_context,
            )

            class_labels = inputs.pop("class_labels", None)
            if class_labels is not None:
                set_vlm_tmr_class_labels(model, class_labels)
            set_vlm_tmr_token_context(model, inputs.get("input_ids"), inputs.get("attention_mask"))
            try:
                outputs = model(**inputs)
                loss = outputs.loss
                if self.tmr_aux_weight > 0 and class_labels is not None:
                    aux_loss = collect_vlm_tmr_routing_loss(
                        model,
                        class_labels,
                        head_label_ids=self.tmr_head_label_ids,
                        tail_label_ids=self.tmr_tail_label_ids,
                        head_weights=self.tmr_head_weights,
                        tail_weights=self.tmr_tail_weights,
                        expert_layout=self.tmr_expert_layout,
                        class_label_ids=self.tmr_class_label_ids,
                        class_expert_weight=self.tmr_class_expert_weight,
                        menu_label_id=self.tmr_menu_label_id,
                        menu_expert_weight=self.tmr_menu_expert_weight,
                        entropy_weight=self.tmr_router_entropy_weight,
                        balance_weight=self.tmr_router_balance_weight,
                    )
                    if aux_loss is not None:
                        loss = loss + self.tmr_aux_weight * aux_loss
            finally:
                clear_vlm_tmr_class_labels(model)
                clear_vlm_tmr_token_context(model)
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            if not self.class_balanced_sampling:
                return super().get_train_dataloader()

            from torch.utils.data import DataLoader, WeightedRandomSampler

            from vlm_tmr_lora import class_balanced_sample_weights

            weights = class_balanced_sample_weights(self.train_dataset.rows)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    return VlmTMRTrainer


def train_qwen(args):
    (
        torch,
        LoraConfig,
        get_peft_model,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = import_training_deps()

    model_name = args.model_name_or_path or "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = args.torch_dtype
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = target_modules_from_args(args)
    image_token_arg = args.attnres_image_token_ids if args.adapter_method == "attnres_lora" else args.tmr_image_token_ids
    image_token_ids = discover_qwen_image_token_ids(processor, model, image_token_arg)
    args.tmr_resolved_image_token_ids = image_token_ids
    args.attnres_resolved_image_token_ids = image_token_ids
    tmr_adapter_config = None
    attnres_adapter_config = None
    callbacks = []
    if args.adapter_method == "lora":
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainer_cls = Trainer
    elif args.adapter_method == "tmr_lora":
        from vlm_tmr_lora import (
            inject_vlm_tmr_lora,
            load_vlm_tmr_lora_adapter,
            save_vlm_tmr_lora_adapter,
            set_only_vlm_tmr_trainable,
            trainable_parameter_summary,
        )

        replacement_count = inject_vlm_tmr_lora(
            model,
            target_modules=target_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            num_experts=args.tmr_num_experts,
            routing=args.tmr_routing,
            head_label_ids=parse_int_list(args.tmr_head_labels),
            tail_label_ids=parse_int_list(args.tmr_tail_labels),
            oracle_head_weights=parse_float_list(args.tmr_head_weights),
            oracle_tail_weights=parse_float_list(args.tmr_tail_weights),
            oracle_fallback=args.tmr_oracle_fallback,
            expert_layout=args.tmr_expert_layout,
            class_label_ids=parse_int_list(args.tmr_class_labels),
            class_expert_weight=args.tmr_class_expert_weight,
            menu_label_id=args.tmr_menu_label,
            menu_expert_weight=args.tmr_menu_expert_weight,
            depth_attention=args.tmr_depth_attention,
            depth_block_size=args.tmr_depth_block_size,
            depth_max_blocks=args.tmr_depth_max_blocks,
            depth_context_scale=args.tmr_depth_context_scale,
            attnres_mode=args.tmr_attnres_mode,
            attnres_block_size=args.tmr_attnres_block_size,
            attnres_max_blocks=args.tmr_attnres_max_blocks,
            attnres_context_scale=args.tmr_attnres_context_scale,
            attnres_residual_scale=args.tmr_attnres_residual_scale,
            image_token_ids=image_token_ids,
        )
        if replacement_count == 0:
            raise ValueError(f"No target Linear modules were replaced. target_modules={target_modules}")
        if args.init_adapter_dir:
            missing, unexpected = load_vlm_tmr_lora_adapter(model, args.init_adapter_dir, strict=True)
            print(
                f"Initialized VLM-TMR-LoRA adapter from {args.init_adapter_dir}; "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        set_only_vlm_tmr_trainable(model)
        total_params, trainable_params, pct = trainable_parameter_summary(model)
        print(f"Injected VLM-TMR-LoRA into {replacement_count} Linear modules")
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {pct:.4f}")
        trainer_cls = build_vlm_trainer_class(Trainer)
        tmr_adapter_config = tmr_adapter_config_from_args(args, target_modules)
        callbacks.append(build_tmr_epoch_save_callback(TrainerCallback, save_vlm_tmr_lora_adapter, tmr_adapter_config))
    elif args.adapter_method == "attnres_lora":
        from vlm_tmr_lora import (
            inject_vlm_attnres_lora,
            load_vlm_attnres_lora_adapter,
            save_vlm_attnres_lora_adapter,
            set_only_vlm_attnres_trainable,
            trainable_parameter_summary,
        )

        replacement_count = inject_vlm_attnres_lora(
            model,
            target_modules=target_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            attnres_block_size=args.attnres_block_size,
            attnres_max_blocks=args.attnres_max_blocks,
            attnres_scale=args.attnres_scale,
            image_token_ids=image_token_ids,
        )
        if replacement_count == 0:
            raise ValueError(f"No target Linear modules were replaced. target_modules={target_modules}")
        if args.init_adapter_dir:
            missing, unexpected = load_vlm_attnres_lora_adapter(model, args.init_adapter_dir, strict=True)
            print(
                f"Initialized VLM-AttnRes-LoRA adapter from {args.init_adapter_dir}; "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        set_only_vlm_attnres_trainable(model)
        total_params, trainable_params, pct = trainable_parameter_summary(model)
        print(f"Injected VLM-AttnRes-LoRA into {replacement_count} Linear modules")
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {pct:.4f}")
        trainer_cls = build_vlm_trainer_class(Trainer)
        attnres_adapter_config = attnres_adapter_config_from_args(args, target_modules)
        callbacks.append(build_tmr_epoch_save_callback(TrainerCallback, save_vlm_attnres_lora_adapter, attnres_adapter_config))
    else:
        raise ValueError(f"Unsupported adapter_method: {args.adapter_method}")

    train_dataset = JsonlInstructionDataset(args.train_jsonl, max_samples=args.max_train_samples)
    eval_dataset = JsonlInstructionDataset(args.val_jsonl, max_samples=args.max_eval_train_samples) if args.val_jsonl else None
    collator = QwenVLCollator(
        processor,
        max_length=args.max_length,
        include_class_labels=args.adapter_method == "tmr_lora",
    )

    save_strategy = args.save_strategy
    if save_strategy is None:
        save_strategy = "no" if args.adapter_method in {"tmr_lora", "attnres_lora"} else "epoch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=(eval_dataset is not None and save_strategy != "no"),
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        bf16=args.torch_dtype == "bf16",
        fp16=args.torch_dtype == "fp16",
        remove_unused_columns=False,
        report_to=["tensorboard"] if args.tensorboard else [],
        seed=args.seed,
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if args.adapter_method == "tmr_lora":
        trainer_kwargs.update(
            tmr_aux_weight=args.tmr_aux_weight,
            tmr_head_label_ids=parse_int_list(args.tmr_head_labels),
            tmr_tail_label_ids=parse_int_list(args.tmr_tail_labels),
            tmr_head_weights=parse_float_list(args.tmr_head_weights),
            tmr_tail_weights=parse_float_list(args.tmr_tail_weights),
            tmr_expert_layout=args.tmr_expert_layout,
            tmr_class_label_ids=parse_int_list(args.tmr_class_labels),
            tmr_class_expert_weight=args.tmr_class_expert_weight,
            tmr_menu_label_id=args.tmr_menu_label,
            tmr_menu_expert_weight=args.tmr_menu_expert_weight,
            tmr_router_entropy_weight=args.tmr_router_entropy_weight,
            tmr_router_balance_weight=args.tmr_router_balance_weight,
            class_balanced_sampling=args.class_balanced_sampling,
        )
    elif args.adapter_method == "attnres_lora":
        trainer_kwargs.update(
            tmr_aux_weight=0.0,
            class_balanced_sampling=args.class_balanced_sampling,
        )
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    if args.adapter_method == "tmr_lora":
        save_vlm_tmr_lora_adapter(model, args.output_dir, config=tmr_adapter_config)
    elif args.adapter_method == "attnres_lora":
        from vlm_tmr_lora import save_vlm_attnres_lora_adapter

        save_vlm_attnres_lora_adapter(model, args.output_dir, config=attnres_adapter_config)
    else:
        trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    return model, processor


def load_qwen_for_eval(args):
    (
        torch,
        LoraConfig,
        get_peft_model,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = import_training_deps()

    if args.adapter_method not in {"tmr_lora", "attnres_lora"}:
        raise ValueError("--eval_only currently supports --adapter_method tmr_lora or attnres_lora")

    model_name = args.model_name_or_path or "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = args.torch_dtype
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32

    adapter_dir = args.eval_adapter_dir or args.output_dir
    config_name = "attnres_adapter_config.json" if args.adapter_method == "attnres_lora" else "tmr_adapter_config.json"
    config_path = Path(adapter_dir) / config_name
    adapter_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model, "config"):
        model.config.use_cache = True

    target_modules = adapter_config.get("target_modules") or target_modules_from_args(args)
    if args.adapter_method == "attnres_lora":
        from vlm_tmr_lora import inject_vlm_attnres_lora, load_vlm_attnres_lora_adapter

        image_token_ids = adapter_config.get("attnres_image_token_ids")
        if image_token_ids is None:
            image_token_ids = discover_qwen_image_token_ids(processor, model, args.attnres_image_token_ids)
        replacement_count = inject_vlm_attnres_lora(
            model,
            target_modules=target_modules,
            rank=adapter_config.get("lora_rank", args.lora_rank),
            alpha=adapter_config.get("lora_alpha", args.lora_alpha),
            dropout=adapter_config.get("lora_dropout", args.lora_dropout),
            attnres_block_size=adapter_config.get("attnres_block_size", args.attnres_block_size),
            attnres_max_blocks=adapter_config.get("attnres_max_blocks", args.attnres_max_blocks),
            attnres_scale=adapter_config.get("attnres_scale", args.attnres_scale),
            image_token_ids=image_token_ids,
        )
        if replacement_count == 0:
            raise ValueError(f"No target Linear modules were replaced. target_modules={target_modules}")
        missing, unexpected = load_vlm_attnres_lora_adapter(model, adapter_dir, strict=True)
        adapter_label = "VLM-AttnRes-LoRA"
    else:
        from vlm_tmr_lora import inject_vlm_tmr_lora, load_vlm_tmr_lora_adapter

        image_token_ids = adapter_config.get("tmr_image_token_ids")
        if image_token_ids is None:
            image_token_ids = discover_qwen_image_token_ids(processor, model, args.tmr_image_token_ids)
        replacement_count = inject_vlm_tmr_lora(
            model,
            target_modules=target_modules,
            rank=adapter_config.get("lora_rank", args.lora_rank),
            alpha=adapter_config.get("lora_alpha", args.lora_alpha),
            dropout=adapter_config.get("lora_dropout", args.lora_dropout),
            num_experts=adapter_config.get("tmr_num_experts", args.tmr_num_experts),
            routing=adapter_config.get("tmr_routing", args.tmr_routing),
            head_label_ids=adapter_config.get("tmr_head_labels", parse_int_list(args.tmr_head_labels)),
            tail_label_ids=adapter_config.get("tmr_tail_labels", parse_int_list(args.tmr_tail_labels)),
            oracle_head_weights=adapter_config.get("tmr_head_weights", parse_float_list(args.tmr_head_weights)),
            oracle_tail_weights=adapter_config.get("tmr_tail_weights", parse_float_list(args.tmr_tail_weights)),
            oracle_fallback=adapter_config.get("tmr_oracle_fallback", args.tmr_oracle_fallback),
            expert_layout=adapter_config.get("tmr_expert_layout", args.tmr_expert_layout),
            class_label_ids=adapter_config.get("tmr_class_labels", parse_int_list(args.tmr_class_labels)),
            class_expert_weight=adapter_config.get("tmr_class_expert_weight", args.tmr_class_expert_weight),
            menu_label_id=adapter_config.get("tmr_menu_label", args.tmr_menu_label),
            menu_expert_weight=adapter_config.get("tmr_menu_expert_weight", args.tmr_menu_expert_weight),
            depth_attention=adapter_config.get("tmr_depth_attention", args.tmr_depth_attention),
            depth_block_size=adapter_config.get("tmr_depth_block_size", args.tmr_depth_block_size),
            depth_max_blocks=adapter_config.get("tmr_depth_max_blocks", args.tmr_depth_max_blocks),
            depth_context_scale=adapter_config.get("tmr_depth_context_scale", args.tmr_depth_context_scale),
            attnres_mode=adapter_config.get("tmr_attnres_mode", args.tmr_attnres_mode),
            attnres_block_size=adapter_config.get("tmr_attnres_block_size", args.tmr_attnres_block_size),
            attnres_max_blocks=adapter_config.get("tmr_attnres_max_blocks", args.tmr_attnres_max_blocks),
            attnres_context_scale=adapter_config.get("tmr_attnres_context_scale", args.tmr_attnres_context_scale),
            attnres_residual_scale=adapter_config.get("tmr_attnres_residual_scale", args.tmr_attnres_residual_scale),
            image_token_ids=image_token_ids,
        )
        if replacement_count == 0:
            raise ValueError(f"No target Linear modules were replaced. target_modules={target_modules}")
        missing, unexpected = load_vlm_tmr_lora_adapter(model, adapter_dir, strict=True)
        adapter_label = "VLM-TMR-LoRA"
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    print(
        f"Loaded {adapter_label} adapter from {adapter_dir}; "
        f"replaced={replacement_count} missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model, processor


def generate_one_qwen(model, processor, image_path, question, max_new_tokens):
    import torch

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    from vlm_tmr_lora import clear_vlm_tmr_token_context, set_vlm_tmr_token_context

    set_vlm_tmr_token_context(model, inputs.get("input_ids"), inputs.get("attention_mask"))
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        clear_vlm_tmr_token_context(model)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def evaluate_after_training(args, model, processor):
    from vlm_tmr_lora import clear_vlm_tmr_class_labels

    clear_vlm_tmr_class_labels(model)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()

    classnames = read_classnames(Path(args.data_dir) / "classnames.txt")
    image_root = args.image_root or infer_yelp_image_root()
    if image_root is None:
        image_root = "."
    records = read_yelp_records(args.test_split, data_dir=args.data_dir, image_root=image_root, include_caption=True)
    if args.max_eval_samples is not None:
        records = records[: args.max_eval_samples]

    rows = load_jsonl(args.test_jsonl) if args.test_jsonl else None
    question_by_id = {row["id"]: row["question"] for row in rows} if rows else {}

    raw_responses = []
    pred_labels = []
    successes = []
    for idx, record in enumerate(records, start=1):
        question = question_by_id.get(record.photo_id)
        if question is None:
            from build_vlm_lora_dataset import build_instruction

            question = build_instruction(record.caption)
        try:
            raw = generate_one_qwen(model, processor, record.image_path, question, args.max_new_tokens)
            pred, success = parse_label_response(raw, classnames)
        except Exception as exc:
            raw = f"ERROR: {type(exc).__name__}: {exc}"
            pred, success = "unknown", False
        raw_responses.append(raw)
        pred_labels.append(pred)
        successes.append(success)
        if idx % 50 == 0 or idx == len(records):
            print(f"[eval {idx}/{len(records)}] processed")

    eval_dir = os.path.join(args.output_dir, "eval_test")
    metrics = save_evaluation_outputs(
        eval_dir,
        records=records,
        pred_labels=pred_labels,
        class_names=classnames,
        raw_responses=raw_responses,
        successes=successes,
        config=vars(args),
    )
    print(f"Saved VLM-LoRA eval outputs to {eval_dir}")
    print(f"accuracy={metrics['accuracy']:.2f} macro_f1={metrics['macro_f1']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a generative VLM with LoRA for Yelp classification.")
    parser.add_argument("--model", type=str, default="qwen2_5_vl", choices=["qwen2_5_vl"])
    parser.add_argument("--adapter_method", type=str, default="lora", choices=["lora", "tmr_lora", "attnres_lora"])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default="outputs/vlm_lora_dataset/train_hf.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="outputs/vlm_lora_dataset/val_hf.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="outputs/vlm_lora_dataset/test_hf.jsonl")
    parser.add_argument("--data_dir", type=str, default="datasets/Yelp")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--test_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_lora/qwen2_5_vl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--projector_lora", action="store_true")
    parser.add_argument("--tmr_routing", type=str, default="learned", choices=["learned", "oracle", "uniform", "random", "vanilla_lora", "shared"])
    parser.add_argument("--tmr_num_experts", type=int, default=3)
    parser.add_argument("--tmr_head_labels", type=str, default="1,2")
    parser.add_argument("--tmr_tail_labels", type=str, default="0,3,4")
    parser.add_argument("--tmr_head_weights", type=str, default="0.5,0.5,0.0")
    parser.add_argument("--tmr_tail_weights", type=str, default="0.5,0.0,0.5")
    parser.add_argument("--tmr_aux_weight", type=float, default=0.05)
    parser.add_argument("--tmr_oracle_fallback", type=str, default="uniform", choices=["uniform", "random", "vanilla_lora", "shared"])
    parser.add_argument("--tmr_expert_layout", type=str, default="head_tail", choices=["head_tail", "shared_class"])
    parser.add_argument("--tmr_class_labels", type=str, default="0,1,2,3,4")
    parser.add_argument("--tmr_class_expert_weight", type=float, default=0.6)
    parser.add_argument("--tmr_menu_label", type=int, default=3)
    parser.add_argument("--tmr_menu_expert_weight", type=float, default=0.7)
    parser.add_argument("--tmr_router_entropy_weight", type=float, default=0.0)
    parser.add_argument("--tmr_router_balance_weight", type=float, default=0.0)
    parser.add_argument("--tmr_depth_attention", action="store_true")
    parser.add_argument("--tmr_depth_block_size", type=int, default=8)
    parser.add_argument("--tmr_depth_max_blocks", type=int, default=8)
    parser.add_argument("--tmr_depth_context_scale", type=float, default=0.1)
    parser.add_argument("--tmr_attnres", action="store_true")
    parser.add_argument("--tmr_attnres_mode", type=str, default="none", choices=["none", "router", "full"])
    parser.add_argument("--tmr_attnres_block_size", type=int, default=7)
    parser.add_argument("--tmr_attnres_max_blocks", type=int, default=8)
    parser.add_argument("--tmr_attnres_context_scale", type=float, default=0.03)
    parser.add_argument("--tmr_attnres_residual_scale", type=float, default=0.01)
    parser.add_argument("--tmr_image_token_ids", type=str, default="")
    parser.add_argument("--attnres_block_size", type=int, default=7)
    parser.add_argument("--attnres_max_blocks", type=int, default=8)
    parser.add_argument("--attnres_scale", type=float, default=0.03)
    parser.add_argument("--attnres_image_token_ids", type=str, default="")
    parser.add_argument("--class_balanced_sampling", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default=None, choices=["no", "epoch"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_adapter_dir", type=str, default=None)
    parser.add_argument("--init_adapter_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_tmr_runtime_args(args)
    set_random_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_to_text(vars(args)))

    if args.eval_only:
        model, processor = load_qwen_for_eval(args)
        evaluate_after_training(args, model, processor)
        return

    model, processor = train_qwen(args)
    if not args.skip_eval:
        evaluate_after_training(args, model, processor)


if __name__ == "__main__":
    main()
