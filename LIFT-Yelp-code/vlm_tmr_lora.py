import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_HEAD_LABELS = (1, 2)  # food, inside in datasets/Yelp/classnames.txt
DEFAULT_TAIL_LABELS = (0, 3, 4)  # drink, menu, outside
DEFAULT_CLASS_LABELS = (0, 1, 2, 3, 4)  # drink, food, inside, menu, outside
DEFAULT_MENU_LABEL = 3
ATTNRES_STREAMS = ("all", "visual", "text")


def _labels_mask(labels: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in ids:
        mask = mask | (labels == int(label_id))
    return mask


def _as_tuple(values: Optional[Sequence[int]], default: Sequence[int]):
    if values is None or len(values) == 0:
        return tuple(int(x) for x in default)
    return tuple(int(x) for x in values)


def routing_targets_from_labels(
    labels: torch.Tensor,
    num_experts: int,
    head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
    tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
    head_weights: Sequence[float] = (0.5, 0.5, 0.0),
    tail_weights: Sequence[float] = (0.5, 0.0, 0.5),
    expert_layout: str = "head_tail",
    class_label_ids: Sequence[int] = DEFAULT_CLASS_LABELS,
    class_expert_weight: float = 0.6,
    menu_label_id: int = DEFAULT_MENU_LABEL,
    menu_expert_weight: float = 0.7,
    dtype=None,
) -> torch.Tensor:
    labels = labels.long().view(-1)
    dtype = dtype or torch.float32

    if expert_layout == "shared_class":
        class_label_ids = _as_tuple(class_label_ids, DEFAULT_CLASS_LABELS)
        expected = 1 + len(class_label_ids)
        if num_experts != expected:
            raise ValueError(f"shared_class routing expects {expected} experts, got {num_experts}")

        target = torch.zeros(labels.shape[0], num_experts, device=labels.device, dtype=dtype)
        target[:, 0] = 1.0
        for expert_offset, label_id in enumerate(class_label_ids, start=1):
            mask = labels == int(label_id)
            if not mask.any():
                continue
            expert_weight = float(menu_expert_weight) if int(label_id) == int(menu_label_id) else float(class_expert_weight)
            expert_weight = max(0.0, min(1.0, expert_weight))
            target[mask, 0] = 1.0 - expert_weight
            target[mask, expert_offset] = expert_weight
        return target

    if expert_layout != "head_tail":
        raise ValueError(f"Unsupported expert_layout: {expert_layout}")
    if num_experts != 3:
        raise ValueError("head_tail routing target supervision expects 3 experts")

    target = torch.zeros(labels.shape[0], num_experts, device=labels.device, dtype=dtype)
    head = torch.tensor(head_weights, device=labels.device, dtype=dtype)
    tail = torch.tensor(tail_weights, device=labels.device, dtype=dtype)
    head_mask = _labels_mask(labels, head_label_ids)
    tail_mask = _labels_mask(labels, tail_label_ids)
    target[head_mask] = head
    target[tail_mask] = tail
    unknown_mask = ~(head_mask | tail_mask)
    if unknown_mask.any():
        target[unknown_mask, 0] = 1.0
    return target


class VLMTokenContext:
    def __init__(self, image_token_ids: Optional[Sequence[int]] = None):
        self.image_token_ids = tuple(int(x) for x in (image_token_ids or ()))
        self.input_ids = None
        self.attention_mask = None

    def set(self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None):
        self.input_ids = input_ids.detach() if input_ids is not None else None
        self.attention_mask = attention_mask.detach() if attention_mask is not None else None

    def clear(self):
        self.input_ids = None
        self.attention_mask = None

    def masks(self, batch_size: int, seq_len: int, device):
        if self.input_ids is None or self.input_ids.dim() != 2:
            return None
        if self.input_ids.shape[0] != batch_size or self.input_ids.shape[1] != seq_len:
            return None

        input_ids = self.input_ids.to(device=device)
        if self.attention_mask is not None and self.attention_mask.shape == self.input_ids.shape:
            all_mask = self.attention_mask.to(device=device).bool()
        else:
            all_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

        image_mask = torch.zeros_like(all_mask, dtype=torch.bool, device=device)
        for token_id in self.image_token_ids:
            image_mask = image_mask | (input_ids == int(token_id))
        image_mask = image_mask & all_mask
        text_mask = all_mask & ~image_mask
        return {
            "all": all_mask,
            "visual": image_mask if image_mask.any() else None,
            "text": text_mask if text_mask.any() else None,
        }


class VLMTMRDepthMemory:
    """Per-forward block summaries used for legacy depth-aware TMR routing."""

    def __init__(self, max_blocks: int = 8):
        self.max_blocks = int(max_blocks)
        self.reset()

    def reset(self, *args, **kwargs):
        self.block_sums = {}
        self.block_counts = {}

    def previous(self, feature_dim: int, block_id: int, batch_size: int):
        block_ids = sorted(
            bid for dim, bid in self.block_sums.keys()
            if dim == int(feature_dim) and bid < int(block_id)
        )
        if self.max_blocks > 0:
            block_ids = block_ids[-self.max_blocks:]

        summaries = []
        for bid in block_ids:
            key = (int(feature_dim), int(bid))
            summary = self.block_sums[key] / float(self.block_counts[key])
            if summary.shape[0] == batch_size:
                summaries.append(summary)
        if not summaries:
            return None
        return torch.stack(summaries, dim=1)

    def update(self, feature_dim: int, block_id: int, features: torch.Tensor):
        key = (int(feature_dim), int(block_id))
        features = features.detach()
        if key in self.block_sums and self.block_sums[key].shape == features.shape:
            self.block_sums[key] = self.block_sums[key] + features
            self.block_counts[key] += 1
        else:
            self.block_sums[key] = features
            self.block_counts[key] = 1


class VLMTMRAttnResMemory:
    """Block-level, modality-aware summaries for AttnRes-style TMR routing."""

    def __init__(self, max_blocks: int = 8, image_token_ids: Optional[Sequence[int]] = None):
        self.max_blocks = int(max_blocks)
        self.token_context = VLMTokenContext(image_token_ids=image_token_ids)
        self.reset()

    def reset(self, *args, **kwargs):
        self.block_sums = {}
        self.block_counts = {}

    def set_token_context(self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None):
        self.token_context.set(input_ids, attention_mask)

    def clear_token_context(self):
        self.token_context.clear()

    def token_masks(self, batch_size: int, seq_len: int, device):
        return self.token_context.masks(batch_size, seq_len, device)

    def previous(self, feature_dim: int, block_id: int, batch_size: int, stream: str):
        block_ids = sorted(
            bid for dim, bid, stored_stream in self.block_sums.keys()
            if dim == int(feature_dim) and stored_stream == stream and bid < int(block_id)
        )
        if self.max_blocks > 0:
            block_ids = block_ids[-self.max_blocks:]

        summaries = []
        for bid in block_ids:
            key = (int(feature_dim), int(bid), stream)
            summary = self.block_sums[key] / float(self.block_counts[key])
            if summary.shape[0] == batch_size:
                summaries.append(summary)
        if not summaries:
            return None
        return torch.stack(summaries, dim=1)

    def update(self, feature_dim: int, block_id: int, summaries: Dict[str, torch.Tensor]):
        for stream, features in summaries.items():
            if features is None:
                continue
            key = (int(feature_dim), int(block_id), stream)
            features = features.detach()
            if key in self.block_sums and self.block_sums[key].shape == features.shape:
                self.block_sums[key] = self.block_sums[key] + features
                self.block_counts[key] += 1
            else:
                self.block_sums[key] = features
                self.block_counts[key] = 1


def attach_vlm_tmr_depth_memory(model: nn.Module, memory: VLMTMRDepthMemory):
    old_handle = getattr(model, "_vlm_tmr_depth_memory_hook", None)
    if old_handle is not None:
        old_handle.remove()
    handle = model.register_forward_pre_hook(lambda module, inputs: memory.reset())
    model._vlm_tmr_depth_memory = memory
    model._vlm_tmr_depth_memory_hook = handle


def attach_vlm_tmr_attnres_memory(model: nn.Module, memory: VLMTMRAttnResMemory):
    old_handle = getattr(model, "_vlm_tmr_attnres_memory_hook", None)
    if old_handle is not None:
        old_handle.remove()
    handle = model.register_forward_pre_hook(lambda module, inputs: memory.reset())
    model._vlm_tmr_attnres_memory = memory
    model._vlm_tmr_attnres_memory_hook = handle


def set_vlm_tmr_token_context(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
):
    memory = getattr(model, "_vlm_tmr_attnres_memory", None)
    if memory is not None:
        memory.set_token_context(input_ids, attention_mask)


def clear_vlm_tmr_token_context(model: nn.Module):
    memory = getattr(model, "_vlm_tmr_attnres_memory", None)
    if memory is not None:
        memory.clear_token_context()


class VLMTMRLoRALinear(nn.Module):
    """Frozen linear layer plus multi-expert LoRA residual for VLM tuning."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        num_experts: int = 3,
        routing: str = "learned",
        head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
        tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
        oracle_head_weights: Sequence[float] = (0.5, 0.5, 0.0),
        oracle_tail_weights: Sequence[float] = (0.5, 0.0, 0.5),
        oracle_fallback: str = "uniform",
        expert_layout: str = "head_tail",
        class_label_ids: Sequence[int] = DEFAULT_CLASS_LABELS,
        class_expert_weight: float = 0.6,
        menu_label_id: int = DEFAULT_MENU_LABEL,
        menu_expert_weight: float = 0.7,
        depth_attention: bool = False,
        depth_memory: Optional[VLMTMRDepthMemory] = None,
        depth_block_id: int = 0,
        depth_context_scale: float = 0.1,
        attnres_mode: str = "none",
        attnres_memory: Optional[VLMTMRAttnResMemory] = None,
        attnres_block_id: int = 0,
        attnres_context_scale: float = 0.03,
        attnres_residual_scale: float = 0.01,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if expert_layout == "shared_class":
            expected = 1 + len(_as_tuple(class_label_ids, DEFAULT_CLASS_LABELS))
            if num_experts != expected:
                raise ValueError(f"shared_class layout expects {expected} experts, got {num_experts}")
        if attnres_mode not in {"none", "router", "full"}:
            raise ValueError(f"Unsupported attnres_mode: {attnres_mode}")

        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.num_experts = num_experts
        self.routing = routing
        self.oracle_fallback = oracle_fallback
        self.expert_layout = expert_layout
        self.head_label_ids = tuple(int(x) for x in head_label_ids)
        self.tail_label_ids = tuple(int(x) for x in tail_label_ids)
        self.class_label_ids = _as_tuple(class_label_ids, DEFAULT_CLASS_LABELS)
        self.class_expert_weight = float(class_expert_weight)
        self.menu_label_id = int(menu_label_id)
        self.menu_expert_weight = float(menu_expert_weight)

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(num_experts, self.in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, rank, self.out_features, device=device, dtype=dtype))
        for expert_idx in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[expert_idx], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[expert_idx])

        self.router = nn.Linear(self.in_features, num_experts, bias=True, device=device, dtype=dtype)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.register_buffer(
            "oracle_head_weights",
            torch.tensor(oracle_head_weights, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "oracle_tail_weights",
            torch.tensor(oracle_tail_weights, device=device, dtype=dtype),
            persistent=False,
        )
        self.current_class_labels = None
        self.last_routing_weights = None
        self.last_depth_weights = None
        self.last_attnres_weights = {}
        self._last_attnres_contexts = {}

        self.depth_attention = bool(depth_attention)
        self.depth_memory = depth_memory
        self.depth_block_id = int(depth_block_id)
        if self.depth_attention:
            self.depth_router = nn.Linear(self.in_features, num_experts, bias=False, device=device, dtype=dtype)
            nn.init.zeros_(self.depth_router.weight)
            self.depth_context_scale = nn.Parameter(
                torch.tensor(float(depth_context_scale), device=device, dtype=dtype)
            )
        else:
            self.depth_router = None
            self.depth_context_scale = None

        self.attnres_mode = attnres_mode
        self.attnres_memory = attnres_memory
        self.attnres_block_id = int(attnres_block_id)
        self.attnres_enabled = attnres_mode != "none"
        if self.attnres_enabled:
            self.attnres_routers = nn.ModuleDict(
                {
                    stream: nn.Linear(self.in_features, num_experts, bias=False, device=device, dtype=dtype)
                    for stream in ATTNRES_STREAMS
                }
            )
            for router in self.attnres_routers.values():
                nn.init.zeros_(router.weight)
            self.attnres_context_scale = nn.Parameter(
                torch.tensor(float(attnres_context_scale), device=device, dtype=dtype)
            )
        else:
            self.attnres_routers = nn.ModuleDict()
            self.attnres_context_scale = None

        if self.attnres_mode == "full":
            self.attnres_A = nn.Parameter(torch.zeros(self.in_features, rank, device=device, dtype=dtype))
            self.attnres_B = nn.Parameter(torch.zeros(rank, self.out_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.attnres_A, a=math.sqrt(5))
            nn.init.zeros_(self.attnres_B)
            self.attnres_residual_gate = nn.Linear(self.in_features, 1, bias=True, device=device, dtype=dtype)
            nn.init.zeros_(self.attnres_residual_gate.weight)
            nn.init.zeros_(self.attnres_residual_gate.bias)
            self.attnres_residual_scale = nn.Parameter(
                torch.tensor(float(attnres_residual_scale), device=device, dtype=dtype)
            )
        else:
            self.attnres_A = None
            self.attnres_B = None
            self.attnres_residual_gate = None
            self.attnres_residual_scale = None

    def set_class_labels(self, labels: Optional[torch.Tensor]):
        self.current_class_labels = labels

    def _pool_with_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or x.dim() != 3:
            return x.detach().mean(dim=1) if x.dim() == 3 else x.detach()
        weights = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x.detach() * weights).sum(dim=1) / denom

    def _summaries_for_router(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            features = x.detach()
            return {stream: features for stream in ATTNRES_STREAMS}

        all_features = x.detach().mean(dim=1)
        summaries = {stream: all_features for stream in ATTNRES_STREAMS}
        if x.dim() != 3 or not self.attnres_enabled or self.attnres_memory is None:
            return summaries

        masks = self.attnres_memory.token_masks(x.shape[0], x.shape[1], x.device)
        if masks is None:
            return summaries
        for stream in ATTNRES_STREAMS:
            summaries[stream] = self._pool_with_mask(x, masks.get(stream))
        return summaries

    def _features_for_router(self, x: torch.Tensor) -> torch.Tensor:
        return self._summaries_for_router(x)["all"]

    def routing_from_labels(self, labels: torch.Tensor, device, dtype) -> torch.Tensor:
        return routing_targets_from_labels(
            labels.to(device=device),
            num_experts=self.num_experts,
            head_label_ids=self.head_label_ids,
            tail_label_ids=self.tail_label_ids,
            head_weights=self.oracle_head_weights.to(device=device, dtype=dtype).tolist(),
            tail_weights=self.oracle_tail_weights.to(device=device, dtype=dtype).tolist(),
            expert_layout=self.expert_layout,
            class_label_ids=self.class_label_ids,
            class_expert_weight=self.class_expert_weight,
            menu_label_id=self.menu_label_id,
            menu_expert_weight=self.menu_expert_weight,
            dtype=dtype,
        )

    def _fallback_routing(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.oracle_fallback in {"vanilla_lora", "shared"}:
            routing = torch.zeros(batch_size, self.num_experts, device=device, dtype=dtype)
            routing[:, 0] = 1.0
            return routing
        if self.oracle_fallback == "random":
            return torch.softmax(torch.rand(batch_size, self.num_experts, device=device, dtype=dtype), dim=-1)
        return torch.full((batch_size, self.num_experts), 1.0 / self.num_experts, device=device, dtype=dtype)

    def _rms_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _legacy_depth_context_from_memory(self, features: torch.Tensor):
        if not self.depth_attention or self.depth_memory is None:
            self.last_depth_weights = None
            return None
        memory = self.depth_memory.previous(self.in_features, self.depth_block_id, features.shape[0])
        if memory is None:
            self.last_depth_weights = None
            return None

        query = self._rms_norm(features)
        keys = self._rms_norm(memory)
        scores = torch.einsum("bd,bmd->bm", query, keys) / math.sqrt(float(self.in_features))
        weights = torch.softmax(scores, dim=-1).to(dtype=features.dtype)
        context = torch.einsum("bm,bmd->bd", weights, memory.to(dtype=features.dtype))
        self.last_depth_weights = weights
        return context

    def _attnres_context_from_memory(self, features: torch.Tensor, stream: str):
        if not self.attnres_enabled or self.attnres_memory is None:
            self.last_attnres_weights[stream] = None
            return None
        memory = self.attnres_memory.previous(self.in_features, self.attnres_block_id, features.shape[0], stream)
        if memory is None:
            self.last_attnres_weights[stream] = None
            return None

        query = self._rms_norm(features)
        keys = self._rms_norm(memory)
        scores = torch.einsum("bd,bmd->bm", query, keys) / math.sqrt(float(self.in_features))
        weights = torch.softmax(scores, dim=-1).to(dtype=features.dtype)
        context = torch.einsum("bm,bmd->bd", weights, memory.to(dtype=features.dtype))
        self.last_attnres_weights[stream] = weights
        return context

    def _update_depth_memory(self, features: torch.Tensor):
        if self.depth_attention and self.depth_memory is not None:
            self.depth_memory.update(self.in_features, self.depth_block_id, features)

    def _update_attnres_memory(self, summaries: Dict[str, torch.Tensor]):
        if self.attnres_enabled and self.attnres_memory is not None:
            self.attnres_memory.update(self.in_features, self.attnres_block_id, summaries)

    def make_routing(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        labels = self.current_class_labels
        summaries = None
        features_for_memory = None
        self._last_attnres_contexts = {}

        if self.routing in {"vanilla_lora", "shared"}:
            routing = torch.zeros(batch_size, self.num_experts, device=device, dtype=dtype)
            routing[:, 0] = 1.0
        elif self.routing == "uniform":
            routing = torch.full((batch_size, self.num_experts), 1.0 / self.num_experts, device=device, dtype=dtype)
        elif self.routing == "random":
            routing = torch.softmax(torch.rand(batch_size, self.num_experts, device=device, dtype=dtype), dim=-1)
        elif self.routing == "oracle":
            if labels is None or labels.numel() != batch_size:
                routing = self._fallback_routing(batch_size, device, dtype)
            else:
                routing = self.routing_from_labels(labels, device=device, dtype=dtype)
        elif self.routing == "learned":
            summaries = self._summaries_for_router(x)
            features = summaries["all"].to(dtype=self.router.weight.dtype)
            features_for_memory = features
            logits = self.router(features)

            if self.attnres_enabled:
                scale = self.attnres_context_scale.to(dtype=logits.dtype)
                for stream in ATTNRES_STREAMS:
                    stream_features = summaries[stream].to(dtype=self.router.weight.dtype)
                    context = self._attnres_context_from_memory(stream_features, stream)
                    if context is None:
                        continue
                    self._last_attnres_contexts[stream] = context
                    router = self.attnres_routers[stream]
                    depth_logits = router(context.to(dtype=router.weight.dtype))
                    logits = logits + scale * depth_logits.to(dtype=logits.dtype)
            else:
                depth_context = self._legacy_depth_context_from_memory(features)
                if depth_context is not None:
                    depth_logits = self.depth_router(depth_context.to(dtype=self.depth_router.weight.dtype))
                    scale = self.depth_context_scale.to(dtype=logits.dtype)
                    logits = logits + scale * depth_logits.to(dtype=logits.dtype)

            routing = torch.softmax(logits, dim=-1).to(dtype=dtype)
        else:
            raise ValueError(f"Unsupported VLM-TMR routing mode: {self.routing}")

        if self.attnres_enabled:
            if summaries is None:
                summaries = self._summaries_for_router(x)
            self._update_attnres_memory({k: v.to(dtype=self.router.weight.dtype) for k, v in summaries.items()})
        elif self.depth_attention:
            if features_for_memory is None:
                features_for_memory = self._features_for_router(x).to(dtype=self.router.weight.dtype)
            self._update_depth_memory(features_for_memory)

        if routing.shape != (batch_size, self.num_experts):
            raise ValueError(f"routing must have shape {(batch_size, self.num_experts)}, got {tuple(routing.shape)}")
        self.last_routing_weights = routing
        return routing

    def _attnres_residual(self, x_drop: torch.Tensor):
        if self.attnres_mode != "full" or "all" not in self._last_attnres_contexts:
            return None
        context = self._last_attnres_contexts["all"]
        depth_delta = (x_drop @ self.attnres_A) @ self.attnres_B
        gate = torch.sigmoid(self.attnres_residual_gate(context.to(dtype=self.attnres_residual_gate.weight.dtype)))
        gate = gate.to(dtype=depth_delta.dtype)
        view_shape = [gate.shape[0]] + [1] * (depth_delta.dim() - 1)
        scale = self.attnres_residual_scale.to(dtype=depth_delta.dtype)
        return depth_delta * gate.view(*view_shape) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_drop = self.dropout(x)
        routing = self.make_routing(x_drop)

        delta = None
        for expert_idx in range(self.num_experts):
            expert_delta = (x_drop @ self.lora_A[expert_idx]) @ self.lora_B[expert_idx]
            weight = routing[:, expert_idx]
            view_shape = [routing.shape[0]] + [1] * (expert_delta.dim() - 1)
            expert_delta = expert_delta * weight.view(*view_shape)
            delta = expert_delta if delta is None else delta + expert_delta

        attnres_delta = self._attnres_residual(x_drop)
        if attnres_delta is not None:
            delta = delta + attnres_delta
        return base_out + delta * self.scaling


class VLMAttnResLoRALinear(nn.Module):
    """No-gate AttnRes-LoRA: local LoRA plus cross-layer attention residual."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        attnres_memory: Optional[VLMTMRAttnResMemory] = None,
        attnres_block_id: int = 0,
        attnres_scale: float = 0.03,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.attnres_memory = attnres_memory
        self.attnres_block_id = int(attnres_block_id)
        self.last_attnres_weights = {}
        self._last_attnres_contexts = {}

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.attnres_A = nn.ParameterDict()
        self.attnres_B = nn.ParameterDict()
        for stream in ATTNRES_STREAMS:
            a = nn.Parameter(torch.zeros(self.in_features, rank, device=device, dtype=dtype))
            b = nn.Parameter(torch.zeros(rank, self.out_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.attnres_A[stream] = a
            self.attnres_B[stream] = b
        self.attnres_scale = nn.Parameter(torch.tensor(float(attnres_scale), device=device, dtype=dtype))

    def _pool_with_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or x.dim() != 3:
            return x.detach().mean(dim=1) if x.dim() == 3 else x.detach()
        weights = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x.detach() * weights).sum(dim=1) / denom

    def _summaries(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            features = x.detach()
            return {stream: features for stream in ATTNRES_STREAMS}

        all_features = x.detach().mean(dim=1)
        summaries = {stream: all_features for stream in ATTNRES_STREAMS}
        if x.dim() != 3 or self.attnres_memory is None:
            return summaries

        masks = self.attnres_memory.token_masks(x.shape[0], x.shape[1], x.device)
        if masks is None:
            return summaries
        for stream in ATTNRES_STREAMS:
            summaries[stream] = self._pool_with_mask(x, masks.get(stream))
        return summaries

    def _rms_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _context_from_memory(self, features: torch.Tensor, stream: str):
        if self.attnres_memory is None:
            self.last_attnres_weights[stream] = None
            return None
        memory = self.attnres_memory.previous(self.in_features, self.attnres_block_id, features.shape[0], stream)
        if memory is None:
            self.last_attnres_weights[stream] = None
            return None
        query = self._rms_norm(features)
        keys = self._rms_norm(memory)
        scores = torch.einsum("bd,bmd->bm", query, keys) / math.sqrt(float(self.in_features))
        weights = torch.softmax(scores, dim=-1).to(dtype=features.dtype)
        context = torch.einsum("bm,bmd->bd", weights, memory.to(dtype=features.dtype))
        self.last_attnres_weights[stream] = weights
        return context

    def _update_memory(self, summaries: Dict[str, torch.Tensor]):
        if self.attnres_memory is not None:
            self.attnres_memory.update(self.in_features, self.attnres_block_id, summaries)

    def _expand_context(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return context.to(dtype=x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
        return context.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_drop = self.dropout(x)
        summaries = self._summaries(x_drop)
        self._last_attnres_contexts = {}

        local_delta = (x_drop @ self.lora_A) @ self.lora_B
        attnres_delta = None
        n_contexts = 0
        for stream in ATTNRES_STREAMS:
            features = summaries[stream].to(dtype=self.attnres_A[stream].dtype)
            context = self._context_from_memory(features, stream)
            if context is None:
                continue
            self._last_attnres_contexts[stream] = context
            context_tokens = self._expand_context(context, x_drop)
            # The residual uses attention-selected previous-block context directly,
            # rather than choosing among experts or applying class-conditioned gates.
            stream_delta = ((x_drop + context_tokens) @ self.attnres_A[stream]) @ self.attnres_B[stream]
            attnres_delta = stream_delta if attnres_delta is None else attnres_delta + stream_delta
            n_contexts += 1

        self._update_memory({k: v.to(dtype=self.lora_A.dtype) for k, v in summaries.items()})

        delta = local_delta
        if attnres_delta is not None and n_contexts > 0:
            delta = delta + self.attnres_scale.to(dtype=delta.dtype) * (attnres_delta / float(n_contexts))
        return base_out + delta * self.scaling


def iter_vlm_attnres_lora_layers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, VLMAttnResLoRALinear):
            yield module


def inject_vlm_attnres_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    attnres_block_size: int = 7,
    attnres_max_blocks: int = 8,
    attnres_scale: float = 0.03,
    image_token_ids: Optional[Sequence[int]] = None,
) -> int:
    target_set = set(target_modules)
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in target_set and name not in target_set:
            continue
        replacements.append((name, module))

    memory = VLMTMRAttnResMemory(attnres_max_blocks, image_token_ids=image_token_ids)
    attnres_block_size = max(1, int(attnres_block_size))
    for replacement_idx, (name, linear) in enumerate(replacements):
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            VLMAttnResLoRALinear(
                linear,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                attnres_memory=memory,
                attnres_block_id=replacement_idx // attnres_block_size,
                attnres_scale=attnres_scale,
            ),
        )
    if replacements:
        attach_vlm_tmr_attnres_memory(model, memory)
    return len(replacements)


def set_only_vlm_attnres_trainable(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in iter_vlm_attnres_lora_layers(model):
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
        for param in layer.attnres_A.parameters():
            param.requires_grad_(True)
        for param in layer.attnres_B.parameters():
            param.requires_grad_(True)
        layer.attnres_scale.requires_grad_(True)


def get_vlm_attnres_lora_state_dict(model: nn.Module):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, VLMAttnResLoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            for stream in ATTNRES_STREAMS:
                state[f"{name}.attnres_A.{stream}"] = module.attnres_A[stream].detach().cpu()
                state[f"{name}.attnres_B.{stream}"] = module.attnres_B[stream].detach().cpu()
            state[f"{name}.attnres_scale"] = module.attnres_scale.detach().cpu()
    return state


def save_vlm_attnres_lora_adapter(model: nn.Module, output_dir: str, config: dict):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(get_vlm_attnres_lora_state_dict(model), os.path.join(output_dir, "attnres_adapter.pt"))
    with open(os.path.join(output_dir, "attnres_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_vlm_attnres_lora_adapter(model: nn.Module, adapter_dir: str, strict: bool = True):
    adapter_path = os.path.join(adapter_dir, "attnres_adapter.pt")
    state = torch.load(adapter_path, map_location="cpu")
    expected_keys = set(get_vlm_attnres_lora_state_dict(model).keys())
    loaded_keys = set(state.keys())
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"AttnRes-LoRA adapter mismatch for {adapter_dir}: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)
    return missing, unexpected


def iter_vlm_tmr_lora_layers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, VLMTMRLoRALinear):
            yield module


def set_vlm_tmr_class_labels(model: nn.Module, labels: Optional[torch.Tensor]):
    for layer in iter_vlm_tmr_lora_layers(model):
        layer.set_class_labels(labels)


def clear_vlm_tmr_class_labels(model: nn.Module):
    set_vlm_tmr_class_labels(model, None)


def collect_vlm_tmr_routing_loss(
    model: nn.Module,
    labels: torch.Tensor,
    head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
    tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
    head_weights: Sequence[float] = (0.5, 0.5, 0.0),
    tail_weights: Sequence[float] = (0.5, 0.0, 0.5),
    expert_layout: str = "head_tail",
    class_label_ids: Sequence[int] = DEFAULT_CLASS_LABELS,
    class_expert_weight: float = 0.6,
    menu_label_id: int = DEFAULT_MENU_LABEL,
    menu_expert_weight: float = 0.7,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
) -> Optional[torch.Tensor]:
    losses = []
    for layer in iter_vlm_tmr_lora_layers(model):
        routing = layer.last_routing_weights
        if routing is None or routing.requires_grad is False or labels.numel() != routing.shape[0]:
            continue
        target = routing_targets_from_labels(
            labels.to(device=routing.device),
            num_experts=layer.num_experts,
            head_label_ids=head_label_ids,
            tail_label_ids=tail_label_ids,
            head_weights=head_weights,
            tail_weights=tail_weights,
            expert_layout=expert_layout,
            class_label_ids=class_label_ids,
            class_expert_weight=class_expert_weight,
            menu_label_id=menu_label_id,
            menu_expert_weight=menu_expert_weight,
            dtype=routing.dtype,
        )
        loss = -(target * torch.log(routing.clamp_min(1e-6))).sum(dim=-1).mean()
        if entropy_weight > 0:
            entropy = -(routing * torch.log(routing.clamp_min(1e-6))).sum(dim=-1).mean()
            loss = loss + float(entropy_weight) * entropy
        if balance_weight > 0:
            loss = loss + float(balance_weight) * (routing.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
        losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def _get_parent_module(model: nn.Module, module_name: str):
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_vlm_tmr_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    num_experts: int = 3,
    routing: str = "learned",
    head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
    tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
    oracle_head_weights: Sequence[float] = (0.5, 0.5, 0.0),
    oracle_tail_weights: Sequence[float] = (0.5, 0.0, 0.5),
    oracle_fallback: str = "uniform",
    expert_layout: str = "head_tail",
    class_label_ids: Sequence[int] = DEFAULT_CLASS_LABELS,
    class_expert_weight: float = 0.6,
    menu_label_id: int = DEFAULT_MENU_LABEL,
    menu_expert_weight: float = 0.7,
    depth_attention: bool = False,
    depth_block_size: int = 8,
    depth_max_blocks: int = 8,
    depth_context_scale: float = 0.1,
    attnres_mode: str = "none",
    attnres_block_size: int = 7,
    attnres_max_blocks: int = 8,
    attnres_context_scale: float = 0.03,
    attnres_residual_scale: float = 0.01,
    image_token_ids: Optional[Sequence[int]] = None,
) -> int:
    class_label_ids = _as_tuple(class_label_ids, DEFAULT_CLASS_LABELS)
    if expert_layout == "shared_class":
        num_experts = 1 + len(class_label_ids)

    target_set = set(target_modules)
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in target_set and name not in target_set:
            continue
        replacements.append((name, module))

    depth_memory = VLMTMRDepthMemory(depth_max_blocks) if depth_attention and attnres_mode == "none" else None
    attnres_memory = (
        VLMTMRAttnResMemory(attnres_max_blocks, image_token_ids=image_token_ids)
        if attnres_mode != "none"
        else None
    )
    depth_block_size = max(1, int(depth_block_size))
    attnres_block_size = max(1, int(attnres_block_size))

    for replacement_idx, (name, linear) in enumerate(replacements):
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            VLMTMRLoRALinear(
                linear,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_experts=num_experts,
                routing=routing,
                head_label_ids=head_label_ids,
                tail_label_ids=tail_label_ids,
                oracle_head_weights=oracle_head_weights,
                oracle_tail_weights=oracle_tail_weights,
                oracle_fallback=oracle_fallback,
                expert_layout=expert_layout,
                class_label_ids=class_label_ids,
                class_expert_weight=class_expert_weight,
                menu_label_id=menu_label_id,
                menu_expert_weight=menu_expert_weight,
                depth_attention=depth_attention and attnres_mode == "none",
                depth_memory=depth_memory,
                depth_block_id=replacement_idx // depth_block_size,
                depth_context_scale=depth_context_scale,
                attnres_mode=attnres_mode,
                attnres_memory=attnres_memory,
                attnres_block_id=replacement_idx // attnres_block_size,
                attnres_context_scale=attnres_context_scale,
                attnres_residual_scale=attnres_residual_scale,
            ),
        )
    if depth_memory is not None:
        attach_vlm_tmr_depth_memory(model, depth_memory)
    if attnres_memory is not None:
        attach_vlm_tmr_attnres_memory(model, attnres_memory)
    return len(replacements)


def trainable_parameter_summary(model: nn.Module):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return total, trainable, pct


def set_only_vlm_tmr_trainable(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in iter_vlm_tmr_lora_layers(model):
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
        if layer.routing == "learned":
            for param in layer.router.parameters():
                param.requires_grad_(True)
            if layer.depth_attention:
                for param in layer.depth_router.parameters():
                    param.requires_grad_(True)
                layer.depth_context_scale.requires_grad_(True)
            if layer.attnres_enabled:
                for router in layer.attnres_routers.values():
                    for param in router.parameters():
                        param.requires_grad_(True)
                layer.attnres_context_scale.requires_grad_(True)
                if layer.attnres_mode == "full":
                    layer.attnres_A.requires_grad_(True)
                    layer.attnres_B.requires_grad_(True)
                    for param in layer.attnres_residual_gate.parameters():
                        param.requires_grad_(True)
                    layer.attnres_residual_scale.requires_grad_(True)


def get_vlm_tmr_lora_state_dict(model: nn.Module):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, VLMTMRLoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.router.weight"] = module.router.weight.detach().cpu()
            state[f"{name}.router.bias"] = module.router.bias.detach().cpu()
            if module.depth_attention:
                state[f"{name}.depth_router.weight"] = module.depth_router.weight.detach().cpu()
                state[f"{name}.depth_context_scale"] = module.depth_context_scale.detach().cpu()
            if module.attnres_enabled:
                for stream, router in module.attnres_routers.items():
                    state[f"{name}.attnres_routers.{stream}.weight"] = router.weight.detach().cpu()
                state[f"{name}.attnres_context_scale"] = module.attnres_context_scale.detach().cpu()
                if module.attnres_mode == "full":
                    state[f"{name}.attnres_A"] = module.attnres_A.detach().cpu()
                    state[f"{name}.attnres_B"] = module.attnres_B.detach().cpu()
                    state[f"{name}.attnres_residual_gate.weight"] = module.attnres_residual_gate.weight.detach().cpu()
                    state[f"{name}.attnres_residual_gate.bias"] = module.attnres_residual_gate.bias.detach().cpu()
                    state[f"{name}.attnres_residual_scale"] = module.attnres_residual_scale.detach().cpu()
    return state


def save_vlm_tmr_lora_adapter(model: nn.Module, output_dir: str, config: dict):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(get_vlm_tmr_lora_state_dict(model), os.path.join(output_dir, "tmr_adapter.pt"))
    with open(os.path.join(output_dir, "tmr_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_vlm_tmr_lora_adapter(model: nn.Module, adapter_dir: str, strict: bool = True):
    adapter_path = os.path.join(adapter_dir, "tmr_adapter.pt")
    state = torch.load(adapter_path, map_location="cpu")
    expected_keys = set(get_vlm_tmr_lora_state_dict(model).keys())
    loaded_keys = set(state.keys())
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"TMR adapter mismatch for {adapter_dir}: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.load_state_dict(state, strict=False)
    return missing, unexpected


def class_balanced_sample_weights(rows):
    counts = Counter(int(row["label"]) for row in rows)
    return [1.0 / counts[int(row["label"])] for row in rows]
