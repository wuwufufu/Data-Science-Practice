import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HEAD_LABELS = (1, 2)  # food, inside in datasets/Yelp/classnames.txt
DEFAULT_TAIL_LABELS = (0, 3, 4)  # drink, menu, outside


class TMRLoRA(nn.Module):
    """Multi-expert LoRA residual used by TMR-LoRA-v1.

    The module returns only the low-rank residual, matching the existing
    LoRA class in peft_modules.py. Existing frozen linear projections are
    applied by the caller.
    """

    def __init__(
        self,
        in_dim: int,
        bottle_dim: int,
        out_dim: Optional[int] = None,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        num_experts: int = 3,
        routing: str = "uniform",
        dtype=None,
        head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
        tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
        oracle_head_weights: Sequence[float] = (0.5, 0.5, 0.0),
        oracle_tail_weights: Sequence[float] = (0.5, 0.0, 0.5),
        batch_dim: int = 0,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if bottle_dim < 1:
            raise ValueError("bottle_dim/rank must be >= 1")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bottle_dim = bottle_dim
        self.num_experts = num_experts
        self.routing = routing
        self.batch_dim = batch_dim
        self.scaling = float(alpha if alpha is not None else 1.0) / float(bottle_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head_label_ids = tuple(int(x) for x in head_label_ids)
        self.tail_label_ids = tuple(int(x) for x in tail_label_ids)

        self.lora_A = nn.Parameter(torch.zeros(num_experts, in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, bottle_dim, out_dim, dtype=dtype))
        for expert_idx in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[expert_idx], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[expert_idx])

        self.router = nn.Linear(in_dim, num_experts, dtype=dtype)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.register_buffer("oracle_head_weights", torch.tensor(oracle_head_weights, dtype=dtype or torch.float32))
        self.register_buffer("oracle_tail_weights", torch.tensor(oracle_tail_weights, dtype=dtype or torch.float32))
        self.last_routing_weights = None

    @property
    def dtype(self):
        return self.lora_A.dtype

    def _batch_size(self, x: torch.Tensor) -> int:
        batch_dim = self.batch_dim if self.batch_dim >= 0 else x.dim() + self.batch_dim
        return x.shape[batch_dim]

    def _features_for_router(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        batch_dim = self.batch_dim if self.batch_dim >= 0 else x.dim() + self.batch_dim
        reduce_dims = [dim for dim in range(x.dim() - 1) if dim != batch_dim]
        feat = x
        for dim in sorted(reduce_dims, reverse=True):
            feat = feat.mean(dim=dim)
        return feat

    def _labels_mask(self, labels: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for label_id in ids:
            mask = mask | (labels == int(label_id))
        return mask

    def routing_from_labels(self, labels: torch.Tensor, device, dtype) -> torch.Tensor:
        labels = labels.to(device=device).long().view(-1)
        routing = torch.zeros(labels.shape[0], self.num_experts, device=device, dtype=dtype)
        head_weights = self.oracle_head_weights.to(device=device, dtype=dtype)
        tail_weights = self.oracle_tail_weights.to(device=device, dtype=dtype)
        head_mask = self._labels_mask(labels, self.head_label_ids)
        tail_mask = self._labels_mask(labels, self.tail_label_ids)
        routing[head_mask] = head_weights
        routing[tail_mask] = tail_weights
        unknown_mask = ~(head_mask | tail_mask)
        if unknown_mask.any():
            routing[unknown_mask, 0] = 1.0
        return routing

    def make_routing(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = self._batch_size(x)
        device = x.device
        dtype = x.dtype
        if routing_weights is not None:
            routing = routing_weights.to(device=device, dtype=dtype)
        elif self.routing in {"vanilla_lora", "shared"}:
            routing = torch.zeros(batch_size, self.num_experts, device=device, dtype=dtype)
            routing[:, 0] = 1.0
        elif self.routing == "oracle":
            if labels is None:
                raise ValueError("oracle routing requires labels")
            routing = self.routing_from_labels(labels, device=device, dtype=dtype)
        elif self.routing == "learned":
            features = self._features_for_router(x).to(dtype=self.router.weight.dtype)
            routing = torch.softmax(self.router(features), dim=-1).to(dtype=dtype)
        elif self.routing == "uniform":
            routing = torch.full((batch_size, self.num_experts), 1.0 / self.num_experts, device=device, dtype=dtype)
        elif self.routing == "random":
            routing = torch.softmax(torch.rand(batch_size, self.num_experts, device=device, dtype=dtype), dim=-1)
        else:
            raise ValueError(f"Unsupported TMR routing mode: {self.routing}")

        if routing.shape != (batch_size, self.num_experts):
            raise ValueError(
                f"routing weights must have shape {(batch_size, self.num_experts)}, got {tuple(routing.shape)}"
            )
        self.last_routing_weights = routing.detach()
        return routing

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_drop = self.dropout(x)
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_outputs.append((x_drop @ self.lora_A[expert_idx]) @ self.lora_B[expert_idx])
        stacked = torch.stack(expert_outputs, dim=-2)

        routing = self.make_routing(x, labels=labels, routing_weights=routing_weights)
        batch_dim = self.batch_dim if self.batch_dim >= 0 else x.dim() + self.batch_dim
        expert_dim = stacked.dim() - 2
        view_shape = [1] * stacked.dim()
        view_shape[batch_dim] = routing.shape[0]
        view_shape[expert_dim] = self.num_experts
        view_shape[-1] = 1
        mixed = (stacked * routing.view(*view_shape)).sum(dim=expert_dim)
        return mixed * self.scaling


class TMRLoRALinear(nn.Module):
    """Frozen linear layer plus TMR-LoRA residual, useful for tests/prototypes."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        routing: str = "uniform",
        num_experts: int = 3,
        head_label_ids: Sequence[int] = DEFAULT_HEAD_LABELS,
        tail_label_ids: Sequence[int] = DEFAULT_TAIL_LABELS,
    ):
        super().__init__()
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        self.tmr_lora = TMRLoRA(
            in_dim=base_linear.in_features,
            bottle_dim=rank,
            out_dim=base_linear.out_features,
            alpha=alpha,
            dropout=dropout,
            num_experts=num_experts,
            routing=routing,
            dtype=base_linear.weight.dtype,
            head_label_ids=head_label_ids,
            tail_label_ids=tail_label_ids,
            batch_dim=0,
        )

    def forward(self, x, labels=None, routing_weights=None):
        base = F.linear(x, self.weight, self.bias)
        return base + self.tmr_lora(x, labels=labels, routing_weights=routing_weights)


def iter_tmr_lora_modules(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, TMRLoRA):
            yield submodule
