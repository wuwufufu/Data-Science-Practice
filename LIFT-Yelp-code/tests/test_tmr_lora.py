import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

module_path = Path(__file__).resolve().parents[1] / "models" / "tmr_lora.py"
spec = importlib.util.spec_from_file_location("tmr_lora", module_path)
tmr_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmr_lora)
TMRLoRALinear = tmr_lora.TMRLoRALinear


def test_zero_lora_matches_base_linear():
    torch.manual_seed(0)
    base = nn.Linear(5, 4)
    layer = TMRLoRALinear(base, rank=2, alpha=2.0, routing="uniform")
    with torch.no_grad():
        layer.tmr_lora.lora_A.zero_()
        layer.tmr_lora.lora_B.zero_()
    x = torch.randn(3, 5)
    expected = base(x)
    actual = layer(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_one_hot_routing_activates_only_requested_expert():
    torch.manual_seed(1)
    base = nn.Linear(5, 4)
    layer = TMRLoRALinear(base, rank=2, alpha=2.0, routing="uniform")
    x = torch.randn(3, 5)

    with torch.no_grad():
        for expert_idx in range(3):
            layer.tmr_lora.lora_A[expert_idx].fill_(0.1 * (expert_idx + 1))
            layer.tmr_lora.lora_B[expert_idx].fill_(0.2 * (expert_idx + 1))

    for expert_idx in range(3):
        routing = torch.zeros(x.shape[0], 3)
        routing[:, expert_idx] = 1.0
        actual_delta = layer.tmr_lora(x, routing_weights=routing)
        expected_delta = (
            (x @ layer.tmr_lora.lora_A[expert_idx])
            @ layer.tmr_lora.lora_B[expert_idx]
        ) * layer.tmr_lora.scaling
        assert torch.allclose(actual_delta, expected_delta, atol=1e-6)


def test_backward_freezes_base_and_updates_lora():
    torch.manual_seed(2)
    base = nn.Linear(5, 4)
    layer = TMRLoRALinear(base, rank=2, alpha=2.0, routing="uniform")
    with torch.no_grad():
        layer.tmr_lora.lora_A.normal_(0, 0.02)
        layer.tmr_lora.lora_B.normal_(0, 0.02)

    x = torch.randn(6, 5)
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    out = layer(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    assert layer.weight.grad is None
    assert layer.bias.grad is None
    assert layer.tmr_lora.lora_A.grad is not None
    assert layer.tmr_lora.lora_B.grad is not None
    assert layer.tmr_lora.lora_A.grad.abs().sum().item() > 0
    assert layer.tmr_lora.lora_B.grad.abs().sum().item() > 0


if __name__ == "__main__":
    test_zero_lora_matches_base_linear()
    test_one_hot_routing_activates_only_requested_expert()
    test_backward_freezes_base_and_updates_lora()
    print("TMR-LoRA sanity checks passed.")
