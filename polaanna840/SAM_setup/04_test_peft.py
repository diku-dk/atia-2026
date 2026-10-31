
import torch

from models.lora import build_lora
from models.adapter import build_adapter


device = "cuda" if torch.cuda.is_available() else "cpu"


def inspect_model(model, adapted_layers, method):

    print("\n" + "=" * 70)
    print(method)
    print("=" * 70)

    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Adapted locations: {len(adapted_layers)}")

    for name in adapted_layers:
        print(" ", name)

    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")

    trainable_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad
    ]

    print("\nTrainable parameter names:")

    for name in trainable_names:
        print(" ", name)

    return trainable_names


# LORA
lora_model, lora_layers = build_lora(
    rank=4,
    alpha=4,
    device=device
)

lora_names = inspect_model(
    lora_model,
    lora_layers,
    "LORA"
)

assert len(lora_layers) == 10
assert all(
    "lora_A" in name or "lora_B" in name
    for name in lora_names
)


# ADAPTER
adapter_model, adapter_layers = build_adapter(
    bottleneck_dim=16,
    device=device
)
adapter_names = inspect_model(
    adapter_model,
    adapter_layers,
    "ADAPTER"
)

assert len(adapter_layers) == 10
assert all(
    ".adapter." in name
    for name in adapter_names
)


# CHECK INITIAL MODIFICATIONS
print("\n" + "=" * 70)
print("INITIALIZATION CHECKS")
print("=" * 70)

with torch.no_grad():

    # LoRA: B is initialized to zero, so its update is zero.
    for module in lora_model.modules():
        if hasattr(module, "lora_B"):
            assert torch.count_nonzero(
                module.lora_B.weight
            ).item() == 0

    # Adapter: up-projection is initialized to zero.
    for module in adapter_model.modules():
        if hasattr(module, "adapter") and hasattr(module.adapter, "up"):
            assert torch.count_nonzero(
                module.adapter.up.weight
            ).item() == 0

            assert torch.count_nonzero(
                module.adapter.up.bias
            ).item() == 0

print("LoRA and Adapter initial updates are zero.")
print("Trainable parameter checks passed.")

