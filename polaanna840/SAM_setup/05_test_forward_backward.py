
"""

TEST A: ZERO-SHOT EQUIVALENCE
Check whether the original MobileSAM, LoRA MobileSAM,
and Adapter MobileSAM produce identical image encoder
outputs before adaptation training.

TEST B: GRADIENT FLOW
Check whether the newly introduced PEFT parameters
can receive nonzero gradients while pretrained
parameters remain frozen.

Both tests use synthetic inputs.
 NuInsSeg data is not used.

"""

import torch
import torch.nn.functional as F
from models.base import load_mobile_sam
from models.lora import build_lora
from models.adapter import build_adapter



# SETTINGS
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
print("Device:", device)


# LOAD MODELS
print("\nLoading models...")
zero_shot = load_mobile_sam(device=device)

lora_model, lora_layers = build_lora(
    rank=4,
    alpha=4,
    device=device
)

adapter_model, adapter_layers = build_adapter(
    bottleneck_dim=16,
    device=device
)

# Evaluation mode keeps pretrained model behavior fixed.
# It does not disable gradient computation.
zero_shot.eval()
lora_model.eval()
adapter_model.eval()

assert len(lora_layers) == 10
assert len(adapter_layers) == 10



# TEST A: ZERO-SHOT EQUIVALENCE
print("\n" + "=" * 70)
print("TEST A: ZERO-SHOT EQUIVALENCE")
print("=" * 70)

# Synthetic image, not from NuInsSeg.
# TinyViT expects a 1024 x 1024 input.
image = torch.randn(
    1, 3, 1024, 1024,
    device=device
)

with torch.no_grad():
    zs_features = zero_shot.image_encoder(image)
    lora_features = lora_model.image_encoder(image)
    adapter_features = adapter_model.image_encoder(image)


def compare_outputs(reference, adapted, method):

    max_diff = (
        reference - adapted
    ).abs().max().item()

    mean_diff = (
        reference - adapted
    ).abs().mean().item()

    print(f"\n{method}")
    print(f"Maximum absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference:    {mean_diff:.10f}")

    assert torch.allclose(
        reference,
        adapted,
        atol=1e-5,
        rtol=1e-5
    ), f"{method}: Output differs from zero-shot!"

    print(f"{method}: PASS")


compare_outputs(
    zs_features,
    lora_features,
    "LoRA"
)

compare_outputs(
    zs_features,
    adapter_features,
    "Adapter"
)

del zs_features, lora_features, adapter_features
del image


# ------------------------------------------------------------------------------
# TEST B: GRADIENT FLOW
print("\n" + "=" * 70)
print("TEST B: GRADIENT FLOW")
print("=" * 70)

# This is a diagnostic test, not segmentation training.
#
# We test one modified module from each model.
#
# LoRA:
#   A initialized randomly
#   B initialized to zero
#   B should receive nonzero gradients initially.
#
# Adapter:
#   Down-projection initialized randomly
#   Up-projection initialized to zero
#   Up-projection should receive nonzero gradients initially.
#
# The other projection can have zero gradients
# on the first backward pass.


def check_gradients(model, method):
    model.eval()
    model.zero_grad(set_to_none=True)


    # SELECT MODULE AND SYNTHETIC INPUT
    if method == "LoRA":
        module = (
            model.image_encoder
            .layers[1].blocks[0].attn.qkv
        )
        input_dim = module.original_layer.in_features
    elif method == "Adapter":
        module = (
            model.image_encoder
            .layers[1].blocks[0].adapter
        )
        input_dim = module.down.in_features
    else:
        raise ValueError(f"Unknown method: {method}")

    x = torch.randn(
        2, 16, input_dim,
        device=device
    )


    # FORWARD PASS
    output = module(x)
    # Random synthetic target.
    # Unlike output.square().mean(), this creates a
    # nonzero learning signal for the zero-initialized
    # output projection.
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)


    # BACKWARD PASS
    loss.backward()

    print(f"\n{method}")
    print(f"Diagnostic loss: {loss.item():.6f}")


    # INSPECT GRADIENTS IN TESTED MODULE
    print("\nGradient norms in tested module:")
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            print(f"  {name}: None")
        else:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {grad_norm:.10f}")


    # CHECK EXPECTED NONZERO GRADIENT
    if method == "LoRA":
        output_projection = module.lora_B.weight
    else:
        output_projection = module.up.weight
    assert output_projection.grad is not None, (
        f"{method}: Output projection has no gradient!"
    )

    assert torch.isfinite(output_projection.grad).all(), (
        f"{method}: Non-finite gradients detected!"
    )

    assert output_projection.grad.norm().item() > 0, (
        f"{method}: Output projection gradient is zero!"
    )


    # VERIFY FROZEN PARAMETERS
    frozen_with_grad = []
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            frozen_with_grad.append(name)

    print(
        "\nFrozen parameters receiving gradients:",
        len(frozen_with_grad)
    )

    assert len(frozen_with_grad) == 0, (
        f"{method}: Frozen parameters received gradients!"
    )


    # VERIFY TRAINABLE PARAMETER NAMES
    trainable_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    if method == "LoRA":
        assert all(
            "lora_A" in name or "lora_B" in name
            for name in trainable_names
        )
    else:
        assert all(
            ".adapter." in name
            for name in trainable_names
        )
    print(f"{method}: PASS")


# RUN GRADIENT TESTS
check_gradients(
    lora_model,
    "LoRA"
)

check_gradients(
    adapter_model,
    "Adapter"
)



# FINAL SUMMARY
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Zero-shot equivalence: PASS")
print("LoRA gradient flow:    PASS")
print("Adapter gradient flow: PASS")
print("Frozen weights:        PASS")

print("\nAll checks passed.")
