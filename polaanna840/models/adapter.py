
import torch
import torch.nn as nn

from models.base import load_mobile_sam


class BottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter.

    Input:  (B, N, C)
    Output: (B, N, C)
    """

    def __init__(self, dim, bottleneck_dim=16):
        super().__init__()

        self.down = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Initially, the adapter produces zero output.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.activation(self.down(x)))


class TinyViTBlockWithAdapter(nn.Module):
    """
    Keep the original TinyViT block intact,
    then add a trainable residual adapter.
    """

    def __init__(self, original_block, dim, bottleneck_dim=16):
        super().__init__()

        self.original_block = original_block

        self.adapter = BottleneckAdapter(
            dim=dim,
            bottleneck_dim=bottleneck_dim
        )

    def forward(self, x):
        x = self.original_block(x)
        return x + self.adapter(x)


def add_adapters_to_tinyvit(image_encoder, bottleneck_dim=16):
    """
    Add one adapter after each TinyViT transformer block.
    """

    adapted_layers = []

    for layer_idx, layer in enumerate(image_encoder.layers):

        if not hasattr(layer, "blocks"):
            continue

        for block_idx, block in enumerate(layer.blocks):

            if type(block).__name__ != "TinyViTBlock":
                continue

            # Determine feature dimension from the MLP.
            dim = block.mlp.fc2.out_features

            layer.blocks[block_idx] = TinyViTBlockWithAdapter(
                original_block=block,
                dim=dim,
                bottleneck_dim=bottleneck_dim
            )

            adapted_layers.append(
                f"image_encoder.layers.{layer_idx}."
                f"blocks.{block_idx}"
            )

    if len(adapted_layers) != 10:
        raise RuntimeError(
            f"Expected 10 TinyViT blocks, "
            f"but adapted {len(adapted_layers)}."
        )

    return adapted_layers


def build_adapter(bottleneck_dim=16, device=None):
    """
    Load pretrained MobileSAM, freeze original weights,
    and insert bottleneck adapters into TinyViT.
    """

    model = load_mobile_sam(device=device)

    for param in model.parameters():
        param.requires_grad = False

    adapted_layers = add_adapters_to_tinyvit(
        model.image_encoder,
        bottleneck_dim=bottleneck_dim
    )

    # Move newly added adapter parameters to the target device.
    if device is not None:
        model.to(device)

    return model, adapted_layers
