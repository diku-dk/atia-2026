#  original MobileSAM
# freezes it
# adds LoRA

import torch
import torch.nn as nn

from models.base import load_mobile_sam


class LoRALinear(nn.Module):
    """
    Wrap a frozen Linear layer with a trainable low-rank LoRA update.
    """

    def __init__(self, original_layer, rank=4, alpha=4):
        super().__init__()

        if not isinstance(original_layer, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear layer.")

        self.original_layer = original_layer

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original pretrained projection
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Low-rank matrices
        self.lora_A = nn.Linear(
            in_features,
            rank,
            bias=False
        )

        self.lora_B = nn.Linear(
            rank,
            out_features,
            bias=False
        )

        # Standard LoRA-style initialization:
        # A starts random, B starts at zero.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=5 ** 0.5
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

    def forward(self, x):
        original_output = self.original_layer(x)

        lora_update = self.lora_B(
            self.lora_A(x)
        )

        return original_output + self.scaling * lora_update



def add_lora_to_tinyvit(image_encoder, rank=4, alpha=4):
    """
    Add LoRA to the qkv projection of every TinyViT attention block.
    """

    adapted_layers = []

    for layer_idx, layer in enumerate(image_encoder.layers):

        # Stage 0 is convolutional and has no TinyViT attention blocks
        if not hasattr(layer, "blocks"):
            continue

        for block_idx, block in enumerate(layer.blocks):

            if not hasattr(block, "attn"):
                continue

            if not hasattr(block.attn, "qkv"):
                continue

            original_qkv = block.attn.qkv

            if not isinstance(original_qkv, nn.Linear):
                continue

            block.attn.qkv = LoRALinear(
                original_layer=original_qkv,
                rank=rank,
                alpha=alpha
            )

            name = (
                f"image_encoder.layers.{layer_idx}."
                f"blocks.{block_idx}.attn.qkv"
            )

            adapted_layers.append(name)

    return adapted_layers


def build_lora(rank=4, alpha=4, device=None):
    """
    Load MobileSAM, freeze all pretrained parameters,
    and add LoRA to TinyViT attention qkv projections.
    """

    model = load_mobile_sam(device=device)

    #freeze eveyrhting
    for param in model.parameters():
        param.requires_grad = False

    # add trainable lora parameters to tinyvit
    adapted_layers = add_lora_to_tinyvit(model.image_encoder, rank=rank, alpha=alpha)
    return model, adapted_layers
