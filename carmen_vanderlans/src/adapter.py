
"""Parameter-efficient Adapter fine-tuning for MobileSAM."""

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter following Gu et al."""

    def __init__(self, dim, ratio=0.25, skip_connect=True):
        super().__init__()

        hidden_dim = int(dim * ratio)
        if hidden_dim < 1:
            raise ValueError("Adapter bottleneck must be positive")

        self.down = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)
        self.skip_connect = skip_connect

    def forward(self, x):
        update = self.up(self.activation(self.down(x)))

        if self.skip_connect:
            return x + update

        return update


def add_encoder_adapters(block):
    """Add the two Gu-style adapters to a TinyViT block."""

    dim = block.attn.qkv.in_features

    block.space_adapter = BottleneckAdapter(
        dim, skip_connect=True
    )
    block.mlp_adapter = BottleneckAdapter(
        dim, skip_connect=False
    )

    # Adapt the attention output before its residual connection.
    def attention_hook(module, inputs, output):
        return block.space_adapter(output)

    # Add a parallel adapter branch to the original MLP.
    def mlp_hook(module, inputs, output):
        return output + 0.5 * block.mlp_adapter(inputs[0])

    block.attn.register_forward_hook(attention_hook)
    block.mlp.register_forward_hook(mlp_hook)


def add_decoder_adapters(block):
    """Add two Gu-style adapters to a decoder transformer block."""

    dim = block.norm2.normalized_shape[0]

    block.space_adapter = BottleneckAdapter(
        dim, skip_connect=True
    )
    block.mlp_adapter = BottleneckAdapter(
        dim, skip_connect=False
    )

    # Applied after token-to-image cross-attention and before norm2.
    def norm_hook(module, inputs):
        return (block.space_adapter(inputs[0]),)

    # Parallel MLP adapter, scaled by 0.5.
    def mlp_hook(module, inputs, output):
        return output + 0.5 * block.mlp_adapter(inputs[0])

    block.norm2.register_forward_pre_hook(norm_hook)
    block.mlp.register_forward_hook(mlp_hook)


def apply_adapter(model):
    """Freeze pretrained MobileSAM and add trainable adapters."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder_count = 0

    for stage in model.image_encoder.layers[1:]:
        for block in stage.blocks:
            if hasattr(block, "attn") and hasattr(block, "mlp"):
                add_encoder_adapters(block)
                encoder_count += 1

    decoder_count = 0

    for block in model.mask_decoder.transformer.layers:
        add_decoder_adapters(block)
        decoder_count += 1

    if encoder_count != 10:
        raise RuntimeError(
            f"Expected 10 TinyViT blocks, found {encoder_count}"
        )

    if decoder_count != 2:
        raise RuntimeError(
            f"Expected 2 decoder blocks, found {decoder_count}"
        )

    allowed = (".space_adapter.", ".mlp_adapter.")

    for name, parameter in model.named_parameters():
        should_train = any(part in name for part in allowed)

        if parameter.requires_grad != should_train:
            raise RuntimeError(
                f"Incorrect trainable status: {name}"
            )

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print(f"Encoder blocks adapted: {encoder_count}")
    print(f"Decoder blocks adapted: {decoder_count}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.3f}%")

    return model

