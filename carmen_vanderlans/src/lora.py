
import math

import torch
from torch import nn


class LoRAQKV(nn.Module):
    """Add LoRA updates to Q and V in a combined QKV projection."""

    def __init__(self, original, rank=4):
        super().__init__()
        self.original = original
        dim = original.in_features

        self.q_a = nn.Linear(dim, rank, bias=False)
        self.q_b = nn.Linear(rank, dim, bias=False)
        self.v_a = nn.Linear(dim, rank, bias=False)
        self.v_b = nn.Linear(rank, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.q_a, self.v_a):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in (self.q_b, self.v_b):
            nn.init.zeros_(layer.weight)

    def forward(self, x):
        qkv = self.original(x)
        q_update = self.q_b(self.q_a(x))
        v_update = self.v_b(self.v_a(x))
        return qkv + torch.cat(
            (q_update, torch.zeros_like(q_update), v_update), dim=-1
        )


class LoRALinear(nn.Module):
    """Add a low-rank update to an individual attention projection."""

    def __init__(self, original, rank=4):
        super().__init__()
        self.original = original

        self.a = nn.Linear(original.in_features, rank, bias=False)
        self.b = nn.Linear(rank, original.out_features, bias=False)

        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.original(x) + self.b(self.a(x))


def apply_lora(model, rank=4):
    """Freeze MobileSAM and insert trainable LoRA into encoder and decoder."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder_count = 0

    # TinyViT has attention blocks in layers 1, 2 and 3.
    for stage in model.image_encoder.layers[1:]:
        for block in stage.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "qkv"):
                attention.qkv = LoRAQKV(attention.qkv, rank=rank)
                encoder_count += 1

    decoder = model.mask_decoder.transformer
    decoder_count = 0

    attention_modules = []

    for block in decoder.layers:
        attention_modules.extend([
            block.self_attn,
            block.cross_attn_token_to_image,
            block.cross_attn_image_to_token,
        ])

    attention_modules.append(decoder.final_attn_token_to_image)

    for attention in attention_modules:
        attention.q_proj = LoRALinear(attention.q_proj, rank=rank)
        attention.v_proj = LoRALinear(attention.v_proj, rank=rank)
        decoder_count += 1

    # Check that the intended architecture was actually modified.
    if encoder_count != 10:
        raise RuntimeError(
            f"Expected 10 TinyViT attention blocks, found {encoder_count}"
        )

    if decoder_count == 0:
        raise RuntimeError("No mask decoder attention modules were adapted")

    # Verify all original weights are still frozen.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not any(
            part in name
            for part in (
                ".q_a.", ".q_b.", ".v_a.", ".v_b.",
                ".a.", ".b.",
            )
        ):
            raise RuntimeError(f"Unexpected trainable parameter: {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Encoder attention blocks adapted: {encoder_count}")
    print(f"Decoder attention modules adapted: {decoder_count}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.3f}%")

    return model

