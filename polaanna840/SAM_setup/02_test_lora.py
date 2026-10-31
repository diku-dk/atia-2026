from models.lora import build_lora


model, adapted_layers = build_lora(
    rank=4,
    alpha=4
)


print("\nLoRA layers:")
for name in adapted_layers:
    print("  ", name)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\nParameter counts:")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_params / total_params:.4f}%"
)


print("\nTrainable parameters:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(
            f"{name:80s} "
            f"{tuple(param.shape)}"
        )