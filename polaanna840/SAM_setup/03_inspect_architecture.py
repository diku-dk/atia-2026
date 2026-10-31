from pathlib import Path
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry


# 1. LOAD ORIGINAL MOBILE SAM
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAM_CHECKPOINT = (
    PROJECT_ROOT
    / "SAM_code_github"
    / "MobileSAM"
    / "weights"
    / "mobile_sam.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Checkpoint exists:", SAM_CHECKPOINT.exists())
print("Device:", device)

mobile_sam = sam_model_registry["vit_t"](
    checkpoint=str(SAM_CHECKPOINT)
)

mobile_sam.to(device)
mobile_sam.eval()
print("\nMobileSAM loaded successfully!")


# 2. TOP-LEVEL ARCHITECTURE
print("\n" + "=" * 70)
print("TOP-LEVEL MOBILE SAM COMPONENTS")
print("=" * 70)

for name, module in mobile_sam.named_children():
    print(f"{name:25s} {type(module).__name__}")


# 3. TINYVIT IMAGE ENCODER STRUCTURE
print("\n" + "=" * 70)
print("TINYVIT IMAGE ENCODER HIERARCHY")
print("=" * 70)

encoder = mobile_sam.image_encoder

for name, module in encoder.named_modules():
    if name == "":
        continue

    depth = name.count(".")
    indent = "  " * depth

    print(f"{indent}{name.split('.')[-1]}: {type(module).__name__}")


# 4. FIND ATTENTION MODULES
print("\n" + "=" * 70)
print("ATTENTION MODULES")
print("=" * 70)

for name, module in encoder.named_modules():
    if "attention" in type(module).__name__.lower():
        print(f"{name:75s} {type(module).__name__}")



# 5. FIND QKV PROJECTIONS
print("\n" + "=" * 70)
print("QKV PROJECTIONS - POSSIBLE LORA INSERTION POINTS")
print("=" * 70)

qkv_count = 0

for name, module in encoder.named_modules():
    if name.split(".")[-1].lower() == "qkv":
        qkv_count += 1

        print(f"\n{name}")
        print(f"  Type: {type(module).__name__}")

        if isinstance(module, nn.Linear):
            print(f"  Input features:  {module.in_features}")
            print(f"  Output features: {module.out_features}")
            print(f"  Weight shape:    {tuple(module.weight.shape)}")

print(f"\nTotal QKV modules: {qkv_count}")


# 6. FIND MLP MODULES
print("\n" + "=" * 70)
print("MLP MODULES")
print("=" * 70)

for name, module in encoder.named_modules():
    if type(module).__name__.lower() == "mlp":
        print(f"\n{name}")
        print(f"  Type: {type(module).__name__}")

        for child_name, child in module.named_children():
            print(f"  {child_name:15s} {type(child).__name__}")

            if isinstance(child, nn.Linear):
                print(
                    f"    {child.in_features} -> {child.out_features}"
                )



# 7. FIND TRANSFORMER BLOCKS
print("\n" + "=" * 70)
print("TRANSFORMER BLOCKS - POSSIBLE ADAPTER INSERTION POINTS")
print("=" * 70)

block_count = 0

for name, module in encoder.named_modules():

    class_name = type(module).__name__.lower()

    if class_name in ("tinyvitblock", "tiny_vit_block"):
        block_count += 1

        print(f"\n{name}")
        print(f"  Type: {type(module).__name__}")

        print("  Direct children:")

        for child_name, child in module.named_children():
            print(f"    {child_name:15s} {type(child).__name__}")

print(f"\nTotal TinyViT transformer blocks: {block_count}")



# 8. FIND OTHER BLOCK TYPES
print("\n" + "=" * 70)
print("OTHER BLOCK TYPES")
print("=" * 70)

for name, module in encoder.named_modules():

    class_name = type(module).__name__.lower()

    if "block" in class_name and class_name not in (
        "tinyvitblock",
        "tiny_vit_block",
    ):
        print(f"{name:75s} {type(module).__name__}")



# 9. PARAMETER COUNTS BY COMPONENT
print("\n" + "=" * 70)
print("PARAMETER COUNTS BY COMPONENT")
print("=" * 70)

for name, module in mobile_sam.named_children():

    count = sum(p.numel() for p in module.parameters())

    print(f"{name:25s} {count:,}")

print("\nInspection complete!")








import inspect

block = mobile_sam.image_encoder.layers[1].blocks[0]

print("\n" + "=" * 70)
print("TINYVIT BLOCK FORWARD METHOD")
print("=" * 70)

print(inspect.getsource(type(block).forward))

print("\n" + "=" * 70)
print("ATTENTION FORWARD METHOD")
print("=" * 70)

print(inspect.getsource(type(block.attn).forward))

print("\n" + "=" * 70)
print("MLP FORWARD METHOD")
print("=" * 70)

print(inspect.getsource(type(block.mlp).forward))