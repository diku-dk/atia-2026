# verify that one real NuInsSeg nucleus can pass through the full segmentation pipeline,
# produce a loss, and send gradients back to the LoRA or adapter parameters


'''
For each method this test verifies that:
- A real nucleus is segmented using its ground-truth bounding box.
- The predicted logits and ground-truth mask have matching dimensions.
- A segmentation loss can be calculated.
- Gradients reach the trainable LoRA or adapter parameters.
- Frozen pretrained parameters receive no gradients.
- One optimizer update can run.
One technical limitation: this is a minimal gradient-flow test, using binary cross-entropy only.
It doesn't establish our final training loss, optimizer settings, or hyperparameters.
Also, this is a full-resolution gradient-enabled forward/backward pass. It require more memory than Step 7, particularly on CPU.
'''


from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from models.losses import dice_bce_loss

from mobile_sam import SamPredictor

from models.lora import build_lora
from models.adapter import build_adapter
from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    extract_nucleus,
    get_nucleus_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Load one valid nucleus (implementation test only)
pairs = []

for tissue_dir in sorted(DATA_ROOT.iterdir()):
    if not tissue_dir.is_dir():
        continue

    image_dir = tissue_dir / "tissue images"
    mask_dir = tissue_dir / "label masks modify"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        continue

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / f"{image_path.stem}.tif"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))

assert len(pairs) == 665

image_path, mask_path = pairs[0]
image, instance_mask = load_nuinsseg_sample(image_path, mask_path)

nucleus_id = get_nucleus_ids(instance_mask)[0]
gt_mask, bbox = extract_nucleus(instance_mask, nucleus_id)

print("Sample:", image_path.stem)
print("Nucleus ID:", nucleus_id)



# 2. Gradient-enabled segmentation forward

def forward_with_gradients(model, image, bbox):

    predictor = SamPredictor(model)

    # SamPredictor's image preprocessing is reused,
    # but we run the model components with gradients.
    transformed_image = predictor.transform.apply_image(image)

    image_tensor = torch.as_tensor(
        transformed_image,
        device=DEVICE
    ).permute(2, 0, 1).contiguous()[None, :, :, :]

    image_embedding = model.image_encoder(
        model.preprocess(image_tensor)
    )

    transformed_box = predictor.transform.apply_boxes(
        bbox[None, :],
        image.shape[:2]
    )

    box_tensor = torch.as_tensor(
        transformed_box,
        dtype=torch.float32,
        device=DEVICE
    )

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None
    )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    masks = model.postprocess_masks(
        low_res_masks,
        input_size=transformed_image.shape[:2],
        original_size=image.shape[:2]
    )

    return masks



# 3. Test one backward pass
def test_method(method_name, model):

    print(f"\nTesting {method_name}")

    model.eval()

    # Only PEFT parameters are trainable
    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    assert len(trainable_parameters) > 0

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-4
    )

    optimizer.zero_grad()

    logits = forward_with_gradients(
        model, image, bbox
    )

    target = torch.as_tensor(
        gt_mask.astype(np.float32),
        device=DEVICE
    )[None, None, :, :]

    assert logits.shape == target.shape

    loss, dice_loss, bce_loss = dice_bce_loss(logits, target)

    print("Dice loss:", dice_loss.item())
    print("BCE loss:", bce_loss.item())



    print("Prediction shape:", tuple(logits.shape))
    print("Loss:", loss.item())

    loss.backward()

    # Verify at least one trainable parameter
    # receives a nonzero gradient.
    gradient_norms = [
        p.grad.norm().item()
        for p in trainable_parameters
        if p.grad is not None
    ]

    assert gradient_norms, "No gradients received"
    assert any(g > 0 for g in gradient_norms), (
        "All trainable gradients are zero"
    )

    frozen_with_gradients = [
        name for name, p in model.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]

    assert not frozen_with_gradients, (
        f"Frozen parameters received gradients: "
        f"{frozen_with_gradients[:5]}"
    )

    optimizer.step()

    print("Trainable parameters with gradients:",
          len(gradient_norms))
    print("Maximum gradient norm:", max(gradient_norms))
    print("Frozen parameters with gradients:", 0)
    print(f"{method_name}: PASS")


# 4. Run LoRA and adapter tests
lora_model, _ = build_lora(
    rank=4,
    alpha=4,
    device=DEVICE
)

test_method("LoRA", lora_model)

del lora_model

adapter_model, _ = build_adapter(
    bottleneck_dim=16,
    device=DEVICE
)

test_method("Adapter", adapter_model)

print("\nALL SEGMENTATION GRADIENT TESTS PASSED")
