
import torch
import torch.nn.functional as F


def dice_bce_loss(logits, target, smooth=1e-6):
    """
    Combined Dice + binary cross-entropy loss.

    logits: (B, 1, H, W), raw model predictions
    target: (B, 1, H, W), binary ground-truth masks
    """

    if logits.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {logits.shape} vs {target.shape}"
        )

    target = target.float()

    # Binary cross-entropy
    bce = F.binary_cross_entropy_with_logits(
        logits, target
    )

    # Soft Dice loss
    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    dice_loss = 1 - dice.mean()

    total_loss = dice_loss + bce

    return total_loss, dice_loss, bce
