
import torch

from models.losses import dice_bce_loss


# A small synthetic ground-truth mask
target = torch.zeros((1, 1, 32, 32))
target[:, :, 10:20, 10:20] = 1.0


# Test 1: Nearly perfect prediction
perfect_logits = torch.where(
    target == 1,
    torch.tensor(10.0),
    torch.tensor(-10.0)
)

perfect_loss, perfect_dice, perfect_bce = dice_bce_loss(
    perfect_logits, target
)


# Test 2: Completely wrong prediction
wrong_logits = -perfect_logits

wrong_loss, wrong_dice, wrong_bce = dice_bce_loss(
    wrong_logits, target
)


# Test 3: Verify gradients
trainable_logits = torch.zeros_like(
    target,
    requires_grad=True
)

loss, dice, bce = dice_bce_loss(
    trainable_logits, target
)

loss.backward()


# Assertions
assert perfect_loss.item() < 0.01
assert wrong_loss.item() > perfect_loss.item()

assert trainable_logits.grad is not None
assert torch.isfinite(trainable_logits.grad).all()
assert trainable_logits.grad.abs().sum().item() > 0

print("Nearly perfect prediction loss:", perfect_loss.item())
print("Completely wrong prediction loss:", wrong_loss.item())

print("\nGradient test: PASS")
print("Dice + BCE loss test: PASS")
