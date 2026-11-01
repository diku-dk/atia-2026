import numpy as np


def dice_score(prediction, ground_truth):
    """Calculate the Dice similarity coefficient."""

    prediction = np.asarray(prediction) > 0
    ground_truth = np.asarray(ground_truth) > 0

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape.")

    intersection = np.logical_and(prediction, ground_truth).sum()

    denominator = prediction.sum() + ground_truth.sum()

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)
