import numpy as np


def preprocess_mri(image):
    """Convert a Cheng MRI slice into a three-channel uint8 image."""

    image = image.astype(np.float32)
    nonzero = image[image > 0]

    if nonzero.size == 0:
        raise ValueError("MRI image contains no positive pixels.")

    lower = np.percentile(nonzero, 0.5)
    upper = np.percentile(nonzero, 99.5)

    if upper <= lower:
        raise ValueError("Invalid MRI intensity range.")

    # Clip and rescale nonzero MRI intensities
    normalised = np.clip(image, lower, upper)
    normalised = (normalised - lower) / (upper - lower)
    normalised *= 255.0

    # Preserve black background
    normalised[image == 0] = 0

    # Convert grayscale to RGB
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    rgb = np.repeat(normalised[:, :, None], 3, axis=2)

    return rgb
