import h5py
import numpy as np


def load_cheng_sample(path):
    """Load one Cheng MRI image and its metadata."""

    with h5py.File(path, "r") as f:
        data = f["cjdata"]

        # Restore MATLAB image orientation from HDF5 storage
        image = data["image"][:].T
        mask = data["tumorMask"][:].T
        label = int(data["label"][0, 0])

        pid_codes = data["PID"][:].flatten()
        patient_id = "".join(chr(int(c)) for c in pid_codes)

    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes differ: {path}")

    if not np.any(mask > 0):
        raise ValueError(f"Empty tumour mask: {path}")

    if label not in (1, 2, 3):
        raise ValueError(f"Unexpected tumour label: {label}")

    return {
        "image": image,
        "mask": mask > 0,
        "label": label,
        "patient_id": patient_id,
        "path": str(path),
    }
