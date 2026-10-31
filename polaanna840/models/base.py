from pathlib import Path
import torch
from mobile_sam import sam_model_registry


# oad a fresh original MobileSAM.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAM_CHECKPOINT = (
    PROJECT_ROOT
    / "SAM_code_github"
    / "MobileSAM"
    / "weights"
    / "mobile_sam.pt"
)


def load_mobile_sam(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = sam_model_registry["vit_t"](
        checkpoint=str(SAM_CHECKPOINT)
    )

    model.to(device)

    return model