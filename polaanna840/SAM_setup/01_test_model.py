# tests whether MobileSAM can be loaded successfully

from pathlib import Path

import torch
from mobile_sam import sam_model_registry


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

print("MobileSAM loaded successfully!")



