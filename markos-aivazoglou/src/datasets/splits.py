"""Split-seed dataset roots, used by the experiment runner."""

from pathlib import Path

# Split seeds converted by scripts/split_seeds.sh, one dataset root each (data/seed<N>/).
SPLIT_SEEDS = (42, 0, 1)


def data_root(seed: int) -> Path:
    """Dataset root of one split seed, resolved relative to this file
    (src/datasets/splits.py -> project root -> data/seed<N>/), independent of the caller's cwd."""
    return Path(__file__).resolve().parents[2] / "data" / f"seed{seed}"
