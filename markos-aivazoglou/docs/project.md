# Project

## Context

This directory is one student's project (`markos-aivazoglou`) inside the shared course repository `diku-dk/atia-2026` (NDAK15013U, Advanced Topics in Image Analysis, University of Copenhagen).

Sibling directories at the repo root (`template/`, `martin_norgaard/`, `tobi3010/`, …) belong to other students or the teaching team.

## Research project

Applying research methods to a computer vision / image analysis topic: improving small-object **instance segmentation** on high-resolution images in **agricultural settings**. Planned axes of investigation:

- Survey and compare different approaches for small-object instance segmentation on high-res images (e.g. tiling/SAHI-style inference, multi-scale training, high-res-aware architectures).
- Compare these approaches across multiple models, spanning both large-scale and edge-deployment variants (frameworks in use: Ultralytics YOLO and EoMT; more may be added as decisions are made).

## Coding rules

- Use the frameworks' high-level APIs and default behaviour (Ultralytics; for EoMT, upstream's own Lightning code and defaults). Do not build custom data loaders, metric passes, benchmarks or other performance/memory workarounds (premature optimisations) unless explicitly asked. When a default looks slow or memory-hungry, report it and let the user decide.
- EoMT: reuse upstream's code (copied into `src/eomt/` and `src/datasets/`, a provenance header each) and its entry points (LightningCLI, Lightning checkpoints) directly. Our code is thin glue only: no re-implementations, no wrapper/subclass layers. If upstream needs a change, edit the copy and note it in its header.

## Shared-repository rules (from the root README)

- Only change files inside `markos-aivazoglou/`. Do not modify other students' directories or shared course files (root `README.md`, `LICENSE`, `template/`).
- This is an ordinary directory in the course repo. Never run `git init` here or change the repo's remotes.
- Work on the branch `student/markos-aivazoglou`, not `main`. From the repo root, stage only this directory (`git add -- markos-aivazoglou`), never `git add .`.
- To pick up shared updates: commit, then `git fetch origin && git merge origin/main`.
- Submission is a PR from the student branch into `main`, titled like `ATIA project — markos-aivazoglou — <short title>`. The teaching team merges it.
- Never commit datasets, checkpoints or weights, logs or `wandb/` runs, `.venv/`, or credentials. `.gitignore` already excludes `/data/*` (except `data/README.md`), `/checkpoints/`, `/weights/`, `/logs/` and `/wandb/`. Put small result tables and figures in `results/`.

## Environment

The project uses a **uv**-managed virtualenv at `.venv/` (Python 3.13), not the conda setup the root README suggests. Installed packages include `torch`, `ultralytics` (YOLO; also gives the `yolo` CLI), `opencv-python`, `numpy`, `polars`, `matplotlib`, `seaborn` and `tqdm`, plus EoMT's dependencies: `lightning` 2.6.6, `transformers` **4.56.1** (upstream's pin; 5.x renamed the DINOv3 `layer` attribute `src/eomt/models/vit.py` relies on), `timm`, `torchmetrics` (EoMT's metric base classes; `faster-coco-eval` is installed but no longer used since `src/eomt/mask_ap.py` replaced its mask mAP), `pycocotools` (rasterises EoMT's training polygons; hotcoco stays the scorer) and `jsonargparse[signatures]` ≥ 4.39 (required by lightning 2.6's CLI). The DINOv3 weights on HF are gated (access granted to this account; `facebook/dinov3-vits16-pretrain-lvd1689m` is cached). Versions are pinned in `requirements.txt` in this directory (`uv pip install -r requirements.txt`); update it (`uv pip freeze`) whenever a dependency is added.

```bash
source .venv/bin/activate        # or prefix commands with `uv run`
uv pip install -e .              # installs the package from setup.py
```

Always run project Python through `uv run ...` (e.g. `uv run scripts/dataset_stats.py`), not a bare `python`, so it uses `.venv`.

## Layout

Generated from the True Neutral Cookiecutter:
- `src/`: reusable implementation code. The importable package is literally named `src` (`setup.py` has `name="src"`).
- `scripts/`: execution entry points and notebooks.
- `tests/`: unittest tests (`uv run -m unittest discover tests`).
- `data/`: data-access and preprocessing instructions go in `data/README.md`. The data itself is git-ignored.
- `docs/`, `results/`: documentation and small committed outputs.

## Documentation expectations

The project `README.md` must eventually cover: the research question, related papers, which parts are reused and which are your own implementation, setup and dependency versions, dataset access/preprocessing/splits, exact commands to reproduce the experiments, seeds, metrics, baselines, how the results were generated, runtime and hardware needs, and limitations.
