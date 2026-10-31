# MobileSAM Adaptation for Nuclei Instance Segmentation

## Overview

This project investigates whether parameter-efficient fine-tuning (PEFT) improves MobileSAM's performance on nuclei instance segmentation and whether these improvements vary with annotation ambiguity.

Two PEFT approaches, Low-Rank Adaptation (LoRA) and bottleneck Adapters, are compared against pretrained zero-shot MobileSAM on the NuInsSeg dataset.

The evaluation uses tissue-disjoint five-fold cross-validation, paired nucleus-level segmentation metrics, and tissue-level statistical comparisons.

## Dataset

The project uses **NuInsSeg**, a dataset of 665 H&E-stained histopathology image patches (512 × 512 pixels), covering 31 tissue types and containing 30,698 annotated nuclei.

**Dataset access:** [NuInsSeg on Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg/code)

The dataset provides histopathology images, corrected nucleus instance masks, and vague-region annotations. The original images and masks are not redistributed in this repository.

Each nucleus is segmented using a tight bounding-box prompt derived from its ground-truth mask. The same prompts are used for zero-shot MobileSAM, LoRA, and Adapter.

### Annotation ambiguity

Annotation ambiguity is quantified using the overlap between each nucleus and its corresponding vague-region mask:

\[
r_i = \frac{|N_i \cap V|}{|N_i|}
\]

where \(N_i\) is the ground-truth nucleus mask and \(V\) is the vague-region mask.

The resulting score ranges from 0 (no overlap) to 1 (complete overlap). The analysis considers three groups: \(r_i = 0\), \(0 < r_i < 1\), and \(r_i = 1\), alongside the continuous overlap measure.

## Experimental Design

### Cross-validation

The experiments use **five-fold tissue-disjoint cross-validation**. All images belonging to the same tissue type remain within a single fold.

For each iteration:
- Three folds are used for training.
- One fold is used for validation.
- One fold is held out for testing.

The frozen tissue assignments are provided in `results/splits_final_v1/`.

### Models

Three approaches are evaluated:

- **Zero-shot MobileSAM:** Pretrained model without task-specific fine-tuning.
- **LoRA:** Low-rank adaptation of the TinyViT image encoder.
- **Adapter:** Bottleneck adapter modules inserted into the TinyViT image encoder.

LoRA and Adapter are trained using three random seeds (42, 123, 456) across five cross-validation iterations, resulting in **30 fine-tuning experiments**.

### Training configuration

| Parameter | Value |
|---|---|
| Maximum epochs | 20 |
| Early stopping patience | 3 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| LoRA rank | 4 |
| LoRA alpha | 4 |
| Adapter bottleneck dimension | 16 |
| Prompt batch size | 8 |
| Training nuclei | All available nuclei |
| Random seeds | 42, 123, 456 |

The final evaluation manifest records the configurations, checkpoint hashes, selected epochs, and validation losses for all 30 runs.

## Repository Structure

```text
polaanna840/
├── experiments/                  # Training, evaluation and analysis scripts
├── models/                       # MobileSAM, LoRA, Adapter and loss implementations
├── SAM_setup/                    # Model setup and diagnostic scripts
├── results/
│   ├── splits_final_v1/          # Frozen tissue-level cross-validation splits
│   ├── analysis/
│   │   └── heldout_v2/           # Final statistical analysis tables
│   ├── evaluation/
│   │   └── manifest.json         # Evaluation and checkpoint metadata
│   ├── dataset_inventory_corrected.csv
│   ├── nucleus_vague_overlap_by_tissue.csv
│   └── tissue_profile.csv
├── requirements.txt
└── README.md
```

The dataset, pretrained weights, trained checkpoints, and large prediction files are excluded.

## Installation and Setup

The final evaluation environment used Python 3.11.3 and PyTorch 2.2.2+cu121.

### 1. Install Python dependencies

From the `polaanna840/` project directory:

```bash
pip install -r requirements.txt
```

### 2. Install MobileSAM

The project uses the original [MobileSAM implementation](https://github.com/ChaoningZhang/MobileSAM).

Clone the repository into the expected location:

```bash
git clone https://github.com/ChaoningZhang/MobileSAM.git SAM_code_github/MobileSAM
pip install -e SAM_code_github/MobileSAM
```

Download the pretrained MobileSAM checkpoint from the official MobileSAM repository and place it at:

```text
SAM_code_github/MobileSAM/weights/mobile_sam.pt
```

### 3. Prepare NuInsSeg

Download the dataset from:

[https://www.kaggle.com/datasets/ipateam/nuinsseg/code](https://www.kaggle.com/datasets/ipateam/nuinsseg/code)

Extract the dataset under:

```text
data/NuInsSeg/
```

The scripts expect tissue-specific directories containing the images, corrected nucleus masks, and vague-region masks.

## Running the Experiments

Run the commands below from the `polaanna840/` directory.

### 1. Training

```bash
python -m experiments.train_peft
```

This runs the five cross-validation iterations for LoRA and Adapter using three random seeds.

Training outputs are stored under:

```text
results/training/final_full_nuclei/
```

The training script checks existing completed runs and refuses to overwrite incomplete experiment directories.

### 2. Evaluation

Evaluation requires a CUDA GPU and the trained checkpoints.

```bash
python -m experiments.evaluate_heldout \
  --expected-split-sha 1bc26ee83c8e345cb84bcf33ae3e3fe474a869942005724fa0115b1c6ca4104f \
  --vague-template "{data_root}/{tissue}/vague areas/mask binary/{stem}.png" \
  --output results/evaluation/reproduction_v1
```

The evaluation script verifies the frozen split hash and refuses to overwrite an existing output directory.

### 3. Statistical analysis

```bash
python -m experiments.analyze_heldout \
  --evaluation results/evaluation/reproduction_v1 \
  --output results/analysis/reproduction_v1
```

The analysis uses the completed nucleus-level evaluation results to compare segmentation performance and adaptation gains across methods, tissues, folds, seeds, and annotation ambiguity groups.

## Evaluation and Statistical Analysis

Segmentation performance is evaluated using hard binary per-nucleus Dice scores. Adaptation gains are defined relative to zero-shot MobileSAM:

\[
\Delta_i^{m} = \mathrm{Dice}_i^{m} - \mathrm{Dice}_i^{ZS}
\]

where \(m\) denotes LoRA or Adapter.

The analysis includes overall comparisons, tissue-level and fold-level summaries, paired comparisons between methods, and comparisons across ambiguity groups.

The final analysis uses equal-weight tissue means, with results averaged across training seeds within each tissue. Bootstrap resampling uses 10,000 replicates with seed 2026.

Bootstrap intervals are interpreted as descriptive sensitivity intervals rather than fully calibrated inferential confidence intervals for the overlapping cross-validation training sets.

The final aggregated results are available in:

```text
results/analysis/heldout_v2/
```

## Reproducibility

The frozen tissue-level split is provided in `results/splits_final_v1/tissue_folds.json`.

Its SHA-256 hash is:

```text
1bc26ee83c8e345cb84bcf33ae3e3fe474a869942005724fa0115b1c6ca4104f
```

This hash matches the split recorded in the final evaluation manifest.

The manifest at `results/evaluation/manifest.json` documents the 30 trained checkpoint identities, model configurations, evaluation environment, and frozen fold assignments.

The original experiments were run on a Slurm GPU cluster. The repository provides the code, frozen splits, and aggregated analysis outputs, but does not include the underlying dataset or trained model checkpoints.