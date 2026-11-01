# Parameter-Efficient Fine-Tuning of MobileSAM for Brain Tumour Segmentation

## Overview

This project studies whether parameter-efficient fine-tuning (PEFT) can improve MobileSAM's performance on brain tumour segmentation while keeping computational requirements low.

This is done by comparing two PEFT methods, Low-Rank Adaptation (LoRA) and Adapter, with zero-shot MobileSAM and MedSAM on the Cheng brain tumour MRI dataset.

The main focus is on how the amount of labelled training data and the number of different patients used for training affect segmentation performance. The computational cost of LoRA and Adapter is also compared.

This project was carried out as part of the Advanced Topics in Image Analysis (ATIA) course at the University of Copenhagen in 2026.

## Dataset

The project uses the **Cheng brain tumour dataset**, containing 3,064 contrast-enhanced T1-weighted MRI slices from 233 patients. The dataset includes three tumour types: glioma, meningioma and pituitary tumours.

The images are divided into training, validation and test sets at the patient level, so images from the same patient do not appear in different sets.

| Split | Patients | Images |
|---|---:|---:|
| Training | 163 | 2,140 |
| Validation | 35 | 452 |
| Test | 35 | 472 |
| **Total** | **233** | **3,064** |

The patient-level split and training subsets are provided in `data/`. The original MRI images are not included in this repository.

All models receive the same bounding-box prompts derived from the ground-truth tumour masks.

## Experimental Design

### Training data size and patient diversity

The main experiments use **10%, 25%, 50% and 100%** of the available training images, with five random seeds per training fraction. LoRA and Adapter use the same training subsets for each seed.

To investigate patient diversity, additional experiments compare subsets containing the same number of training images but different numbers of patients. These comparisons are performed at the 10% and 25% training fractions.

The training subsets for all five seeds are included in `data/`.

### Models

Four model configurations are evaluated:

- **Zero-shot MobileSAM:** The pretrained model without fine-tuning.
- **Zero-shot MedSAM:** The pretrained medical segmentation model without fine-tuning.
- **MobileSAM + LoRA:** MobileSAM fine-tuned using low-rank adaptation.
- **MobileSAM + Adapter:** MobileSAM fine-tuned using bottleneck adapter modules.

For LoRA and Adapter, the pretrained MobileSAM parameters are frozen, and only the additional trainable parameters are updated.

### Training configuration

| Parameter | Value |
|---|---|
| Maximum epochs | 30 |
| Early stopping patience | 20 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.1 |
| Learning-rate scheduler | StepLR, factor 0.5 every 10 epochs |
| LoRA rank | 4 |
| Loss function | Binary cross-entropy + Dice loss |
| Training fractions | 10%, 25%, 50%, 100% |
| Random seeds | 5 |

Additional experiments compare different LoRA ranks and learning rates.

## Repository Structure

```text
carmen_vanderlans/
├── src/                       # Training, evaluation and analysis code
├── scripts/                   # Slurm experiment scripts
├── data/
│   ├── budgets/               # Original training subsets
│   ├── experiment_v2/         # Main training subsets
│   ├── experiment_extended/   # Patient-diversity subsets
│   ├── archive_original/      # Input CSV for generating updated subsets
│   ├── cheng_audit.csv
│   ├── cheng_patient_split.csv
│   ├── cheng_validation_files.csv
│   └── cheng_test_files.csv
├── results/
│   ├── baselines/             # Zero-shot results
│   ├── main/                  # LoRA and Adapter results
│   ├── diversity/             # Patient-diversity analysis
│   ├── rank/                  # LoRA rank comparison
│   ├── learning_rate/         # Learning-rate comparison
│   ├── compute/               # Computational-cost results
│   └── qualitative/           # Detailed segmentation results
├── figures/                   # Figures used in the report
├── requirements.txt
└── README.md
```

The MRI images, pretrained model weights, trained checkpoints and large prediction files are not included.

## Installation and Setup

The experiments were run using Python 3.10.22 and PyTorch 2.14.1+cu126.

### 1. Install Python dependencies

From the `carmen_vanderlans/` directory:

```bash
pip install -r requirements.txt
```

A compatible CUDA-enabled PyTorch installation is also required.

### 2. Set up MobileSAM and MedSAM

The project uses the official implementations of MobileSAM and MedSAM.

**MobileSAM:** https://github.com/ChaoningZhang/MobileSAM

Commit: `f706ad9c4eb7f219c00d9050e46328518ffb65d2`

**MedSAM:** https://github.com/bowang-lab/MedSAM

Commit: `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488`

Clone both repositories into the project directory, use the commits listed above and follow their installation instructions.

Download the pretrained model checkpoints and place them at:

```text
MobileSAM/weights/mobile_sam.pt
MedSAM/weights/medsam_vit_b.pth
```

### 3. Prepare the Cheng dataset

Download the Cheng brain tumour dataset and extract the MRI files into:

```text
data/cheng/
```

The original filenames and folder structure should be kept. The patient-level split and training subsets are already included in `data/`.

## Running the Experiments

The experiments were run on the University of Copenhagen GPU cluster using Slurm. The following commands should be submitted from the `carmen_vanderlans/` directory after setting up the dataset and models.

### Main experiments

Train LoRA and Adapter using the different training fractions and five random seeds:

```bash
sbatch scripts/run_v2_training.sh
```

After training has finished, evaluate the trained models:

```bash
sbatch scripts/run_v2_testing.sh
```

### Zero-shot baselines

```bash
sbatch scripts/run_mobilesam_test.sh
sbatch scripts/run_medsam_corrected.sh
```

### Patient-diversity experiments

For the 25% comparison:

```bash
sbatch scripts/run_diversity25.sh
```

After training has finished:

```bash
sbatch scripts/run_diversity25_eval.sh
```

The scripts for the 10% comparison are also included in `scripts/`.

### Additional experiments

LoRA rank comparison:

```bash
sbatch scripts/run_rank_experiments.sh
```

After training:

```bash
sbatch scripts/run_rank_testing.sh
```

Learning-rate comparison:

```bash
sbatch scripts/run_lr_extended.sh
```

Computational-cost comparison:

```bash
sbatch scripts/run_peft_benchmark.sh
```

The scripts use the university's Slurm configuration and may need to be adjusted for other computing environments.

## Results and Reproducibility

Segmentation performance is measured using Dice scores calculated for individual MRI slices and averaged at the patient level. The computational comparison considers trainable parameters, GPU memory usage and training time.

The `results/` folder contains the quantitative results used in the report, while `figures/` contains the segmentation performance plots and qualitative examples.

The patient-level split and training subsets for all five seeds are included. The original experiment code has been kept, with some file paths adjusted to fit the GitHub structure.

The repository does not include the MRI images or model checkpoints. Reproducing the experiments requires the Cheng dataset, the pretrained MobileSAM and MedSAM models, and a suitable GPU.