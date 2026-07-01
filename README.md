<div align="center">
<h1>MuCoNet</h1>
<h3>MuCoNet: Multiscale Context Grouping and Detail Compensation with Global Modulation for Polyp Segmentation</h3>
</div>

This repository provides training, inference, and evaluation code for **MuCoNet**. Follow the steps below to reproduce the results reported in the paper.

---

## Table of Contents

- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Pretrained Weights](#pretrained-weights)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Model Complexity](#model-complexity)
- [Reproducing Paper Numbers](#reproducing-paper-numbers)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Requirements

- Linux with NVIDIA GPU
- **Python 3.10**
- **CUDA 11.8+**
- **PyTorch 2.1.1**, **torchvision 0.16.1**
- **mamba-ssm 1.2.0.post1**, **causal-conv1d 1.1.1**
- **mmcv >= 2.0.0**

---

## Environment Setup

```bash
conda create -n muconet python=3.10 -y
conda activate muconet
cd MuCoNet

# 1) Install PyTorch (adjust CUDA version if needed)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

# 2) Install Mamba dependencies (order matters)
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.2.0.post1

# 3) Build the local selective-scan CUDA extension (recommended for speed)
cd lib/vmamba/kernels/selective_scan
pip install .
cd ../../../..

# 4) Install remaining dependencies
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "import torch; from lib.model import MuCoNet; print('OK', torch.__version__)"
```

If `selective_scan_cuda` import fails, ensure step 2 (`mamba-ssm`) completed successfully. Step 3 is optional but improves training speed.

---

## Pretrained Weights

MuCoNet uses **PVTv2-B2** as the encoder backbone. Download the ImageNet pretrained weights and place them at:

```
lib/pvt_v2_b2.pth
```

Download link (official PVT release):

- https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth

The model loads this file automatically at initialization (`lib/model.py`).

---

## Dataset Preparation

We follow the **PraNet** train/test split protocol:

- **Training set (1450 images):** 900 from Kvasir-SEG + 550 from CVC-ClinicDB
- **Test sets:** remaining Kvasir / ClinicDB images + full CVC-300, CVC-ColonDB, ETIS-LaribPolypDB

### Download

| Split | Link |
|-------|------|
| Train | [Google Drive](https://drive.google.com/drive/folders/1NVEDXDeIvKHw55dOnL6CbbbsiWrg41FH?usp=drive_link) |
| Test  | [Google Drive](https://drive.google.com/drive/folders/12i58jDzDGE8MiQ-QxPxiltbX8GkzwaG4?usp=drive_link) |

### Expected Directory Layout

```
MuCoNet/
├── data/
│   ├── TrainDataset/
│   │   ├── images/
│   │   └── masks/
│   └── TestDataset/
│       ├── Kvasir/
│       │   ├── images/
│       │   └── masks/
│       ├── CVC-ClinicDB/
│       ├── CVC-300/
│       ├── CVC-ColonDB/
│       └── ETIS-LaribPolypDB/
├── data/checkpoint/          # created during training
├── result/                   # created during testing
└── logs/                     # training logs
```

Each test subset must contain paired `images/` and `masks/` folders. Image and mask filenames must match (except extension).

---

## Training

### Paper Settings

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | **24** |
| Epochs | 200 |
| Input size | 352 × 352 |
| LR schedule | ×0.1 every 30 epochs |
| Multi-scale training | 0.75, 1.0, 1.25 |
| Loss | Structure loss on 3 decoder outputs (equal weight) |
| Data augmentation | Off by default (`--augmentation False`) |

### Run Training

```bash
# Paper configuration (batch size 24; requires sufficient GPU memory)
python train.py --batchsize 24

# Custom paths (defaults are ./data/TrainDataset and ./data/TestDataset)
python train.py \
  --batchsize 24 \
  --epoch 200 \
  --lr 1e-4 \
  --trainsize 352 \
  --train_path ./data/TrainDataset \
  --test_path ./data/TestDataset \
  --train_save ./data/checkpoint/
```

**Hardware notes:**

- `train.py` uses `DataParallel` on GPUs `[0, 1]` by default. For a **single GPU**, edit `device_ids = [0, 1]` to `device_ids = [0]` in `train.py`, or set `CUDA_VISIBLE_DEVICES=0`.
- If OOM occurs with batch 24, reduce `--batchsize` (e.g., 16) and scale learning rate proportionally, or use gradient accumulation (not implemented in this repo).

### Checkpoint Saving Strategy

During training, the model is validated on all five test sets **every epoch**. Checkpoints are saved under `./data/checkpoint/`:

| Filename | Condition |
|----------|-----------|
| `train_doublebest_{dataset}.pth` | Both mDice **and** mIoU improve on `{dataset}` |
| `train_best_iou_{dataset}.pth` | Only mIoU improves |
| `train_best_dice_{dataset}.pth` | Only mDice improves |

`test.py` uses **`train_doublebest_{dataset}.pth`** for each dataset. Training logs are written to `logs/train_logger.txt`.

---

## Testing

Generate predicted masks for all five test sets:

```bash
python test.py
```

**Outputs:**

```
result/
├── CVC-300/
├── CVC-ClinicDB/
├── CVC-ColonDB/
├── ETIS-LaribPolypDB/
└── Kvasir/
```

Each folder contains grayscale `.png` prediction maps (0–255), resized to the original GT resolution. Predictions are min–max normalized before saving, matching the PraNet evaluation protocol.

**Requirements:**

- Checkpoints must exist at `./data/checkpoint/train_doublebest_{dataset}.pth` for all five datasets.
- `test.py` runs on `cuda:0` by default. Edit `device_ids` if needed.

---

## Evaluation

We use six standard polyp-segmentation metrics: **mDice**, **mIoU**, **wFβ**, **Sα**, **maxEξ**, **MAE**.

### Run Evaluation

```bash
python eval.py --config configs/eval.yaml --verbose
```

The config file (`configs/eval.yaml`) specifies:

- `pred_root`: `./result` (predictions from `test.py`)
- `gt_root`: `./data/TestDataset`
- `result_path`: `./result/metrics` (CSV outputs)

Per-dataset CSV files are saved as `result/metrics/result_{dataset}.csv`. The script prints a summary table to stdout when `--verbose` is set.

---

## Model Complexity

Compute Params and FLOPs (input size 352×352):

```bash
python caculate_ability.py
```

Expected output (approximate, from paper):

- **Params:** ~28.6M
- **FLOPs:** ~14.2G

---

## Reproducing Paper Numbers

End-to-end workflow:

```bash
# 1. Environment + PVT weights + datasets (see sections above)

# 2. Train (or place provided checkpoints in ./data/checkpoint/)
python train.py --batchsize 24

# 3. Inference
python test.py

# 4. Metrics
python eval.py --config configs/eval.yaml --verbose
```

### Important Notes

1. **Random split:** Training uses the PraNet 1450-image subset. Ensure you use the same Train/Test packages linked above; do not mix different splits.
2. **Per-dataset checkpoints:** Paper results are obtained by loading the best checkpoint **per test dataset** (`train_doublebest_*.pth`), not a single shared model. This matches the validation logic in `train.py`.
3. **Batch size:** Code default is `--batchsize 16`; the paper reports **24**. Use `--batchsize 24` for exact reproduction.
4. **Seeds:** Training does not fix random seeds by default. Minor metric variance across runs is expected. For strict reproducibility, set seeds for `random`, `numpy`, and `torch` at the top of `train.py`.
5. **Baselines:** Comparison methods (PraNet, SANet, ASPS, etc.) are evaluated with their official code under the same data split; see supplementary materials / comparison repo if provided separately.

---

## Project Structure

```
MuCoNet/
├── train.py              # Training script
├── test.py               # Inference on 5 test sets
├── eval.py               # Metric computation
├── caculate_ability.py   # Params / FLOPs
├── configs/
│   └── eval.yaml         # Evaluation config
├── lib/
│   ├── model.py          # MuCoNet architecture
│   ├── pvtv2.py          # PVTv2 backbone
│   ├── pvt_v2_b2.pth     # Download separately
│   └── vmamba/           # GPSSM / SS2D modules
├── utils/
│   ├── dataloader.py
│   ├── eval_functions.py
│   └── utils.py
├── data/                 # Datasets & checkpoints (not in git)
├── result/               # Predictions & metrics
└── logs/                 # Training logs
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{muconet2025,
  title   = {MuCoNet: Multiscale Context Grouping and Detail Compensation with Global Modulation for Polyp Segmentation},
  author  = {},
  journal = {},
  year    = {2026}
}
```

*(Update author/journal fields upon publication.)*

---

## License

This project is released for academic research purposes. Please refer to the license file and respect the licenses of third-party dependencies (PVT, Mamba, PraNet dataset protocol).
