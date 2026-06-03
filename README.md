# U-Net & Attention U-Net for Multi-Class Pet Image Segmentation and Classification

A multi-task deep learning pipeline combining **Base U-Net** and **Attention U-Net** for joint image segmentation and breed classification on the **Oxford-IIIT Pet dataset**, implemented from scratch in **PyTorch**.

## Overview

This project explores how attention mechanisms in encoder–decoder architectures affect segmentation quality, while sharing an encoder backbone with a 37-class breed classification head. Several ablations are included: binary vs 3-class segmentation, augmentation on/off, learning-rate sweeps, alternative classifier heads (MobileNetV2, EfficientNet), and an EfficientDet-inspired decoder variant.

## Key Features

- **Two architectures from scratch**: Base U-Net (multi-task) and Attention U-Net with attention gates
- **Joint optimization**: pixel-level segmentation + image-level 37-class breed classification
- **Custom loss**: BCE + Dice for segmentation, label-smoothed Cross-Entropy for classification
- **Metrics**: mIoU, Dice coefficient, pixel accuracy, classification accuracy
- **Mixed-precision training** with AdamW optimizer and cosine schedules
- **Ablation studies**: augmentation, learning rate, decoder variants, classifier heads

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds (cats + dogs), trimap segmentation masks (pet / border / background). Downloaded automatically via `torchvision.datasets.OxfordIIITPet`.

- **Train / Val / Test split**: stratified by breed label (72% / 13% / 15%)
- **Image size**: 128×128 (resized)
- **Normalization**: ImageNet mean/std

## Architectures

### Base U-Net V2 (Multi-Task)
Encoder–decoder with skip connections. Encoder features feed a global-pooled classification head (37 classes); decoder produces the binary segmentation mask.

### Attention U-Net V2
Adds **attention gates** at each skip connection so the decoder learns to focus on relevant spatial regions before concatenation with encoder features.

```
g (gating signal from decoder) ─┐
                                ├─► Attention Block ─► gated skip
x (encoder skip features)    ─┘
```

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 8e-5 (tuned) |
| Weight decay | 3e-4 |
| Batch size | 32 |
| Epochs | 25 (with early stopping) |
| Segmentation loss | BCE + Dice |
| Classification loss | CE with label smoothing (0.03) |
| Classification weight | 2.0 |
| Augmentation | Random flip, rotation, color jitter, resized crop |

## Experiments & Ablations

- **Binary vs 3-class segmentation** (pet vs background, then pet / border / background)
- **Augmentation on/off** — quantifies overfitting reduction
- **Learning-rate sweep** — 1e-3 / 5e-4 / 1e-4 / 5e-5 / 1e-5
- **Classifier head comparison** — custom head vs MobileNetV2 vs EfficientNet
- **EfficientDet-inspired decoder** — BiFPN-style multi-scale fusion variant

## File Structure

```
.
├── Building_U-Net_&_Attention_Net.ipynb   # Main notebook (training + evaluation + ablations)
├── README.md
├── requirements.txt
└── results/
    ├── predictions/                       # Sample qualitative outputs
    └── metrics_comparison.csv             # Final comparison table
```

## Installation

```bash
git clone https://github.com/Sadman-Rahman25/Pet-Segmentation-UNet-AttentionUNet.git
cd Pet-Segmentation-UNet-AttentionUNet
pip install -r requirements.txt
```

## Usage

Open the notebook and run cells sequentially:
```bash
jupyter notebook Building_U-Net_&_Attention_Net.ipynb
```

The notebook downloads the dataset, trains both models, runs ablations, and outputs comparison tables and qualitative prediction grids.

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy
pandas
matplotlib
scikit-learn
pillow
tqdm
torchinfo
```

## Future Work

- Replace ImageNet pretraining with self-supervised pretraining (DINO / MAE)
- Extend to instance segmentation
- Deploy as a web demo via Streamlit / Gradio

## Acknowledgements

- Oxford-IIIT Pet Dataset — Parkhi et al., CVPR 2012
- U-Net — Ronneberger et al., MICCAI 2015
- Attention U-Net — Oktay et al., 2018

## Author

**S. M. Sadman Rahman** — [GitHub](https://github.com/Sadman-Rahman25) · [LinkedIn](https://www.linkedin.com/in/sadman-rahman-198a88284)

## License

MIT License — see `LICENSE` file for details.
