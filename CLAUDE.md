# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**PhytoVision** — EfficientNetV2-L image classifier for marine phytoplankton from FlowCAM microscopy images. Training runs on Google Colab (A100 GPU) with data stored on Google Drive.

- Current model: **v9** · 60.62% val accuracy · 8 active classes · 12,315 images
- Stack: TensorFlow 2.19, Keras, Python, mixed FP16 (`mixed_float16` policy)

## Repository files

- `Phytoplankton_Classification_V9.ipynb` — main training notebook (run in Colab)
- `phytovision-v2.html` — standalone deployment dashboard (open in browser; uses Chart.js, no build step)

## Google Drive paths (Colab runtime)

| Resource | Path |
|---|---|
| Library 1 | `/content/drive/MyDrive/Phytoplankton_Data/Library/` |
| Library 2 | `/content/drive/MyDrive/Phytoplankton_Data/Library-2/` |
| FlowCAM CSVs | `/content/drive/MyDrive/BLOOFINZ-FlowCAM-2022/Bloofin-2022-Discrete/BFZ-Cast*/` |
| Model outputs | `/content/drive/MyDrive/Phytoplankton_Deploy/V9/` |

Deploy artifacts saved to `Phytoplankton_Deploy/V9/`: `v9_infer.keras`, `class_names.json`, `val_metrics.json`.

## Active classes (8)

`Crocosphaera`, `Cyanobacteria`, `Mineral particles`, `Nanophytoplankton`, `Nitzschia`, `Picocyanobacteria`, `Prochlorococcus`, `Synechococcus`

(Library-2 adds `Pennate Diatoms` and `Richelia`, bringing the combined dataset to 10 classes / 12,315 images.)

## Training architecture

Two-stage transfer learning from ImageNet weights:

1. **Stage 1 — head only** (backbone frozen): EfficientNetV2-L backbone → GAP → BN → Dropout(0.4) → Dense(512, swish) → Dropout(0.3) → Dense(8, logits). Optimizer: AdamW + CosineDecay (initial LR 2e-4). Loss: `SparseCategoricalCrossentropy(from_logits=True)`.
2. **Stage 2 — gentle fine-tune**: unfreeze top ~5% of backbone layers, keep all BatchNorm frozen, AdamW at 5e-6.

**Class imbalance handling:** balanced sampling via `tf.data.Dataset.sample_from_datasets` with sqrt-inverse-frequency weights mixed 25/75 with the natural distribution. No `class_weight` dict (intentionally omitted for stability with balanced batches).

**Image size:** 384×384. Augmentation: random flip, rotation (±10°), zoom (±10%), translation (±6%).

**Inference model** wraps logits in a `Softmax` layer before saving (so `v9_infer.keras` outputs probabilities directly).

## Key design decisions to preserve

- Output layer uses `dtype="float32"` explicitly (mixed FP16 stability).
- Validation dataset is *not* balanced — uses natural class distribution to reflect real-world performance.
- `ModelCheckpoint` saves `best.keras` (monitors `val_acc`, `mode="max"`).
- `EarlyStopping` patience=6 on `val_acc`.
