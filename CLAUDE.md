# PhytoVision — Phytoplankton Classification

## Project Summary
EfficientNetV2-L classifier for 8-10 marine phytoplankton classes from FlowCAM microscopy.
Current model: v9 · 60.62% val accuracy · 8 classes · 12,315 images

## Google Drive Paths (Colab)
- Library 1: /content/drive/MyDrive/Phytoplankton_Classification/Phytoplankton_Data/Library/
- Library 2: /content/drive/MyDrive/Phytoplankton_Classification/Phytoplankton_Data/Library-2/
- FlowCAM CSVs: /content/drive/MyDrive/BLOOFINZ-FlowCAM-2022/Bloofin-2022-Discrete/BFZ-Cast*/
- Model outputs: /content/drive/MyDrive/Phytoplankton_Deploy/V9/

## Classes (8 active)
Crocosphaera, Cyanobacteria, Mineral particles, Nanophytoplankton,
Nitzschia, Picocyanobacteria, Prochlorococcus, Synechococcus

## Stack
TensorFlow 2.19, Keras, Python, Google Colab A100, mixed FP16

## Key files
- Phytoplankton_Classification_V9.ipynb — main training notebook
- phytovision-v2.html — deployment dashboard