# Multimodal Brain Tumor Segmentation & Radiology Report Generation

A multimodal deep learning pipeline on the BraTS2020 dataset that jointly performs 
tumor segmentation and generates free-text radiology reports from FLAIR MRI scans.

## Notebooks
| Notebook | Description |
|---|---|
| `Team63_Original.ipynb` | Full pipeline — 2.5D preprocessing, segmentation, and report generation |
| `Team63_Optimised_text.ipynb` | Optimised version focused on report generation with patient-level CLIP encoding and beam search decoding |

## Overview
- 2.5D FLAIR MRI preprocessing — 3 consecutive axial slices stacked as channels, resized to 224×224
- CLIP ViT-B/16 vision encoder with chunked slice processing (8 slices/chunk) to manage VRAM
- Custom Transformer decoder (4 layers, 8 heads) with beam search for free-text radiology report generation
- Multi-class tumor segmentation across 4 sub-regions: ET, TC, WT, background
- MONAI spatial + intensity augmentations, weighted random sampling to handle tumor/non-tumor slice imbalance
- Mixed precision training with cosine annealing LR scheduler and warmup

## Tech Stack
`PyTorch` `CLIP` `MONAI` `HuggingFace Transformers` `nibabel` `scikit-learn` `scikit-image`

## Dataset
- [BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) — FLAIR MRI volumes + segmentation masks
- [TextBraTS](https://www.kaggle.com/datasets/pichu99/textbrats) — paired radiology reports

## Evaluation
| Task | Metrics |
|---|---|
| Report Generation | BLEU-1, BLEU-2, corpus BLEU, ROUGE-1, ROUGE-2, ROUGE-L |
| Segmentation | Dice Score, Hausdorff Distance |
