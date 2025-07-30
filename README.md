# Multi-Aspect Vision-Language (MAVL) for Medical Diagnosis

This project implements a multi-stage framework inspired by the following papers:

- [**LRF: Towards Low-Rank Feedback for Multi-Label Medical Image Classification**](https://arxiv.org/abs/2404.18933)
- [**Multi-Aspect Medical Image Classification with Hierarchical Prompting**](https://arxiv.org/abs/2403.07636)

The goal is to enhance medical image classification using a combination of low-rank feature learning, text-guided supervision, and multi-aspect vision-language fusion.

---

##  Project Structure

```bash
.
├── MAVL_Phase1_FeatureEmbeding+LRF+baseline.ipynb   # ViT + Low-Rank Feature Learning (LRFL)
├── MAVL_Phase2_TextEmbedding.ipynb                  # Text embeddings using ClinicalBERT
├── MultiAspect_VisionLanguage_Model.ipynb           # Vision-Language Fusion and Classification

```
---


##  Usage

To run the project, follow the order below:

### 1. Phase 1 – Visual Feature Embedding + LRF

Notebook: `MAVL_Phase1_FeatureEmbeding+LRF+baseline.ipynb`

- Loads pretrained ViT model (via `timm`)
- Applies a low-rank projection to visual features
- Trains the classifier with LRF-style feedback loss
- Saves image embeddings

```python
model = LRFLModel(backbone_name="vit_base_patch16_224", rank=64, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
