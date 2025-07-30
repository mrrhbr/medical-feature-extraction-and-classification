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


##  The Architecture


### 1. Phase 1 – Visual Feature Embedding + LRF  
Notebook: `MAVL_Phase1_FeatureEmbeding+LRF+baseline.ipynb`

- Loads a pretrained ViT model using the `timm` library  
- Applies a low-rank projection layer on visual features  
- Implements LRF-style loss function to regularize training  
- Trains a multi-label classifier  
- Saves low-dimensional visual embeddings for use in Phase 3


```python
# Define the model
model = LRFLModel(backbone_name="vit_base_patch16_224", rank=64, num_classes=5)

# Low-Rank projection inside the model
self.low_rank_proj = nn.Sequential(
    nn.LayerNorm(hidden_dim),
    nn.Linear(hidden_dim, rank, bias=False),
    nn.ReLU()
)

# Loss function (custom LRF)
def lrfl_loss_fn(logits, labels, features, U, V, eta=1e-3):
    bce = nn.BCEWithLogitsLoss()(logits, labels)
    reg = torch.sum((U.T @ features) @ V.T)
    return bce + eta * reg / features.size(0)

### 2. Phase 2 – Text Embedding with ClinicalBERT  
Notebook: `MAVL_Phase2_TextEmbedding.ipynb`

- Loads pretrained `emilyalsentzer/Bio_ClinicalBERT` via HuggingFace  
- Encodes label descriptions (e.g., “Pneumothorax”, “Cardiomegaly”) into dense vectors  
- Aggregates hidden states (mean pooling or `[CLS]`) to produce fixed-size embeddings  
- Saves text embeddings for use in the fusion/classification stage



```python
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Example label list
label_texts = ["Pneumothorax", "Cardiomegaly", "Effusion", ...]

# Tokenize and encode
inputs = tokenizer(label_texts, padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings using mean pooling
text_embeddings = outputs.last_hidden_state.mean(dim=1)  # shape: [num_labels, hidden_dim]

