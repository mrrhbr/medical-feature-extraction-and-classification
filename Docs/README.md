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
```


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




## Phase 3 – Multi-label Disease Classification with Contrastive Pretraining

**Notebook**: `MultiAspect_VisionLanguage_Model_final_version_full_data_+contrastive.ipynb`




### Pipeline Summary

1. **Input Embeddings**:
   - Image embeddings: ViT pooled features → `[B, 64]`
   - Text embeddings: BERT-based (disease-specific) → `[7, 768]`
2. **Projection Heads**:
   - Image: `64 → 128 → 256`
   - Text: `768 → 256 → 256`
3. **Contrastive Alignment**:
   - Contrastive loss applied on positive image–text pairs
4. **Multi-label Classification**:
   - Cosine similarity scores → trained with `BCEWithLogitsLoss`



###  Contrastive Pretraining Details

- **Objective**: Align image embeddings with corresponding disease descriptions
- **Positive Pair**: `(image, mean of positive label embeddings)`
- **Loss Function**:

```python
logits = (img_emb @ txt_emb.T) / temperature
targets = torch.arange(B).to(device)
loss = (cross_entropy(logits, targets) + cross_entropy(logits.T, targets)) / 2

```

### Training Configuration
- **Temperature:** 0.2  
- **Batch size:** 64  
- **Epochs:** 30  
- **Optimizer:** Adam (lr=5e-4)  

---

## 🧮 Multi-label Classification Phase
- After contrastive alignment, cosine similarities between the image and each disease embedding are interpreted as logits.  
- A sigmoid function is applied to convert logits into per-label probabilities.  
- **Loss:** `BCEWithLogitsLoss`  
- **Threshold Tuning:** Sweep thresholds from 0.1 to 0.9 with a step of 0.05 to maximize macro-F1 score.  

---

##  Evaluation Results (Post-Contrastive)

| Disease          | Precision | Recall | F1   | Support |
|------------------|-----------|--------|------|---------|
| Atelectasis      | 0.80      | 0.83   | 0.81 | 80      |
| Cardiomegaly     | 0.75      | 0.72   | 0.73 | 68      |
| Consolidation    | 0.70      | 0.88   | 0.78 | 33      |
| Edema            | 0.63      | 0.56   | 0.59 | 45      |
| Pleural Effusion | 0.84      | 0.82   | 0.83 | 67      |
| Pneumonia        | 0.30      | 0.88   | 0.45 | 8       |
| Pneumothorax     | 0.10      | 0.25   | 0.14 | 8       |

- **Macro F1:** 0.62  
- **Micro F1:** 0.68  
- **Hamming Loss:** 0.1210  

---

##  Notes
-  Contrastive pretraining improved semantic alignment between vision and language.  
-  Enhanced generalization on frequent classes (e.g., Atelectasis, Pleural Effusion).  
-  Poor performance on underrepresented diseases (e.g., Pneumothorax).  
-  Cosine similarity combined with BCE loss works well for multi-label classification.  
-  Threshold sweeping is critical for calibration.  

---


## Dataset
- **Name:** CheXpert (small)  
- **Source:** [Kaggle - CheXpert-v1.0-small Chest X-rays](https://www.kaggle.com/datasets/ashery/chexpert)
- **Size:** ~11 GB  
- **Type:** Frontal chest X-rays with multi-label annotations  
- **Subset Used:** ~30% due to compute constraints
---
