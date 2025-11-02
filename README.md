# 🧠 CLIPFusion: Multimodal Price Prediction using CLIP + Deep Fusion

CLIPFusion is an advanced **multimodal deep learning model** designed to predict product prices by combining **text, image, and tabular data**.  
It leverages **CLIP ViT-B/16** embeddings with a **Deep Fusion Regressor** that integrates contextual, visual, and numeric information for high-accuracy e-commerce predictions.

---

## 🚀 Key Features
- 🖼️ **CLIP-powered embeddings** (ViT-B/16) for both text and images.
- ⚙️ **Deep fusion** of text, image, and tabular data.
- 💾 **Smart caching** for CLIP embeddings.
- ⚡ **GPU-accelerated training** with mixed precision (AMP).
- 📊 **Visualization & Evaluation**: loss curves and bin-wise SMAPE.
- 🌲 **Optional LightGBM stacking** for ensemble refinement.

---

## 🧩 Model Overview

| Component | Description |
|------------|-------------|
| **Text Encoder** | CLIP Transformer text encoder (ViT-B/16) |
| **Image Encoder** | CLIP Vision Transformer backbone |
| **Tabular Encoder** | MLP processing value + has_image |
| **Fusion Layer** | Deep concatenation with normalization & dropout |
| **Regressor** | Dense layer predicting log(price) |
| **Loss Function** | Huber Loss |
| **Metric** | SMAPE (Symmetric Mean Absolute Percentage Error) |

---

## 📊 Performance Summary

| Metric | Value |
|---------|-------|
| **Best Validation SMAPE** | **0.4792** |
| **Epochs** | 40 |
| **Model** | CLIP ViT-B/16 + Deep Fusion |
| **Hardware Used** | RTX 3050 (4GB VRAM) + 22GB RAM |
| **Dataset** | 75K text–image–tabular samples |

📉 Training curves and bin-wise SMAPE visualizations are saved under `artifacts/`.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/CLIPFusion.git
cd CLIPFusion
