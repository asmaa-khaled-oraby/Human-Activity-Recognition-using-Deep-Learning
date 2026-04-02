# Human-Activity-Recognition-using-Deep-Learning
 
<div align="center">
 
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-4CAF50?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.40%25-2196F3?style=flat-square)
![Models](https://img.shields.io/badge/Models-3%20Architectures-9C27B0?style=flat-square)
 
**Classifying human physical activities from wearable IMU sensor data using three progressively more powerful deep learning architectures.**
 
[Results](#-results) • [Models](#-model-architectures) • [Setup](#-quick-start) • [Usage](#-usage) 
 
</div>
 
---
 
## 📋 Overview
 
This project builds an end-to-end pipeline for Human Activity Recognition (HAR) using raw inertial sensor data from smartphones. We evaluate three architectures of increasing expressiveness on the **UCI HAR benchmark**, demonstrating how temporal modelling progressively improves classification accuracy.
 
| Capability | Detail |
|---|---|
| **Task** | 6-class activity classification from 50 Hz IMU signals |
| **Input** | 2.56-second windows (128 samples × 18 channels) |
| **Best Accuracy** | **96.40%** (CNN-LSTM, Macro F1 = 0.9643) |
| **Dataset** | UCI HAR — 10,299 windows, 30 subjects |
| **Framework** | PyTorch 2.0+ |
| **Deployment** | Streamlit web app — upload CSV, get prediction |
 
### Recognised Activities
```
WALKING   WALK_UP   WALK_DOWN   SITTING   STANDING   LAYING
```
 
---
 
## 🏆 Results
 
### Overall Performance
 
| Model | Accuracy | Macro F1 | Macro Prec | Macro Rec | Params | Ep Time |
|---|---|---|---|---|---|---|
| CNN | 0.9484 | 0.9490 | — | — | 160,902 | 1.3 s |
| **CNN-LSTM** ⭐ | **0.9640** | **0.9643** | **0.9650** | **0.9644** | 5,087,110 | 3.2 s |
| CNN-LSTM+Attn | 0.9498 | 0.9493 | — | — | 2,199,942 | 2.3 s |
 
> **Winner: CNN-LSTM** with +1.56 pp accuracy gain over the CNN baseline.
 
### Model Progression Summary
 
```
CNN            0.9484  ──────────────────────────────────────
CNN-LSTM       0.9640  ████████████████████████████████████████ ← BEST  (+1.56 pp)
CNN-LSTM+Attn  0.9498  ────────────────────────────────────────
```
 
### Per-Class Performance — CNN-LSTM (Best Model)
 
| Activity | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| WALKING | 0.9960 | 1.0000 | **0.9980** | 496 |
| WALK_UP | 1.0000 | 0.9554 | 0.9772 | 471 |
| WALK_DOWN | 0.9589 | 1.0000 | 0.9790 | 420 |
| SITTING | 0.9393 | 0.8819 | 0.9097 | 491 |
| STANDING | 0.9050 | 0.9492 | 0.9266 | 532 |
| LAYING | 0.9908 | 1.0000 | 0.9954 | 537 |
| **MACRO** | **0.9650** | **0.9644** | **0.9643** | — |
 
 
---
 
## 🧠 Model Architectures
 
### 1 — CNN Baseline
A stack of three `Conv1d → BatchNorm1d → ReLU` blocks (64 → 128 → 256 filters) followed by Global Average Pooling. Captures local temporal patterns but **discards temporal ordering**.
 
```
Input (B,18,128) → ConvBlock×3 → MaxPool×2 → GAP → FC(256→128→6)
Parameters: 160,902
```
 
### 2 — CNN-LSTM ⭐ Best Model
A deep 5-block CNN backbone (64→128→256→256→512) feeds 32 compressed timesteps into a 2-layer LSTM (hidden size=512). The LSTM recurrently processes the sequence, maintaining a cell state that summarises temporal context across gait cycles and activity transitions. LayerNorm stabilises the output before the 3-layer GELU classifier.
 
```
Input (B,18,128) → CNN(5 blocks) → (B,512,32) → LSTM(2-layer, h=512) → h_n[-1] → FC → Output
Parameters: 5,087,110
```
 
**Why it wins:** Activities have inherent sequential structure — the LSTM's cell state integrates features across all 32 timesteps, capturing phase sequences (heel-strike → loading → push-off) and activity transitions that a CNN alone cannot model.
 
### 3 — CNN-LSTM + Bahdanau Attention
Adds attention over all 32 LSTM hidden states. Despite using 57% fewer parameters than CNN-LSTM, it scored slightly lower in this run — demonstrating that attention is not universally superior; the benefit depends on hidden capacity and training configuration.
 
```
Input → CNN → LSTM → Attention(32 states) → context vector → LayerNorm → FC → Output
Parameters: 2,199,942   |   Most robust to noise (−1.8 pp at σ=0.05 vs −3.4 pp for CNN-LSTM)
```
 
---
 
## 📦 Dataset
 
**UCI Human Activity Recognition Using Smartphones** — [Download here](https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset)
 
 
| Property | Value |
|---|---|
| Subjects | 30 volunteers, aged 19–48 |
| Device | Samsung Galaxy S II (waist-mounted) |
| Sensors | Accelerometer + Gyroscope (3 axes each = 9 channels) |
| Sampling rate | 50 Hz |
| Window | 128 samples (2.56 s), 50% overlap |
| Train split | 7,352 windows (21 subjects) |
| Test split | 2,947 windows (9 subjects) |
 
> The dataset downloads **automatically** on first run from the UCI ML Repository — no manual setup needed.
 
---
 
## ⚙️ Feature Engineering
 
### 1. Z-Score Normalisation
Per-channel normalisation fitted on training data only (no data leakage):
```python
x_norm = (x - μ_train) / (σ_train + 1e-8)
```
 
### 2. FFT Frequency-Domain Fusion
64 FFT magnitude bins (0.4–25 Hz) per channel concatenated with time-domain signals:
```
9 time channels  +  9 FFT channels  =  18-channel fused input  (128 × 18)
```
Walking produces spectral peaks at ~2 Hz; static activities produce near-flat spectra.
 
### 3. Data Augmentation (training only)
| Strategy | Probability | Effect |
|---|---|---|
| Gaussian noise (σ=0.03) | 50% | Simulates sensor calibration drift |
| Circular time shift (±10 samples) | 30% | Simulates window phase variation |
 
---
 
## 🚀 Quick Start
 
### Prerequisites
```bash
pip install torch numpy matplotlib seaborn scikit-learn reportlab streamlit
```
 
### Run on Google Colab (recommended)
1. Open `HAR_CNN-LSTM.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells (`Runtime → Run all`)
 
The dataset downloads automatically (~60 MB).
 
### Run locally
```bash
git clone https://github.com/asmaa-khaled-oraby/Human-Activity-Recognition-using-Deep-Learning.git
cd Human-Activity-Recognition-using-Deep-Learning
pip install -r requirements.txt
jupyter notebook HAR_CNN-LSTM.ipynb
```
 
---
 
## 📁 Project Structure
 
```
Human-Activity-Recognition-using-Deep-Learning/
│
├── HAR_CNN-LSTM.ipynb              # Main notebook — full pipeline
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── HAR_Report_1.pdf
│
└── Results/
    ├── 01_training_curves.png
    ├── 02_confusion_matrix.png
    ├── 03_model_comparison.png
    ├── 04_perclass_f1.png
    ├── 05_hidden_activations_colormap.png
    ├── 05b_perclass_prf.png        ← Per-class Precision / Recall / F1
    ├── 06_inference_confidence.png
    └── 07_robustness.png
```

---
 
## 🔄 Usage
 
### Training
All models train automatically when you run Cell 13:
```python
ALL = {}
for model_name, model_class in MODEL_ZOO.items():
    trained_model, history = train_model(model_class, model_name)
    ALL[model_name] = {'model': trained_model, 'hist': history}
```
 
Training uses:
- **AdamW** optimiser (lr=1e-3, weight_decay=1e-4)
- **CosineAnnealingLR** scheduler (T_max=60, eta_min=5e-5)
- **Early stopping** (patience=12)
- **Gradient clipping** (max_norm=1.0) — prevents LSTM instability
 
### Real-World Inference
```python
import numpy as np
 
# raw_window: numpy array of shape (128, 9) — raw un-normalised sensor data
result = predict_activity(raw_window)
 
print(result['predicted'])    # e.g. 'WALKING'
print(result['confidence'])   # e.g. 92.1
print(result['top_k'])        # e.g. [('WALKING', 92.1), ('WALK_UP', 4.2), ...]
print(result['latency_ms'])   # e.g. 2.7
```
 
**Sample output:**
```
✓  True: WALKING      Pred: WALKING      Conf:  92.1%  Lat: 2.78 ms
✓  True: LAYING       Pred: LAYING       Conf:  92.2%  Lat: 2.44 ms
✗  True: SITTING      Pred: STANDING     Conf:  74.7%  Lat: 2.40 ms
✓  True: WALK_DOWN    Pred: WALK_DOWN    Conf:  92.1%  Lat: 2.31 ms
 
Sample accuracy: 9/10   Avg latency ≈ 2.70 ms
```
 
### Best Model Selection
The best model is selected automatically after evaluation:
```python
BEST = max(ALL.keys(), key=lambda n: ALL[n]['metrics']['macro_f1'])
# → 'CNN-LSTM'
```

---
 
## 📊 Generated Visualizations
 
| Plot | Description |
|---|---|
| `01_training_curves.png` | Train/val loss & accuracy for all models |
| `02_confusion_matrix.png` | Confusion matrix of best model (raw + recall %) |
| `03_model_comparison.png` | Accuracy/F1, parameter count, epoch time |
| `04_perclass_f1.png` | Per-class F1 heatmap across all models |
| `05_hidden_activations.png` | LSTM hidden-state activations (8 sample windows) |
| `05b_perclass_prf.png` | **Per-class Precision / Recall / F1 bar chart** |
| `06_inference_confidence.png` | Confidence distributions for 3 sample windows |
| `07_robustness.png` | Accuracy under noise and channel dropout |
 
---
 
## 🛡️ Robustness
 
The best model (CNN-LSTM) was stress-tested on 300 random test windows:
 
| Condition | Accuracy | Drop | Assessment |
|---|---|---|---|
| No perturbation | 96.4% | — | Baseline |
| Noise σ = 0.05 (real-world) | 93.0% | −3.4 pp | Good |
| Noise σ = 0.10 | 84.7% | −11.7 pp | Moderate |
| Noise σ = 0.20 | 68.6% | −27.8 pp | Significant |
| 1 channel dropped | 79.3% | −17.1 pp | Moderate |
| 2 channels dropped | 71.4% | −25.0 pp | Significant |
| 3 channels dropped | 50.5% | −45.9 pp | Major failure |
 
> **Note:** CNN-LSTM+Attn is more robust to noise (−1.8 pp at σ=0.05) despite lower clean accuracy — a fundamental accuracy–robustness trade-off. For noisy deployment environments, CNN-LSTM+Attn may be preferred.
 
---
 
## 🔬 Key Findings
 
1. **Temporal modelling is essential** — CNN-LSTM outperforms the CNN baseline by +1.56 pp by capturing sequential structure that global average pooling discards.
 
2. **Attention ≠ always better** — CNN-LSTM+Attn scored 1.42 pp lower than CNN-LSTM in this run. Attention benefits depend on hidden capacity, training data size, and LR tuning.
 
3. **SITTING/STANDING is the hardest pair** — Both are static postures with near-zero acceleration variance. The only discriminant is device orientation angle (≈10–15°), easily corrupted by inter-subject variability.
 
4. **FFT fusion is impactful** — Doubling input channels from 9 (time) to 18 (time + frequency) provides direct access to the spectral periodicity distinguishing dynamic from static activities.
 
5. **Inference is fast** — Average latency of **2.7 ms** on CPU enables real-time deployment on mobile devices.
 
---
 
<div align="center">
 
Made with PyTorch · UCI HAR Dataset · Streamlit · Google Colab
 
</div>
