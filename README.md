# 🔥 Identifying Fires from Aerial Footage

Machine learning pipeline for wildfire detection using aerial imagery, focusing on **data leakage mitigation, lightweight CNN design, and real-world deployment trade-offs**.

📄 Full report: see project PDF

---

## Overview

Wildfires are a growing global threat. Early detection is critical.

This project develops **lightweight deep learning models** to classify aerial images into:
- Fire  
- No Fire  
- Lake (ternary extension)

Key challenge:  
Video-derived datasets introduce **temporal data leakage**, leading to misleading validation performance.

---

## Key Results

### Binary classifier (final system)
- Accuracy: **86%**
- Fire Recall: **97.1%**
- Missed fires reduced by **86%**

Outperforms FLAME baseline by ~10 percentage points.

### Ternary classifier
- Accuracy: **73.8%**
- Lake Recall: **100%**

---

## Dataset

- **FLAME dataset (aerial wildfire imagery)**
- Train: 39,375 frames (Zenmuse X4S)
- Test: 8,617 frames (Phantom 3)
- Image size: 254 × 254

### Core issue
- Consecutive frames are highly correlated  
- Causes **data leakage + overestimated validation performance**

---

## Data Pipeline

### 1. De-duplication
- Perceptual hashing (`dhash`)
- Hamming distance threshold: 4
- Reduced dataset: **39,375 → ~2,979 images**

### 2. Train / Validation Split
- Temporal block splitting
- Prevents frame overlap between splits

### 3. Augmentation
- Random flip
- Brightness + contrast variation

### 4. Class Handling
- Class weights for imbalance
- Lake class treated separately

---

## Model Architecture

### Lightweight Micro-CNN (~11k parameters)

Designed for:
- CPU training
- Low compute environments
- Reduced overfitting

#### Structure
- Conv stem (3×3)
- 4 depthwise separable blocks
- Global Average Pooling
- Sigmoid / Softmax output

#### Key design choices
- Depthwise separable convolutions → ~7× fewer parameters  
- Increasing channels as spatial resolution decreases  
- Dropout increases with depth  

---

## Training

- Optimizer: Adam (lr = 1e-3)
- Loss:
  - Binary: Focal Loss (γ = 2)
  - Ternary: Cross-entropy
- Regularisation:
  - Dropout
  - Class weights
- Callbacks:
  - Early stopping
  - ReduceLROnPlateau

---

## Evaluation

Metrics:
- AUC (primary)
- Accuracy
- Precision / Recall

### Key insight
AUC can look strong even when real-world performance is poor.  
Threshold selection is critical.

---

## Post-processing

### Threshold tuning
- Default: 0.5  
- Optimal: **0.34**

### Temporal filter
- Require **N = 3 consecutive detections**

### Impact
- Fire Recall: **78.6% → 97.1%**
- Missed fires dramatically reduced

---

## Binary vs Ternary Insight

- Binary model performance was partially inflated  
- Lake images made "No Fire" easier  
- Ternary model shows true difficulty:
  - **Fire vs No Fire boundary is the core challenge**

---

## Limitations

- Distribution shift (train vs test cameras)
- Limited No Fire diversity
- Small validation set → unstable threshold tuning

---

## Future Work

- More diverse No Fire data
- Transfer learning (MobileNet / EfficientNet)
- Extend temporal filtering to ternary model
- Improve probability calibration

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- imagehash (dhash)
- Matplotlib / Seaborn

---

## How to Run

```bash
# clone repo
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

# install dependencies
pip install -r requirements.txt

# train model and evaluate 
python model.py



