# 🧬 HelixNet

> **Zero-dependency protein structure prediction suite** — paste a sequence, get a full 3D model in seconds.

HelixNet turns any raw amino acid sequence into predicted 3D protein structures using an ensemble of six battle-tested models. It compares classical and deep learning approaches for Q3 secondary structure prediction, then goes further with a multi-task engine that simultaneously outputs Q8 secondary structure, φ/ψ torsion angles, and precise 3D Cα coordinates — all in one forward pass.

---

## 🗂️ Repository Structure

```
HelixNet/
├── CB513_DSSP/                  # DSSP annotation files for CB513 dataset
├── CB513_FASTA/                 # FASTA sequence files for CB513 dataset
├── Final working codes/         # Cleaned, final versions of all model scripts
├── Logistic_FeedforwardNN.py    # Logistic Regression + Feedforward NN (NumPy)
├── Preprocessing.py             # Feature engineering pipeline
├── Parse_FASTA_DSSP_PDP.py      # Parser for FASTA, DSSP, and PDB files
├── SlidingWindow_PSSM.py        # Sliding window + PSSM feature builder
├── cnn.py                       # Deep CNN model
├── Rnn.py                       # Pure RNN model
├── Q8_hybrid_ml_libraries.py    # CNN-GRU-Attention multi-task engine (Q8 + 3D)
├── fetch_protein_sequence.py    # Utility to fetch protein sequences
└── README.md
```

---

## ⚙️ Preprocessing Pipeline

Every valid protein is enriched with **5 residue-level features** before model input:

| Feature | Description |
|---|---|
| One-hot encoding | 20-dimensional amino acid identity vector |
| Hydrophobicity | Kyte-Doolittle scale values |
| Charge | ±1 per residue |
| Polarity | Binary polarity flag |
| 3D coordinates | Normalized Cα coordinates (mean=0, std=1) |

- `build_window_features()` constructs **7-mer context vectors** (161–182 dim) as sliding windows
- Only `valid=True` rows are processed — safe with full logging to `add_features.log`
- Auto-matches `protein_id` → injects real/normalized 3D coordinates

---

## 🤖 Models

### 1. Logistic Regression

Pure NumPy multiclass logistic regression for Q3 secondary structure prediction (Helix / Strand / Coil).

| Parameter | Value |
|---|---|
| Input dim | 20 (one-hot) |
| Output dim | 3 classes (H, E, C) |
| Epochs | 200 |
| Batch size | 1024 |
| Learning rate | 0.01 |
| Loss | Categorical Cross-Entropy |
| **Q3 Accuracy** | **~0.48–0.52** |

---

### 2. Feedforward Neural Network

Fully NumPy-based deep model with manually implemented backpropagation.

| Parameter | Value |
|---|---|
| Architecture | 20 → 64 → 32 → 3 |
| Activation | ReLU (hidden), Softmax (output) |
| Epochs | 200 |
| Batch size | 1024 |
| Learning rate | 0.01 |
| **Q3 Accuracy** | **~0.58–0.62** |

---

### 3. CNN

Predicts dominant secondary structure fold for an entire protein using parallel multi-scale convolutions.

- **Input:** 300 × 7 sliding windows, 21-token vocabulary → 20-dim trainable embeddings
- **Architecture:** 4 parallel Conv1D layers (kernels 3, 5, 7, 9; 128 filters each) → ReLU → MaxPool(2) → Concat → FC(2560 → 128 → 3)
- **Optimizer:** Adam (lr = 5×10⁻⁴) | Epochs: 50 | Batch: 32
- **Split:** Stratified 80/10/10 (no protein leakage)
- **Train Accuracy: 60%**

---

### 4. RNN

Pure PyTorch 2-layer RNN for protein-level Q3 classification.

- **Input:** Fixed 150 residues (center-crop + pad), Skip-Gram embeddings (20 AA → 32-dim), 7-mer sliding windows → 95-dim/position
- **Added features:** Hydrophobicity, Charge, Polarity, Helix/Sheet/Turn propensities; IQR normalization + clip[-4, 4]
- **Architecture:** 2-layer RNN (tanh), Dropout 0.25 between layers, last hidden state → Linear → 3 classes
- **Training:** Adam lr=1.5×10⁻³, weight_decay=3×10⁻³, StepLR γ=0.5 every 30 epochs, Batch=16, grad clip=5.0, early stopping (patience=30)
- **Train Accuracy: 65%**

---

### 5. CNN-RNN Hybrid *(Current Best for Q3)*

Combines convolutional feature extraction with sequential RNN modeling.

- **Input:** Fixed 150 positions, 7-mer sliding windows → shape (150, 7), 64-dim trainable embeddings
- **Architecture:**
  ```
  Embedding (64d)
       ↓
  7-mer windows → flatten → Conv1D (k=3, 5, 7)
       ↓ 128 filters each → ReLU → MaxPool×2
       ↓ concat → 384-dim features
       ↓
  RNN (tanh, 512 hidden) → last hidden state
       ↓
  512 → 128 (ReLU) → 3 classes
  ```
- **Training:** Adam lr=3×10⁻⁴ | 150 epochs | Batch 32 | CUDA-ready | 80/10/10 stratified split
- **🏆 Test Accuracy: 75.2%**

---

### 6. CNN-GRU-Attention Multi-task Engine *(Q8 + Torsion + 3D)*

A single model that simultaneously predicts **8-class secondary structure**, **φ/ψ torsion angles**, and **3D Cα coordinates** in one forward pass.

- **Task:** Q8 per-residue classification + torsion angle regression + 3D coordinate regression
- **Input:** Fixed 150 residues, 7-mer sliding windows → (150, 7), 64-dim learned embeddings
- **Architecture:**
  ```
  Embedding → 3× Conv1D (k=3, pad=1) → 128 filters
  → Bi-GRU 256 → 8-head Multi-head Attention
  → LayerNorm + Residual connections
  → Q8 head (8 classes)
  → Torsion head (150, 2)
  → Coord head (150, 3)
  ```
- **Training:** AdamW 3×10⁻⁴ + weight_decay 1×10⁻⁵ | ReduceLROnPlateau + GradClip 1.0 | Multi-task loss: CE + MSE + RMSD | Batch 8 | 150 epochs | 70/15/15 split | Full checkpoint auto-saved
- **Performance:**

| Metric | Result |
|---|---|
| Q8 Accuracy | 68.4% |
| Torsion MAE | 18.7° |
| RMSD | 4.21 Å ✅ (target: < 5.0 Å) |

**Deliverables:** `best_working_model.pth` (2.4M params) · Training curves (4 plots) · Q8 confusion matrix

---

## 📊 Model Comparison (Q3 Accuracy)

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~48–52% |
| Feedforward NN | ~58–62% |
| CNN | ~60% |
| RNN | ~65% |
| **CNN-RNN Hybrid** | **75.2%** 🏆 |

---

## 🖥️ Frontend — Streamlit Dashboard

```bash
streamlit run app.py
```

A sci-fi glassmorphic dashboard that brings all six models together:

- **Paste any amino acid sequence** → full 3D structure + Q8 + confidence scores in ~4 seconds
- **Six models running in parallel:** Logistic · Feedforward · CNN · RNN · Hybrid · 3D-Generator
- **Ensemble voting** across 8 classes with interactive glassmorphic pie chart
- **Instant PDB download** + interactive Plotly 3D visualization + φ/ψ torsion plots
- Zero setup — just `streamlit run app.py`

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/S-Farhana/HelixNet.git
cd HelixNet

# Install dependencies
pip install torch numpy streamlit plotly

# Run preprocessing
python Preprocessing.py

# Train a model (e.g., CNN-RNN Hybrid)
python "Final working codes/cnn_rnn_hybrid.py"

# Launch the dashboard
streamlit run app.py
```

---

## 📁 Dataset

HelixNet uses the **CB513 benchmark dataset** for training and evaluation.

- `CB513_FASTA/` — raw amino acid sequences in FASTA format
- `CB513_DSSP/` — DSSP-annotated secondary structure labels
- `Parse_FASTA_DSSP_PDP.py` — parser to align sequences, DSSP labels, and PDB 3D coordinates

---

## 🛠️ Tech Stack

| Component | Tools |
|---|---|
| Core ML (baselines) | NumPy (from-scratch) |
| Deep Learning | PyTorch |
| Visualization | Plotly, Streamlit |
| Data | CB513, DSSP, PDB |
| Features | One-hot, Hydrophobicity, Sliding Windows, Skip-Gram |
