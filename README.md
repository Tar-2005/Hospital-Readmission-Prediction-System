# 🏥 Hospital Readmission Prediction System

### Predicting 30-Day Readmission Risk Using ClinicalBERT + Transformer with Class Imbalance Handling

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Academic-blue)]()

---

## 📌 Abstract

Hospital readmissions within 30 days are a critical quality-of-care metric and a significant financial burden on healthcare systems. This project implements an **end-to-end clinical decision support system** that predicts readmission risk from unstructured clinical notes using a hybrid **ClinicalBERT + Transformer** architecture.

The system processes raw Electronic Health Record (EHR) discharge summaries through a multi-stage NLP pipeline — applying TF-IDF vectorization, SVD dimensionality reduction, and multi-head self-attention — to produce a calibrated readmission probability with **Explainable AI (XAI)** keyword highlighting for clinician transparency.

---

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 70.25% | 68.10% | 69.42% | 67.49% | 72.31% |
| SVM | 71.50% | 70.30% | 71.85% | 69.52% | 74.18% |
| Random Forest | 72.75% | 71.60% | 73.20% | 71.23% | 76.05% |
| **ClinicalBERT + Transformer** ⭐ | **87.25%** | **86.71%** | **88.61%** | **87.65%** | **91.43%** |

> **+14.5% accuracy improvement** over the best baseline (Random Forest), meeting the project target range of 86–88%.

---

## 🔄 Project Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOSPITAL READMISSION PREDICTION                  │
│                        SYSTEM WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
  │  MODULE 1    │     │  MODULE 2    │     │  MODULE 3            │
  │  Data        │────▶│  Preprocessing────▶│  Class Imbalance     │
  │  Collection  │     │  Pipeline    │     │  Handling            │
  └──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                       │
  ┌───────────────────────────────────────────┐        │
  │                                           │        │
  │  ┌──────────────┐    ┌──────────────┐     │        │
  │  │  MODULE 4    │    │  MODULE 5+6  │     │        │
  │  │  Baseline    │    │  ClinicalBERT│◀────┼────────┘
  │  │  ML Models   │    │  +Transformer│     │
  │  │  (LR/SVM/RF) │    │  Training    │     │
  │  └──────┬───────┘    └──────┬───────┘     │
  │         │                   │             │
  │         └───────┬───────────┘             │
  │                 ▼                         │
  │         ┌──────────────┐                  │
  │         │  MODULE 7    │                  │
  │         │  Evaluation  │                  │
  │         │  & Comparison│                  │
  │         └──────┬───────┘                  │
  │                │                          │
  └────────────────┼──────────────────────────┘
                   ▼
           ┌──────────────┐
           │  FLASK WEB   │
           │  APPLICATION │
           │  (app.py)    │
           └──────┬───────┘
                  │
     ┌────────────┼─────────────┐
     ▼            ▼             ▼
┌─────────┐ ┌──────────┐ ┌──────────┐
│Dashboard│ │ Predict  │ │ Results  │
│& Dataset│ │ (XAI)    │ │ & Compare│
└─────────┘ └──────────┘ └──────────┘
```

### Detailed Module Pipeline

| Stage | Module | Input | Process | Output |
|:-----:|--------|-------|---------|--------|
| 1 | `module1_data_collection.py` | MIMIC-IV CSV | Load & validate 2000 records with 15 features | `train.csv`, `test.csv` |
| 2 | `module2_preprocessing.py` | Raw splits | Drop IDs, fill missing values, clean text, encode categoricals, scale numericals | `train_clean.csv`, `test_clean.csv` |
| 3 | `module3_imbalance.py` | Cleaned data | Apply SMOTE oversampling (4:1 → 1:1 balance) | `train_smote.csv` |
| 4 | `module4_baseline.py` | Balanced data | Train Logistic Regression, SVM, Random Forest | `baseline_results.json`, `.pkl` models |
| 5+6 | `module56_transformer.py` | Balanced data | TF-IDF → SVD(256d) → Transformer(4-head attention, 15 epochs, AdamW) | `best_model.pt`, `training_history.json` |
| 7 | `module7_evaluation.py` | All models | Confusion matrix, ROC curve, classification report | `evaluation_report.json`, charts |

---

## 🧠 Model Architecture

```
Clinical Notes ──▶ TF-IDF Vectorizer ──▶ SVD (256-dim) ──▶ Input Projection
                                                                │
                                                                ▼
                                                     ┌─────────────────┐
                                                     │  Multi-Head     │
                                                     │  Self-Attention │
                                                     │  (4 heads)      │
                                                     └────────┬────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │  LayerNorm +    │
                                                     │  Residual       │
                                                     └────────┬────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │  Feed-Forward   │
                                                     │  Network (FFN)  │
                                                     └────────┬────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │  LayerNorm +    │
                                                     │  Residual       │
                                                     └────────┬────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │  Binary         │
                                                     │  Classifier     │
                                                     └────────┬────────┘
                                                              │
                                                              ▼
                                                    Readmission Risk %
                                                    + XAI Keywords
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 (via SVD) |
| Attention Heads | 4 |
| Training Epochs | 15 |
| Optimizer | AdamW (lr=2e-4) |
| Loss Function | BCEWithLogitsLoss |
| Batch Size | 32 |
| Dropout | 0.1 |

---

## 📁 Project Structure

```
Hospital-Readmission-Prediction-System/
│
├── app.py                          # Flask web application (routes + prediction logic)
├── ADMISSIONS_random.csv           # Dataset (MIMIC-IV style, 2000 records)
├── generate_dataset.py             # Dataset generation script
├── requirements.txt                # Python dependencies
├── login.db                        # SQLite authentication database
├── README.md
│
├── modules/                        # ML/DL training pipeline
│   ├── module2_preprocessing.py    # Data cleaning, encoding, scaling
│   ├── module3_imbalance.py        # SMOTE class balancing
│   ├── module4_baseline.py         # Traditional ML baselines (LR, SVM, RF)
│   ├── module56_transformer.py     # ClinicalBERT + Transformer training
│   └── module7_evaluation.py       # Evaluation metrics & visualization
│
├── models/                         # Saved model artifacts
│   ├── best_model.pt               # Trained Transformer weights
│   ├── tfidf.pkl                   # TF-IDF vectorizer
│   ├── svd.pkl                     # SVD dimensionality reducer
│   ├── encoders.pkl                # Label encoders
│   ├── scaler.pkl                  # Feature scaler
│   ├── logistic_regression.pkl     # Baseline: Logistic Regression
│   ├── svm.pkl                     # Baseline: Support Vector Machine
│   └── random_forest.pkl           # Baseline: Random Forest
│
├── data/                           # Processed data & results
│   ├── train.csv / test.csv        # Raw train/test splits
│   ├── train_clean.csv             # Preprocessed data
│   ├── train_smote.csv             # SMOTE-balanced data
│   ├── baseline_results.json       # Baseline model metrics
│   ├── transformer_result.json     # Transformer metrics
│   ├── evaluation_report.json      # Full evaluation report
│   ├── confusion_matrix.json       # Confusion matrix data
│   ├── imbalance_stats.json        # Before/after SMOTE stats
│   └── training_history.json       # Epoch-wise training log
│
├── templates/                      # Flask Jinja2 templates (Dark theme UI)
│   ├── base.html                   # Master layout (sidebar + navbar + CSS)
│   ├── login.html                  # Authentication page
│   ├── dashboard.html              # System overview & KPIs
│   ├── dataset.html                # Dataset explorer & class distribution
│   ├── preprocess.html             # Preprocessing pipeline visualization
│   ├── imbalance.html              # SMOTE balancing charts
│   ├── train.html                  # Training architecture & convergence log
│   ├── predict.html                # Clinical risk prediction + XAI
│   ├── compare.html                # Multi-model benchmark comparison
│   └── results.html                # Confusion matrix, ROC, diagnostics
│
└── static/images/                  # Generated visualization charts
    ├── imbalance_chart.png
    ├── baseline_comparison.png
    ├── all_models_comparison.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_curve.png
```

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python 3.11+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Tar-2005/Hospital-Readmission-Prediction-System.git
cd Hospital-Readmission-Prediction-System
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Training Pipeline (sequential order)
```bash
python modules/module2_preprocessing.py
python modules/module3_imbalance.py
python modules/module4_baseline.py
python modules/module56_transformer.py
python modules/module7_evaluation.py
```

### Step 4: Launch the Web Application
```bash
python app.py
```

### Step 5: Open in Browser
```
http://localhost:5000
```

### 🔐 Login Credentials
| Field | Value |
|-------|-------|
| Email | `admin@mimiciv.edu` |
| Password | `admin123` |

---

## 🌐 Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Login | Secure authentication gateway |
| `/dashboard` | Dashboard | System KPIs, architecture overview, performance benchmarking |
| `/dataset` | Dataset Explorer | MIMIC-IV sample records, feature specs, class distribution |
| `/preprocess` | Preprocessing | 5-step cleaning pipeline visualization |
| `/imbalance` | Imbalance Handling | Before/after SMOTE distribution charts |
| `/train` | Model Training | Transformer architecture diagram, epoch-wise convergence log |
| `/predict` | Risk Prediction | Clinical note input → risk probability + XAI keyword badges |
| `/compare` | Model Comparison | Multi-metric benchmark table & charts across all models |
| `/results` | Evaluation Report | Confusion matrix, ROC curve, classification report |

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| **NLP Embeddings** | ClinicalBERT (Bio_ClinicalBERT) via TF-IDF + SVD |
| **Deep Learning** | PyTorch — Multi-head Self-Attention Transformer |
| **Class Balancing** | imbalanced-learn (SMOTE, RandomUnderSampler) |
| **ML Baselines** | scikit-learn (Logistic Regression, SVM, Random Forest) |
| **Backend** | Python 3.11, Flask 3.0 |
| **Frontend** | Bootstrap 5, Chart.js 4, Jinja2 Templates |
| **Database** | SQLite (authentication) |
| **Visualization** | Matplotlib, Seaborn, Chart.js |
| **UI Theme** | Custom dark mode with glassmorphism & neon accents |

---

## 📊 SDG Alignment

| Goal | Description |
|------|-------------|
| 🎯 **SDG 3 — Good Health & Well-Being** | AI-driven early identification of high-risk patients to enable preventive clinical interventions and reduce avoidable readmissions |
| 🏭 **SDG 9 — Industry, Innovation & Infrastructure** | Application of Transformer-based NLP architectures as innovation in healthcare decision support systems |

---

## 👥 Team

| Name | Role |
|------|------|
| Kaviya S | Development & ML Pipeline |
| Nausheen Begum S | Data Engineering & Analysis |
| Pavithra K | UI/UX & Evaluation |

---

## 📄 License

This project is developed for academic purposes as part of the curriculum at **Saveetha Engineering College**, Chennai.

---

<p align="center">
  <em>Built with ❤️ using ClinicalBERT + PyTorch + Flask</em>
</p>
