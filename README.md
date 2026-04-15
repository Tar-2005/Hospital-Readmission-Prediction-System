# 🏥 Hospital Readmission Prediction System
### Using ClinicalBERT + Transformer with Class Variable Imbalance Handling

---

## 📌 Project Overview

This system predicts the risk of **30-day hospital readmission** by analyzing unstructured clinical notes from Electronic Health Records (EHR) using:

- **ClinicalBERT** — pretrained on MIMIC clinical dataset for medical language understanding
- **Attention-based Transformer** — captures long-range contextual dependencies
- **SMOTE** — handles 4:1 class imbalance in readmission data
- **Flask** — full web application for clinical decision support

### 🎯 Key Results
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 70.25% | 67.49% | 72.31% |
| SVM | 71.50% | 69.52% | 74.18% |
| Random Forest | 72.75% | 71.23% | 76.05% |
| **ClinicalBERT+Transformer** ⭐ | **87.25%** | **87.65%** | **91.43%** |

**Improvement over best baseline: +14.5% accuracy**

---

## 📁 Project Structure

```
project/
├── app.py                        # Flask main application
├── ADMISSIONS_random.csv         # Dataset (MIMIC-IV style)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── login.db                      # SQLite auth database (auto-created)
│
├── modules/
│   ├── module2_preprocessing.py  # Data preprocessing pipeline
│   ├── module3_imbalance.py      # SMOTE + UnderSampler
│   ├── module4_baseline.py       # LR, SVM, Random Forest training
│   ├── module56_transformer.py   # ClinicalBERT + Transformer training
│   └── module7_evaluation.py     # Evaluation charts & report
│
├── models/
│   ├── best_model.pt             # Trained Transformer model weights
│   ├── tfidf.pkl                 # TF-IDF vectorizer
│   ├── svd.pkl                   # SVD dimensionality reducer
│   ├── encoders.pkl              # Label encoders for categorical features
│   └── scaler.pkl                # StandardScaler for numerical features
│
├── data/
│   ├── train.csv / test.csv      # Raw train/test splits
│   ├── train_clean.csv           # Preprocessed training data
│   ├── train_smote.csv           # SMOTE-balanced training data
│   ├── baseline_results.json     # Baseline model metrics
│   ├── transformer_result.json   # Transformer model metrics
│   ├── evaluation_report.json    # Full evaluation report
│   ├── confusion_matrix.json     # CM data for Flask
│   ├── imbalance_stats.json      # Before/after SMOTE stats
│   └── training_history.json     # Epoch-wise training log
│
├── templates/
│   ├── base.html                 # Shared layout (navbar + sidebar)
│   ├── login.html                # Login page
│   ├── dashboard.html            # Summary dashboard
│   ├── dataset.html              # Dataset overview
│   ├── preprocess.html           # Preprocessing steps
│   ├── imbalance.html            # Class imbalance visualization
│   ├── train.html                # Model training log & architecture
│   ├── predict.html              # Risk prediction form + test cases
│   ├── compare.html              # Model comparison charts
│   └── results.html              # Full evaluation report
│
└── static/
    └── images/
        ├── imbalance_chart.png
        ├── baseline_comparison.png
        ├── all_models_comparison.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── training_curve.png
```

---

## 🚀 Setup & Run Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run All Training Modules (in order)
```bash
python modules/module2_preprocessing.py
python modules/module3_imbalance.py
python modules/module4_baseline.py
python modules/module56_transformer.py
python modules/module7_evaluation.py
```

### Step 3: Launch Flask Application
```bash
python app.py
```

### Step 4: Open in Browser
```
http://localhost:5000
```

### Login Credentials
```
Login ID : admin@mimiciv.edu
Password : admin123
```

---

## 🌐 Application Routes

| Route | Description |
|-------|-------------|
| `/` | Login page |
| `/dashboard` | Project summary & stats cards |
| `/dataset` | Dataset overview, sample records, class distribution |
| `/preprocess` | Preprocessing pipeline steps & text cleaning demo |
| `/imbalance` | SMOTE vs UnderSampler visualization |
| `/train` | Transformer architecture + epoch training log |
| `/predict` | Clinical note input → readmission risk % |
| `/compare` | All models comparison charts |
| `/results` | Confusion matrix, ROC curve, full evaluation |
| `/logout` | End session |

---

## 🧠 Technical Stack

| Layer | Technology |
|-------|-----------|
| NLP Model | ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) |
| Deep Learning | PyTorch, Multi-head Self-Attention |
| Imbalance | imbalanced-learn (SMOTE, RandomUnderSampler) |
| Backend | Python 3.11, Flask 3.0 |
| ML Baselines | scikit-learn (LR, SVM, RF) |
| Database | SQLite |
| Visualization | Matplotlib, Chart.js |
| Frontend | Bootstrap 5, Jinja2 |

---

## 📊 SDG Alignment

- **SDG 3 — Good Health and Well-Being**: Early AI-driven identification of high-risk readmission patients
- **SDG 9 — Industry, Innovation and Infrastructure**: Transformer-based NLP innovation in healthcare

---

