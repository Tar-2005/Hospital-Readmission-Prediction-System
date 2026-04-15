"""
MODULE 3 — Class Imbalance Handling
- SMOTE oversampling
- RandomUnderSampler
- Save balanced datasets + charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os, pickle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEAT_COLS = ["age","gender","admission_type","admission_location","discharge_location",
             "insurance","language","marital_status","ethnicity","diagnosis","length_of_stay"]

def run():
    print("="*60)
    print("  MODULE 3 — CLASS IMBALANCE HANDLING")
    print("="*60)

    train = pd.read_csv(os.path.join(BASE, "data", "train_clean.csv"))
    X = train[FEAT_COLS]
    y = train["readmission"]

    print(f"\n📊 Before balancing:")
    print(f"   Class 0 : {(y==0).sum():,}  |  Class 1 : {(y==1).sum():,}")
    print(f"   Ratio   : {(y==0).sum()/(y==1).sum():.1f}:1")

    # SMOTE
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    print(f"\n✅ After SMOTE:")
    print(f"   Class 0 : {(y_sm==0).sum():,}  |  Class 1 : {(y_sm==1).sum():,}")

    # RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    print(f"\n✅ After RandomUnderSampler:")
    print(f"   Class 0 : {(y_rus==0).sum():,}  |  Class 1 : {(y_rus==1).sum():,}")

    # Save SMOTE balanced data (used for training)
    df_sm = pd.DataFrame(X_sm, columns=FEAT_COLS)
    df_sm["readmission"] = y_sm
    df_sm.to_csv(os.path.join(BASE, "data", "train_smote.csv"), index=False)

    # Save imbalance stats for Flask
    stats = {
        "before": {"class_0": int((y==0).sum()), "class_1": int((y==1).sum())},
        "smote":  {"class_0": int((y_sm==0).sum()), "class_1": int((y_sm==1).sum())},
        "rus":    {"class_0": int((y_rus==0).sum()), "class_1": int((y_rus==1).sum())},
    }
    import json
    with open(os.path.join(BASE, "data", "imbalance_stats.json"),"w") as f:
        json.dump(stats, f, indent=2)

    # ── Chart ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#F8F9FA')
    colors = ['#1E2761','#4A90D9']
    titles = ['Before Balancing','After SMOTE','After UnderSampler']
    data_sets = [
        [(y==0).sum(), (y==1).sum()],
        [(y_sm==0).sum(), (y_sm==1).sum()],
        [(y_rus==0).sum(), (y_rus==1).sum()],
    ]
    for ax, title, vals in zip(axes, titles, data_sets):
        wedges, texts, autotexts = ax.pie(
            vals, labels=['Not Readmitted','Readmitted'],
            autopct='%1.1f%%', colors=colors,
            textprops={'fontsize':10}, startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2}
        )
        for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12, color='#1E2761')

    plt.suptitle('Class Imbalance — Before vs After Balancing', fontsize=15,
                 fontweight='bold', color='#1E2761', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.join(BASE, "static", "images"), exist_ok=True)
    plt.savefig(os.path.join(BASE, "static", "images", "imbalance_chart.png"),
                bbox_inches='tight', dpi=120)
    plt.close()
    print("\n📈 Chart saved: static/images/imbalance_chart.png")
    print("✅ Module 3 Complete!")

if __name__ == "__main__":
    run()
