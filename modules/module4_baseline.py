"""
MODULE 4 — Baseline ML Models
- Logistic Regression, SVM, Random Forest
- Accuracy, Precision, Recall, F1, ROC-AUC
"""

import pandas as pd, numpy as np, json, pickle, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

FEAT_COLS = ["age","gender","admission_type","admission_location","discharge_location",
             "insurance","language","marital_status","ethnicity","diagnosis","length_of_stay"]

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else y_pred
    return {
        "model": name,
        "accuracy":  round(accuracy_score(y_test, y_pred)*100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0)*100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0)*100, 2),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0)*100, 2),
        "auc":       round(roc_auc_score(y_test, y_prob)*100, 2),
    }

def run():
    print("="*60)
    print("  MODULE 4 — BASELINE ML MODELS")
    print("="*60)

    train = pd.read_csv(os.path.join(BASE, "data", "train_smote.csv"))
    test  = pd.read_csv(os.path.join(BASE, "data", "test_clean.csv"))

    X_train, y_train = train[FEAT_COLS], train["readmission"]
    X_test,  y_test  = test[FEAT_COLS],  test["readmission"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42, kernel='rbf'),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    trained = {}
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        res = evaluate(model, X_test, y_test, name)
        results.append(res)
        trained[name] = model
        print(f"   Accuracy: {res['accuracy']}%  |  F1: {res['f1']}%  |  AUC: {res['auc']}%")

    # Save models
    for name, model in trained.items():
        fname = name.lower().replace(" ","_")
        with open(os.path.join(BASE, "models", f"{fname}.pkl"),"wb") as f:
            pickle.dump(model, f)

    # Save results
    with open(os.path.join(BASE, "data", "baseline_results.json"),"w") as f:
        json.dump(results, f, indent=2)

    # ── Comparison bar chart ───────────────────────────────────────────────────
    metrics = ["accuracy","f1","auc"]
    labels  = ["Accuracy %","F1-Score %","ROC-AUC %"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1E2761','#4A90D9','#28a745']

    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')

    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=res['model'], color=colors[i],
                      edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Baseline ML Models — Performance Comparison", fontsize=14,
                 fontweight='bold', color='#1E2761', pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "static", "images", "baseline_comparison.png"), dpi=120)
    plt.close()

    print("\n\n📊 RESULTS SUMMARY:")
    print(f"{'Model':<25} {'Acc%':>7} {'Pre%':>7} {'Rec%':>7} {'F1%':>7} {'AUC%':>7}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<25} {r['accuracy']:>7} {r['precision']:>7} {r['recall']:>7} {r['f1']:>7} {r['auc']:>7}")

    print("\n✅ Module 4 Complete!")

if __name__ == "__main__":
    run()
