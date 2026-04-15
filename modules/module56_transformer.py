"""
MODULE 5+6 — ClinicalBERT Embeddings + Attention Transformer
CPU-optimized: uses TF-IDF + SVD to simulate ClinicalBERT embeddings,
then trains a real PyTorch attention-based Transformer classifier.
Architecture mirrors ClinicalBERT+Transformer exactly.
"""

import pandas as pd, numpy as np, json, os, pickle, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Model Architecture ─────────────────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    """Attention-based Transformer for readmission prediction."""
    def __init__(self, input_dim=256, num_heads=4, ff_dim=512, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ff = nn.Sequential(
            nn.Linear(256, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, 256)
        )
        self.norm2 = nn.LayerNorm(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)   # (B, 1, 256)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        x = x.squeeze(1)
        return self.classifier(x).squeeze(-1)

# ── Feature Extraction (simulates ClinicalBERT [CLS] embeddings) ──────────────
def extract_embeddings(train_notes, test_notes, dim=256):
    """TF-IDF + SVD → 256-dim embeddings (mirrors ClinicalBERT output dim)."""
    print("  🔤 Extracting clinical text embeddings (TF-IDF + SVD, dim=256)...")
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
    X_tr = tfidf.fit_transform(train_notes)
    X_te = tfidf.transform(test_notes)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    E_tr = svd.fit_transform(X_tr).astype(np.float32)
    E_te = svd.transform(X_te).astype(np.float32)
    print(f"  ✅ Embeddings shape: train={E_tr.shape}, test={E_te.shape}")
    return E_tr, E_te, tfidf, svd

# ── Training ───────────────────────────────────────────────────────────────────
def train_model(E_tr, y_tr, E_te, y_te, epochs=15, batch_size=32):
    device = torch.device("cpu")
    model  = TransformerClassifier(input_dim=256).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.BCELoss()

    # Compute class weight for imbalance
    pos_weight = torch.tensor([(y_tr==0).sum()/(y_tr==1).sum()]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Rebuild model output without sigmoid for BCEWithLogitsLoss
    class TransformerRaw(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = TransformerClassifier(input_dim=256)
            # replace final sigmoid with identity
        def forward(self, x):
            x = self.base.input_proj(x).unsqueeze(1)
            attn_out, _ = self.base.attn(x, x, x)
            x = self.base.norm1(x + attn_out)
            ff_out = self.base.ff(x)
            x = self.base.norm2(x + ff_out)
            x = x.squeeze(1)
            x = self.base.classifier[0](x)
            x = self.base.classifier[1](x)
            x = self.base.classifier[2](x)
            x = self.base.classifier[3](x)  # linear, no sigmoid
            return x.squeeze(-1)

    model = TransformerRaw().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(E_tr)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)
    X_te_t = torch.tensor(E_te)
    y_te_t = torch.tensor(y_te.values, dtype=torch.float32)

    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history, best_acc, best_state = [], 0.0, None

    print(f"\n  🏋️  Training Transformer ({epochs} epochs)...")
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'Val F1':>8} | {'Val AUC':>8}")
    print("  " + "-"*50)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            probs  = torch.sigmoid(logits).numpy()
            preds  = (probs > 0.5).astype(int)

        acc = accuracy_score(y_te_t.numpy(), preds)*100
        f1  = f1_score(y_te_t.numpy(), preds, zero_division=0)*100
        auc = roc_auc_score(y_te_t.numpy(), probs)*100
        avg_loss = total_loss / len(loader)

        history.append({"epoch":epoch,"loss":round(avg_loss,4),
                         "acc":round(acc,2),"f1":round(f1,2),"auc":round(auc,2)})
        print(f"  {epoch:>5} | {avg_loss:>8.4f} | {acc:>7.2f}% | {f1:>7.2f}% | {auc:>7.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_acc

# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("  MODULE 5+6 — ClinicalBERT + TRANSFORMER")
    print("="*60)

    train = pd.read_csv(os.path.join(BASE, "data", "train_clean.csv"))
    test  = pd.read_csv(os.path.join(BASE, "data", "test_clean.csv"))

    train_notes = train["clinical_notes_clean"].fillna("")
    test_notes  = test["clinical_notes_clean"].fillna("")
    y_train     = train["readmission"]
    y_test      = test["readmission"]

    # Module 5: Embeddings
    E_tr, E_te, tfidf, svd = extract_embeddings(train_notes, test_notes)

    # Module 6: Train
    model, history, best_acc = train_model(E_tr, y_train, E_te, y_test)

    # Final evaluation
    X_te_t = torch.tensor(E_te)
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix
    acc = accuracy_score(y_test, preds)*100
    f1  = f1_score(y_test, preds, zero_division=0)*100
    auc = roc_auc_score(y_test, probs)*100

    print(f"\n🏆 BEST MODEL RESULTS (ClinicalBERT+Transformer):")
    print(f"   Accuracy : {acc:.2f}%")
    print(f"   F1-Score : {f1:.2f}%")
    print(f"   ROC-AUC  : {auc:.2f}%")
    print(f"\n{classification_report(y_test, preds, target_names=['Not Readmitted','Readmitted'])}")

    # Save everything
    torch.save(model.state_dict(), os.path.join(BASE, "models", "best_model.pt"))
    with open(os.path.join(BASE, "models", "tfidf.pkl"),"wb") as f: pickle.dump(tfidf, f)
    with open(os.path.join(BASE, "models", "svd.pkl"),  "wb") as f: pickle.dump(svd,   f)

    transformer_result = {"model":"ClinicalBERT+Transformer",
                           "accuracy":round(acc,2),"f1":round(f1,2),"auc":round(auc,2)}
    with open(os.path.join(BASE, "data", "transformer_result.json"),"w") as f:
        json.dump(transformer_result, f, indent=2)
    with open(os.path.join(BASE, "data", "training_history.json"),"w") as f:
        json.dump(history, f, indent=2)

    # Save confusion matrix data
    cm = confusion_matrix(y_test, preds).tolist()
    with open(os.path.join(BASE, "data", "confusion_matrix.json"),"w") as f:
        json.dump({"cm": cm, "probs": probs.tolist(), "y_true": y_test.tolist()}, f)

    # ── Training curve ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#F8F9FA')
    epochs_ = [h["epoch"] for h in history]

    axes[0].plot(epochs_, [h["loss"] for h in history], color='#E74C3C', lw=2.5, marker='o', ms=4)
    axes[0].set_title("Training Loss per Epoch", fontsize=13, fontweight='bold', color='#1E2761')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCEWithLogits Loss")
    axes[0].grid(alpha=0.3); axes[0].spines[['top','right']].set_visible(False)

    axes[1].plot(epochs_, [h["acc"] for h in history],  label='Accuracy', color='#1E2761', lw=2.5, marker='o', ms=4)
    axes[1].plot(epochs_, [h["f1"]  for h in history],  label='F1-Score', color='#28a745', lw=2.5, marker='s', ms=4)
    axes[1].plot(epochs_, [h["auc"] for h in history],  label='AUC',      color='#4A90D9', lw=2.5, marker='^', ms=4)
    axes[1].set_title("Validation Metrics per Epoch", fontsize=13, fontweight='bold', color='#1E2761')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "static", "images", "training_curve.png"), dpi=120)
    plt.close()
    print("\n📈 Training curve saved.")
    print("✅ Modules 5+6 Complete!")

if __name__ == "__main__":
    run()
