"""
module1_data_collection.py
MODULE 1 — Data Collection & Setup
- Load ADMISSIONS_random.csv
- Explore shape, dtypes, null values
- Define target column
- Train/test split (80/20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "ADMISSIONS_random.csv"))

print("=" * 60)
print("  MODULE 1 — DATA COLLECTION & SETUP")
print("=" * 60)

# ── 2. Basic Exploration ───────────────────────────────────────────────────────
print(f"\n📦 Dataset Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📋 Columns           : {list(df.columns)}")

print("\n📊 Data Types:")
print(df.dtypes.to_string())

print("\n🔍 First 3 Rows:")
pd.set_option("display.max_colwidth", 60)
print(df.head(3).to_string())

# ── 3. Null Value Analysis ─────────────────────────────────────────────────────
print("\n⚠️  Missing Values:")
nulls = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_report = pd.DataFrame({"Missing Count": nulls, "Missing %": null_pct})
null_report = null_report[null_report["Missing Count"] > 0]
if null_report.empty:
    print("   No missing values found.")
else:
    print(null_report.to_string())

# ── 4. Target Column Analysis ──────────────────────────────────────────────────
print("\n🎯 Target Column: 'readmission'")
vc = df["readmission"].value_counts()
print(f"   Class 0 (Not Readmitted) : {vc[0]:,} ({vc[0]/len(df)*100:.1f}%)")
print(f"   Class 1 (Readmitted)     : {vc[1]:,} ({vc[1]/len(df)*100:.1f}%)")
print(f"   Imbalance Ratio          : {vc[0]/vc[1]:.1f}:1  ← will be handled in Module 3")

# ── 5. Basic Statistics ────────────────────────────────────────────────────────
print("\n📈 Numerical Feature Statistics:")
print(df[["age", "length_of_stay"]].describe().round(2).to_string())

print("\n📊 Categorical Feature Value Counts (top 3):")
for col in ["admission_type", "insurance", "ethnicity"]:
    print(f"\n   {col}:")
    print(df[col].value_counts().head(3).to_string())

# ── 6. Train / Test Split ──────────────────────────────────────────────────────
X = df.drop(columns=["readmission"])
y = df["readmission"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("  TRAIN / TEST SPLIT (80/20, stratified)")
print("=" * 60)
print(f"   Training set : {X_train.shape[0]:,} rows")
print(f"   Test set     : {X_test.shape[0]:,} rows")
print(f"\n   Train class balance:")
print(f"     Class 0 : {(y_train==0).sum():,} | Class 1 : {(y_train==1).sum():,}")
print(f"   Test class balance:")
print(f"     Class 0 : {(y_test==0).sum():,} | Class 1 : {(y_test==1).sum():,}")

# ── 7. Save split indices for downstream modules ───────────────────────────────
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

train_df = X_train.copy()
train_df["readmission"] = y_train
test_df = X_test.copy()
test_df["readmission"] = y_test

train_df.to_csv(os.path.join(BASE, "data", "train.csv"), index=False)
test_df.to_csv(os.path.join(BASE, "data", "test.csv"),  index=False)

# Save module summary
summary = {
    "total_rows": int(len(df)),
    "total_cols": int(len(df.columns)),
    "train_size": int(len(train_df)),
    "test_size":  int(len(test_df)),
    "class_0_count": int(vc[0]),
    "class_1_count": int(vc[1]),
    "readmission_rate_pct": round(vc[1]/len(df)*100, 1),
    "missing_cols": null_report.index.tolist() if not null_report.empty else [],
}
with open(os.path.join(BASE, "data", "module1_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Module 1 Complete!")
print("   Saved: data/train.csv, data/test.csv, data/module1_summary.json")
print("\n   Ready for → MODULE 2: Data Preprocessing")
