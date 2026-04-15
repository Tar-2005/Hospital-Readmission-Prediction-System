"""
MODULE 2 — Data Preprocessing
- Drop ID/irrelevant columns
- Handle missing values
- Clean clinical text
- Label encode categoricals
- Scale numericals
"""

import pandas as pd
import numpy as np
import re, json, pickle, os
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\.\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.copy()

    # 1. Drop ID columns
    df.drop(columns=["subject_id", "hadm_id"], inplace=True, errors='ignore')

    # 2. Fill missing values
    for col in ["language", "marital_status"]:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col].fillna(mode_val, inplace=True)

    # 3. Clean clinical notes
    df["clinical_notes_clean"] = df["clinical_notes"].apply(clean_text)

    # 4. Encode categoricals
    cat_cols = ["gender","admission_type","admission_location","discharge_location",
                "insurance","language","marital_status","ethnicity","diagnosis"]
    if fit:
        encoders = {}
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in cat_cols:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    # 5. Scale numericals
    num_cols = ["age", "length_of_stay"]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    return df, encoders, scaler

if __name__ == "__main__":
    print("="*60)
    print("  MODULE 2 — DATA PREPROCESSING")
    print("="*60)

    train = pd.read_csv(os.path.join(BASE, "data", "train.csv"))
    test  = pd.read_csv(os.path.join(BASE, "data", "test.csv"))

    train_clean, encoders, scaler = preprocess(train, fit=True)
    test_clean,  _,        _      = preprocess(test,  encoders=encoders, scaler=scaler, fit=False)

    print(f"\n✅ Train shape after preprocessing : {train_clean.shape}")
    print(f"✅ Test  shape after preprocessing : {test_clean.shape}")
    print(f"\n📋 Remaining columns:\n   {list(train_clean.columns)}")
    print(f"\n⚠️  Null check after preprocessing:")
    print(f"   Train nulls: {train_clean.isnull().sum().sum()}")
    print(f"   Test  nulls: {test_clean.isnull().sum().sum()}")

    print("\n📊 Sample cleaned clinical note:")
    print("   BEFORE:", train["clinical_notes"].iloc[0][:100])
    print("   AFTER :", train_clean["clinical_notes_clean"].iloc[0][:100])

    train_clean.to_csv(os.path.join(BASE, "data", "train_clean.csv"), index=False)
    test_clean.to_csv(os.path.join(BASE, "data", "test_clean.csv"),   index=False)

    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    with open(os.path.join(BASE, "models", "encoders.pkl"), "wb") as f: pickle.dump(encoders, f)
    with open(os.path.join(BASE, "models", "scaler.pkl"),   "wb") as f: pickle.dump(scaler,   f)

    print("\n✅ Module 2 Complete! Saved train_clean.csv, test_clean.csv, encoders.pkl, scaler.pkl")
