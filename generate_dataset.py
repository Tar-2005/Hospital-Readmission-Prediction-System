"""
generate_dataset.py
Generates a realistic synthetic MIMIC-IV style ADMISSIONS dataset.
Mirrors real column names, distributions, and class imbalance (~20% readmission).
"""

import numpy as np
import pandas as pd
import random
import string
import os

np.random.seed(42)
random.seed(42)

N = 2000  # total patients

# --- Helpers ---
ADMISSION_TYPES  = ["EMERGENCY", "ELECTIVE", "URGENT", "OBSERVATION ADMIT"]
ADMISSION_LOCS   = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL",
                     "WALK-IN/SELF REFERRAL", "CLINIC REFERRAL"]
DISCHARGE_LOCS   = ["HOME", "HOME HEALTH CARE", "SNF", "REHAB", "LONG TERM CARE HOSPITAL",
                     "DIED", "LEFT AGAINST MEDICAL ADVICE"]
INSURANCES       = ["Medicare", "Medicaid", "Private", "Self Pay", "Other"]
LANGUAGES        = ["ENGLISH", "SPANISH", "CHINESE", "OTHER"]
MARITAL_STATUSES = ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "UNKNOWN"]
ETHNICITIES      = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO",
                     "ASIAN", "UNKNOWN", "OTHER"]
DIAGNOSES = [
    "Congestive heart failure", "Pneumonia", "Sepsis", "COPD exacerbation",
    "Acute kidney injury", "Myocardial infarction", "Diabetes mellitus",
    "Stroke", "Hip fracture", "Urinary tract infection",
    "Gastrointestinal bleeding", "Cellulitis", "Atrial fibrillation"
]

CLINICAL_NOTE_TEMPLATES = [
    # HIGH RISK templates
    "Patient is a {age}-year-old {gender} with a history of {dx1} and {dx2}. "
    "Admitted via emergency department with worsening {symptom}. "
    "Multiple comorbidities noted. Creatinine elevated at {cr:.1f} mg/dL. "
    "Poor social support. Discharged to {dc_loc}. "
    "Follow-up in 7 days recommended due to high readmission risk.",

    "Elderly {gender} patient, age {age}, presenting with acute {dx1}. "
    "History of {dx2} and chronic kidney disease. "
    "Hospital course complicated by {complication}. "
    "Requires close outpatient monitoring. Medications adjusted at discharge.",

    # LOW RISK templates
    "Patient is a {age}-year-old {gender} admitted electively for {dx1}. "
    "No significant comorbidities. Vitals stable throughout admission. "
    "Creatinine {cr:.1f} mg/dL, within normal limits. "
    "Discharged home in good condition. Follow-up scheduled in 4 weeks.",

    "{age}-year-old {gender} with well-controlled {dx1}. "
    "Admitted for routine procedure. Recovered uneventfully. "
    "Labs within normal range. Discharged home with standard instructions.",
]

SYMPTOMS     = ["shortness of breath", "chest pain", "altered mental status",
                 "acute hypoxia", "severe edema", "hemodynamic instability"]
COMPLICATIONS = ["acute respiratory failure", "electrolyte imbalance",
                  "hospital-acquired pneumonia", "delirium", "septic shock"]

def generate_clinical_note(age, gender, dx1, dx2, dc_loc, readmission):
    symptom      = random.choice(SYMPTOMS)
    complication = random.choice(COMPLICATIONS)
    cr = np.random.uniform(1.8, 4.2) if readmission else np.random.uniform(0.6, 1.3)
    # high-risk templates for readmission cases
    if readmission:
        tmpl = random.choice(CLINICAL_NOTE_TEMPLATES[:2])
    else:
        tmpl = random.choice(CLINICAL_NOTE_TEMPLATES[2:])
    return tmpl.format(age=age, gender=gender.lower(), dx1=dx1, dx2=dx2,
                        dc_loc=dc_loc, symptom=symptom, complication=complication, cr=cr)

# --- Generate rows ---
rows = []
subject_id_start = 10000000

for i in range(N):
    subject_id   = subject_id_start + i
    hadm_id      = 20000000 + i
    age          = int(np.clip(np.random.normal(65, 15), 18, 95))
    gender       = random.choice(["M", "F"])
    dx1          = random.choice(DIAGNOSES)
    dx2          = random.choice([d for d in DIAGNOSES if d != dx1])
    admission_type = random.choice(ADMISSION_TYPES)
    admission_loc  = random.choice(ADMISSION_LOCS)
    discharge_loc  = random.choice(DISCHARGE_LOCS)
    insurance      = random.choice(INSURANCES)
    language       = random.choice(LANGUAGES)
    marital_status = random.choice(MARITAL_STATUSES)
    ethnicity      = random.choice(ETHNICITIES)
    los            = int(np.clip(np.random.exponential(5), 1, 30))  # length of stay

    # Realistic class imbalance: ~20% readmission
    readmission = int(np.random.random() < 0.20)

    # Introduce some missing values naturally
    if random.random() < 0.05:
        marital_status = np.nan
    if random.random() < 0.03:
        language = np.nan

    clinical_note = generate_clinical_note(age, gender, dx1, dx2, discharge_loc, readmission)

    rows.append({
        "subject_id":     subject_id,
        "hadm_id":        hadm_id,
        "age":            age,
        "gender":         gender,
        "admission_type": admission_type,
        "admission_location": admission_loc,
        "discharge_location": discharge_loc,
        "insurance":      insurance,
        "language":       language,
        "marital_status": marital_status,
        "ethnicity":      ethnicity,
        "diagnosis":      dx1,
        "length_of_stay": los,
        "clinical_notes": clinical_note,
        "readmission":    readmission,
    })

df = pd.DataFrame(rows)
BASE = os.path.dirname(os.path.abspath(__file__))
df.to_csv(os.path.join(BASE, "ADMISSIONS_random.csv"), index=False)
print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.dtypes)
print("\nClass distribution:\n", df["readmission"].value_counts())
print(f"\nReadmission rate: {df['readmission'].mean()*100:.1f}%")
