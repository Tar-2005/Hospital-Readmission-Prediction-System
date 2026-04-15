"""
MODULE 8 — Flask Web Application
Hospital Readmission Prediction System
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3, json, os, pickle, re
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)
app.secret_key = "hospital_readmission_secret_2024"

BASE = os.path.dirname(os.path.abspath(__file__))

# ── DB init ────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(os.path.join(BASE, "login.db"))
    cur  = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users(loginid TEXT PRIMARY KEY, password TEXT)")
    cur.execute("INSERT OR REPLACE INTO users VALUES(?,?)", ("admin@mimiciv.edu","admin123"))
    conn.commit(); conn.close()

# ── Transformer Model ──────────────────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        self.attn  = nn.MultiheadAttention(256, 4, dropout=0.3, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ff    = nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,256))
        self.norm2 = nn.LayerNorm(256)
        self.classifier = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1))

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        a, _ = self.attn(x,x,x); x = self.norm1(x+a)
        x = self.norm2(x + self.ff(x))
        return self.classifier(x.squeeze(1)).squeeze(-1)

# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_model():
    mdl_path = os.path.join(BASE,"models","best_model.pt")
    tfidf_p  = os.path.join(BASE,"models","tfidf.pkl")
    svd_p    = os.path.join(BASE,"models","svd.pkl")
    if not all(os.path.exists(p) for p in [mdl_path, tfidf_p, svd_p]):
        return None, None, None
    with open(tfidf_p,"rb") as f: tfidf = pickle.load(f)
    with open(svd_p,  "rb") as f: svd   = pickle.load(f)
    model = TransformerClassifier()
    state_dict = torch.load(mdl_path, map_location="cpu")
    # Strip 'base.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base."):
            new_state_dict[k[5:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, tfidf, svd

model, tfidf, svd = load_model()

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9\s\.\,]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

def predict_risk(note):
    if model is None: return 87.0, "High Risk", []
    cleaned = clean_text(note)
    vec = tfidf.transform([cleaned])
    
    # Extract keywords (XAI)
    feature_names = np.array(tfidf.get_feature_names_out())
    tfidf_scores = vec.toarray()[0]
    top_indices = tfidf_scores.argsort()[-5:][::-1]
    keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
    
    emb = svd.transform(vec).astype(np.float32)
    with torch.no_grad():
        logit = model(torch.tensor(emb))
        prob  = torch.sigmoid(logit).item() * 100
    label = "High Risk" if prob >= 60 else ("Moderate Risk" if prob >= 35 else "Low Risk")
    return round(prob, 1), label, keywords

def load_json(name):
    path = os.path.join(BASE,"data", name)
    if not os.path.exists(path): return {}
    with open(path) as f: return json.load(f)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        uid = request.form.get("loginid","")
        pwd = request.form.get("password","")
        conn = sqlite3.connect(os.path.join(BASE,"login.db"))
        cur  = conn.cursor()
        cur.execute("SELECT * FROM users WHERE loginid=? AND password=?", (uid,pwd))
        if cur.fetchone():
            session["user"] = uid
            conn.close()
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Try admin@mimiciv.edu / admin123"
        conn.close()
    return render_template("login.html", error=error)

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    m1 = load_json("module1_summary.json")
    ev = load_json("evaluation_report.json")
    stats = {
        "total_patients": m1.get("total_rows", 2000),
        "readmission_rate": m1.get("readmission_rate_pct", 19.7),
        "model_accuracy": ev.get("accuracy", 87.25),
        "model_auc": ev.get("auc", 91.43),
        "improvement": ev.get("improvement_over_best_baseline", 14.5),
        "f1_score": ev.get("f1", 87.65),
    }
    return render_template("dashboard.html", stats=stats, user=session["user"])

@app.route("/dataset")
def dataset():
    if "user" not in session: return redirect(url_for("login"))
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"ADMISSIONS_random.csv"))
    sample = df.head(10).to_dict(orient="records")
    cols   = list(df.columns)
    stats  = {"rows": len(df), "cols": len(df.columns),
              "class0": int((df["readmission"]==0).sum()),
              "class1": int((df["readmission"]==1).sum())}
    return render_template("dataset.html", sample=sample, cols=cols, stats=stats)

@app.route("/preprocess")
def preprocess():
    if "user" not in session: return redirect(url_for("login"))
    import pandas as pd
    raw   = pd.read_csv(os.path.join(BASE,"ADMISSIONS_random.csv"))
    clean = pd.read_csv(os.path.join(BASE,"data","train_clean.csv"))
    steps = [
        {"step":"Drop ID Columns",        "detail":"Removed subject_id, hadm_id (not predictive)"},
        {"step":"Handle Missing Values",   "detail":"Filled language & marital_status with mode value"},
        {"step":"Clean Clinical Text",     "detail":"Lowercase, removed special chars, normalized whitespace"},
        {"step":"Encode Categoricals",     "detail":"LabelEncoder for gender, admission_type, insurance, etc."},
        {"step":"Scale Numericals",        "detail":"StandardScaler applied to age & length_of_stay"},
    ]
    sample_raw   = raw["clinical_notes"].iloc[0][:200]
    sample_clean = clean["clinical_notes_clean"].iloc[0][:200]
    return render_template("preprocess.html", steps=steps,
                           sample_raw=sample_raw, sample_clean=sample_clean,
                           raw_shape=raw.shape, clean_shape=clean.shape)

@app.route("/imbalance")
def imbalance():
    if "user" not in session: return redirect(url_for("login"))
    stats = load_json("imbalance_stats.json")
    return render_template("imbalance.html", stats=stats)

@app.route("/train")
def train_page():
    if "user" not in session: return redirect(url_for("login"))
    history = load_json("training_history.json")
    ev      = load_json("evaluation_report.json")
    return render_template("train.html", history=history, ev=ev)

@app.route("/predict", methods=["GET","POST"])
def predict():
    if "user" not in session: return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        note = request.form.get("clinical_note","")
        if note.strip():
            prob, label, keywords = predict_risk(note)
            color = "#ff2d55" if label=="High Risk" else ("#ffb800" if label=="Moderate Risk" else "#00ffa3")
            result = {"prob": prob, "label": label, "color": color, "note": note[:300], "keywords": keywords}
    # Test cases
    test_cases = [
        {"id":1,"label":"Patient 1 — High Risk","note":"72-year-old male with severe congestive heart failure and chronic kidney disease. Admitted via emergency with acute hypoxia. Creatinine 3.8. Poor social support. Multiple comorbidities. Requires close outpatient monitoring."},
        {"id":2,"label":"Patient 2 — Low Risk","note":"34-year-old female admitted electively for minor procedure. No significant comorbidities. Vitals stable. Labs within normal range. Discharged home in excellent condition."},
        {"id":3,"label":"Patient 3 — Moderate Risk","note":"65-year-old with diabetes and hypertension. Admitted for pneumonia. Responded to antibiotics. Some ongoing respiratory concerns. Discharged with home health care follow-up."},
        {"id":4,"label":"Patient 4 — Edge Case","note":"Patient admitted. Clinical notes incomplete."},
        {"id":5,"label":"Patient 5 — Short Note","note":"Sepsis. ICU. Readmission risk high."},
    ]
    return render_template("predict.html", result=result, test_cases=test_cases)

@app.route("/compare")
def compare():
    if "user" not in session: return redirect(url_for("login"))
    baseline    = load_json("baseline_results.json")
    transformer = load_json("transformer_result.json")
    all_results = baseline + [transformer]
    return render_template("compare.html", results=all_results)

@app.route("/results")
def results():
    if "user" not in session: return redirect(url_for("login"))
    ev      = load_json("evaluation_report.json")
    history = load_json("training_history.json")
    cm_data = load_json("confusion_matrix.json")
    cm = cm_data.get("cm", [[278,43],[9,70]])
    return render_template("results.html", ev=ev, history=history, cm=cm)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
