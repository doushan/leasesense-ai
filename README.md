# LeaseSense AI — Lease Renewal Prediction MVP

MSc AIML Dissertation | GOOLAB Dusyant (2504_29109)
University of Technology, Mauritius

---

## What This Is

A Flask web application that uses three trained ML models (Logistic Regression,
Decision Tree, Random Forest) to predict vehicle lease renewal propensity.
Designed for Mauritian vehicle leasing SMEs.

---

## Project Structure

```
lease_app/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Full UI
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── config.json
├── requirements.txt
├── Procfile
├── railway.json
└── README.md
```

---

## Deploy to Railway (Step by Step)

### Step 1: Install Git and push to GitHub

```bash
cd lease_app
git init
git add .
git commit -m "Initial commit - LeaseSense AI MVP"
```

Create a new repo on github.com, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/leasesense-ai.git
git push -u origin main
```

### Step 2: Deploy on Railway

1. Go to https://railway.com
2. Click **New Project**
3. Select **Deploy from GitHub repo**
4. Choose your `leasesense-ai` repo
5. Railway auto-detects Python and installs from requirements.txt
6. Click **Deploy**
7. Go to **Settings > Networking > Generate Domain**
8. Your app is live at the generated URL

### Step 3: Verify

Visit your Railway URL. The app should load showing the prediction interface.

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## Model Performance (Test Set: 200 records)

| Model               | AUC-ROC | F1-Score |
|---------------------|---------|----------|
| Baseline (Majority) | 0.500   | 0.000    |
| Logistic Regression | 0.643   | 0.378    |
| Decision Tree       | 0.548   | 0.280    |
| Random Forest       | 0.611   | 0.265    |

All models outperform the baseline. Logistic Regression achieves
the best AUC-ROC on the 1,000-record SME-scale sample.

---

## Decision Engine

- **High Renewal** (Green): Non-renewal probability < 20% → Auto-renew
- **Medium** (Amber): 20%–50% → Schedule follow-up
- **At Risk** (Red): > 50% → Escalate to senior agent
