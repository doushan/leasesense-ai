import os, json, joblib, re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load models once at startup ──────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, 'models')

scaler       = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
with open(os.path.join(MODELS_DIR, 'config.json')) as f:
    config = json.load(f)

MODELS = {
    'random_forest':       joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl')),
    'logistic_regression': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl')),
    'decision_tree':       joblib.load(os.path.join(MODELS_DIR, 'decision_tree.pkl')),
}

MODEL_LABELS = {
    'random_forest':       'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', config=config, model_labels=MODEL_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_key = data.get('model', 'random_forest')
        model = MODELS.get(model_key)
        if model is None:
            return jsonify({'error': 'Invalid model selected'}), 400

        # ── Build feature vector ──────────────────────────────────────────
        raw = data.get('features', {})

        def f(key, default=0):
            try: return float(raw.get(key, default))
            except: return float(default)

        disbursed   = f('disbursed_amount', 50000)
        asset_cost  = f('asset_cost', 65000)
        ltv         = f('ltv', 75)
        credit_score = f('credit_score', 0)
        age         = f('age', 35)
        emp_type    = raw.get('employment_type', 'Unknown')
        credit_hist = f('credit_history_months', 0)

        pri_accts   = f('pri_accounts', 0)
        pri_active  = f('pri_active', 0)
        pri_overdue = f('pri_overdue', 0)
        pri_balance = f('pri_balance', 0)
        pri_sanc    = f('pri_sanctioned', 0)
        pri_disb    = f('pri_disbursed', 0)
        pri_instal  = f('pri_instal', 0)

        sec_accts   = f('sec_accounts', 0)
        sec_active  = f('sec_active', 0)
        sec_overdue = f('sec_overdue', 0)
        sec_balance = f('sec_balance', 0)
        sec_sanc    = f('sec_sanctioned', 0)
        sec_disb    = f('sec_disbursed', 0)
        sec_instal  = f('sec_instal', 0)

        new_accts   = f('new_accts_6m', 0)
        delinquent  = f('delinquent_6m', 0)
        inquiries   = f('inquiries', 0)
        avg_age     = f('avg_acct_age', 0)

        # Engineered
        ltv_ratio        = disbursed / asset_cost if asset_cost > 0 else 0
        credit_hist_flag = 1 if credit_score == 0 else 0
        act_ratio        = pri_active / pri_accts if pri_accts > 0 else 0
        delinq_rate      = pri_overdue / pri_accts if pri_accts > 0 else 0

        row = {
            'disbursed_amount': disbursed, 'asset_cost': asset_cost, 'ltv': ltv,
            'PERFORM_CNS.SCORE': credit_score,
            'PRI.NO.OF.ACCTS': pri_accts, 'PRI.ACTIVE.ACCTS': pri_active,
            'PRI.OVERDUE.ACCTS': pri_overdue, 'PRI.CURRENT.BALANCE': pri_balance,
            'PRI.SANCTIONED.AMOUNT': pri_sanc, 'PRI.DISBURSED.AMOUNT': pri_disb,
            'SEC.NO.OF.ACCTS': sec_accts, 'SEC.ACTIVE.ACCTS': sec_active,
            'SEC.OVERDUE.ACCTS': sec_overdue, 'SEC.CURRENT.BALANCE': sec_balance,
            'SEC.SANCTIONED.AMOUNT': sec_sanc, 'SEC.DISBURSED.AMOUNT': sec_disb,
            'PRIMARY.INSTAL.AMT': pri_instal, 'SEC.INSTAL.AMT': sec_instal,
            'NEW.ACCTS.IN.LAST.SIX.MONTHS': new_accts,
            'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS': delinquent,
            'NO.OF_INQUIRIES': inquiries,
            'avg_acct_age_months': avg_age, 'credit_history_months': credit_hist,
            'ltv_ratio': ltv_ratio, 'credit_history_flag': credit_hist_flag,
            'account_activity_ratio': act_ratio, 'pri_delinquency_rate': delinq_rate,
            'age_at_disbursement': age, 'employment_tenure_proxy': credit_hist,
            'emp_Salaried':    1 if emp_type == 'Salaried' else 0,
            'emp_Self employed': 1 if emp_type == 'Self employed' else 0,
            'emp_Unknown':     1 if emp_type == 'Unknown' else 0,
        }

        input_df = pd.DataFrame([row])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_names
        )
        prob_default = float(model.predict_proba(scaled)[0][1])
        prob_renewal = 1 - prob_default

        # Decision engine
        th_high = config['threshold_high']
        th_med  = config['threshold_medium']

        if prob_default < th_high:
            category = 'HIGH'
            label    = 'High Renewal Propensity'
            action   = 'Auto-renew. Send standard renewal notice to client.'
        elif prob_default <= th_med:
            category = 'MEDIUM'
            label    = 'Medium — Follow Up Needed'
            action   = 'Schedule a follow-up call. Review payment history before renewal.'
        else:
            category = 'LOW'
            label    = 'Low — At Risk of Non-Renewal'
            action   = 'Escalate immediately. Assign to senior agent for retention outreach.'

        # Risk factors
        risk_factors = []
        if credit_score == 0:
            risk_factors.append('No credit bureau history on file')
        if ltv_ratio > 0.85:
            risk_factors.append(f'High loan-to-value ratio ({ltv_ratio:.0%})')
        if pri_overdue > 0:
            risk_factors.append(f'{int(pri_overdue)} overdue primary account(s)')
        if delinquent > 0:
            risk_factors.append(f'{int(delinquent)} delinquent account(s) in last 6 months')
        if age < 25:
            risk_factors.append(f'Young borrower profile (age {int(age)})')
        if credit_hist == 0:
            risk_factors.append('No credit history length available')
        if not risk_factors:
            risk_factors.append('No significant risk factors detected')

        # All three model probs for comparison
        all_probs = {}
        for k, m in MODELS.items():
            all_probs[k] = round(float(m.predict_proba(scaled)[0][1]), 3)

        return jsonify({
            'prob_default': round(prob_default, 3),
            'prob_renewal': round(prob_renewal, 3),
            'category': category,
            'label': label,
            'action': action,
            'risk_factors': risk_factors,
            'all_probs': all_probs,
            'model_used': MODEL_LABELS[model_key],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    return jsonify({
        'results': config['results'],
        'labels': MODEL_LABELS,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
