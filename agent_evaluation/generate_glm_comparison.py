import sqlite3, json, pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from pathlib import Path

def get_metrics(df, pred_col, prob_col, name):
    y_true = df['label'].eq('malicious')
    y_pred = df[pred_col].eq('malicious')
    y_prob = df[prob_col].astype(float)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    acc = (tp + tn) / len(df)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    
    both = (y_true.nunique() == 2)
    auc = roc_auc_score(y_true, y_prob) if both else None
    try:
        p_auc = roc_auc_score(y_true, y_prob, max_fpr=0.1) if both else None
    except Exception:
        p_auc = None
        
    return {
        'Model': name,
        'n': len(df),
        'accuracy': round(float(acc), 6),
        'malware_recall': round(float(rec), 6),
        'malware_precision': round(float(prec), 6),
        'malware_f1': round(float(f1), 6),
        'benign_fpr': round(float(fpr), 6),
        'roc_auc': round(float(auc), 6) if auc is not None else None,
        'partial_auc_max_fpr_0_1': round(float(p_auc), 6) if p_auc is not None else None,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def main():
    root = Path('/home/rornelas5/Wintap_Baseline')
    conn = sqlite3.connect(root / 'agent_evaluation/cache/llm_cache.db')
    c = conn.cursor()
    rows = c.execute("""
        SELECT experiment_id, parsed_json, status, latency 
        FROM response_cache 
        WHERE model_name = 'glm-5.3-flash' AND status = 'success'
    """).fetchall()

    glm_records = []
    for exp_id, p_json, status, lat in rows:
        data = json.loads(p_json)
        glm_records.append({
            'experiment': exp_id,
            'prediction_glm': data.get('prediction'),
            'malware_probability_glm': float(data.get('malware_probability', 0.5)),
            'confidence_glm': float(data.get('confidence', 0.5)),
            'latency_glm': lat
        })

    df_glm = pd.DataFrame(glm_records)
    test_orig = pd.read_csv(root / 'year_evaluation/test_predictions.csv')
    merged = test_orig.merge(df_glm, on='experiment')

    # Save matched 21,655 predictions
    out_preds = root / 'agent_evaluation/agent_glm_official_test_predictions.csv'
    merged.to_csv(out_preds, index=False)
    print(f"Saved 21,655 matched predictions to {out_preds}")

    # 1. Full matched official test set (N = 21,655)
    m_rf_full = get_metrics(merged, 'prediction', 'malware_probability', 'Random Forest (Original Baseline)')
    m_glm_full = get_metrics(merged, 'prediction_glm', 'malware_probability_glm', 'GLM-5.3-Flash (Zero-Shot LLM)')

    # 2. 2024 test cohort (N = 6,677)
    m_24 = merged[merged['year'] == 2024]
    m_rf_24 = get_metrics(m_24, 'prediction', 'malware_probability', 'Random Forest (Original Baseline)')
    m_glm_24 = get_metrics(m_24, 'prediction_glm', 'malware_probability_glm', 'GLM-5.3-Flash (Zero-Shot LLM)')

    summary = pd.DataFrame([
        {'Evaluation Scope': 'Official Test Set (Full Matched N=21,655)', **m_rf_full},
        {'Evaluation Scope': 'Official Test Set (Full Matched N=21,655)', **m_glm_full},
        {'Evaluation Scope': 'Official Test Set (2024 Matched N=6,677)', **m_rf_24},
        {'Evaluation Scope': 'Official Test Set (2024 Matched N=6,677)', **m_glm_24},
    ])

    out_csv = root / 'agent_evaluation/glm_official_test_comparison.csv'
    summary.to_csv(out_csv, index=False)
    print(f"Saved comparison summary to {out_csv}")
    print(summary[['Evaluation Scope', 'Model', 'n', 'accuracy', 'malware_recall', 'malware_precision', 'malware_f1', 'benign_fpr', 'roc_auc']].to_string(index=False))

if __name__ == '__main__':
    main()
