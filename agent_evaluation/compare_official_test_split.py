import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from pathlib import Path

def get_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    acc = (tp + tn) / len(y_true)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_true.nunique() == 2 else None
    try:
        p_auc = roc_auc_score(y_true, y_prob, max_fpr=0.1) if y_true.nunique() == 2 else None
    except Exception:
        p_auc = None
    return {
        'n': len(y_true), 'accuracy': round(float(acc), 6), 'malware_recall': round(float(rec), 6),
        'malware_precision': round(float(prec), 6), 'malware_f1': round(float(f1), 6),
        'benign_fpr': round(float(fpr), 6), 'roc_auc': round(float(auc), 6) if auc is not None else None,
        'partial_auc_max_fpr_0_1': round(float(p_auc), 6) if p_auc is not None else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def main():
    root = Path('/home/rornelas5/Wintap_Baseline')
    orig_test = pd.read_csv(root / 'year_evaluation/test_predictions.csv')
    zs = pd.read_csv(root / 'agent_evaluation/agent_zero_shot_predictions.csv')
    ra = pd.read_csv(root / 'agent_evaluation/agent_retrieval_assisted_predictions.csv')

    # 1. Full official test set from baseline.py (N = 24,747)
    m_rf_full_test = get_metrics(orig_test['label'] == 'malicious', orig_test['prediction'] == 'malicious', orig_test['malware_probability'])

    # 2. Official test set 2024 cohort (N = 7,615)
    test_2024 = orig_test[orig_test['year'] == 2024].copy()
    m_rf_2024 = get_metrics(test_2024['label'] == 'malicious', test_2024['prediction'] == 'malicious', test_2024['malware_probability'])

    # 3. LLM Zero-Shot on official 2024 cohort (N = 7,613 valid)
    m_zs = test_2024.merge(zs, on=['experiment', 'label'], suffixes=('_rf', '_zs'))
    v_zs = m_zs[m_zs['status'].isin(['success', 'cached']) & m_zs['prediction_zs'].isin(['benign', 'malicious']) & m_zs['malware_probability_zs'].notna()]
    m_zs_2024 = get_metrics(v_zs['label'] == 'malicious', v_zs['prediction_zs'] == 'malicious', v_zs['malware_probability_zs'])

    # 4. Matched subset on official test set where Retrieval-Assisted ran (N = 2,910)
    m_ra = test_2024.merge(ra, on=['experiment', 'label'], suffixes=('_rf', '_ra'))
    v_ra = m_ra[m_ra['status'].isin(['success', 'cached']) & m_ra['prediction_ra'].isin(['benign', 'malicious']) & m_ra['malware_probability_ra'].notna()]
    
    m_rf_matched_ra = get_metrics(v_ra['label'] == 'malicious', v_ra['prediction_rf'] == 'malicious', v_ra['malware_probability_rf'])
    m_ra_matched = get_metrics(v_ra['label'] == 'malicious', v_ra['prediction_ra'] == 'malicious', v_ra['malware_probability_ra'])

    # Build summary dataframe
    summary = pd.DataFrame([
        {'Evaluation Scope': 'Official Wintap Test Set (Full)', 'Model': 'Original Random Forest Baseline', **m_rf_full_test},
        {'Evaluation Scope': 'Official Wintap Test Set (2024 Cohort)', 'Model': 'Original Random Forest Baseline', **m_rf_2024},
        {'Evaluation Scope': 'Official Wintap Test Set (2024 Cohort)', 'Model': 'Zero-Shot LLM Agent (Qwen 3.8B)', **m_zs_2024},
        {'Evaluation Scope': 'Official Test Set (Matched Cohort)', 'Model': 'Original Random Forest Baseline', **m_rf_matched_ra},
        {'Evaluation Scope': 'Official Test Set (Matched Cohort)', 'Model': 'Retrieval-Assisted LLM Agent (Qwen 3.8B)', **m_ra_matched}
    ])

    out_csv = root / 'agent_evaluation/official_wintap_test_comparison.csv'
    summary.to_csv(out_csv, index=False)
    print(f"Saved official test comparison to {out_csv}")
    print(summary[['Evaluation Scope', 'Model', 'n', 'accuracy', 'malware_recall', 'malware_precision', 'malware_f1', 'benign_fpr', 'roc_auc']].to_string(index=False))

if __name__ == '__main__':
    main()
