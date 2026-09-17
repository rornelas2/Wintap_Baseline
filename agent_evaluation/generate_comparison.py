import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from pathlib import Path

def compute_metrics(df, pred_col, prob_col, label_col='label'):
    valid_df = df[df[pred_col].isin(['benign', 'malicious']) & df[prob_col].notna()].copy()
    n_valid = len(valid_df)
    n_total = len(df)
    
    y_true = valid_df[label_col].eq('malicious')
    y_pred = valid_df[pred_col].eq('malicious')
    y_prob = valid_df[prob_col].astype(float)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    
    acc = (tp + tn) / n_valid if n_valid else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (recall + (tn / (tn + fp))) if (tn + fp and (tp + fn)) else 0.0
    
    both_classes = (y_true.nunique() == 2)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if both_classes else None
    try:
        partial_auc = float(roc_auc_score(y_true, y_prob, max_fpr=0.1)) if both_classes else None
    except Exception:
        partial_auc = None
        
    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'coverage': round(n_valid / n_total, 4) if n_total else 0.0,
        'benign_count': int(tn + fp),
        'malicious_count': int(tp + fn),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': round(float(acc), 6),
        'malware_recall': round(float(recall), 6),
        'malware_precision': round(float(prec), 6),
        'malware_f1': round(float(f1), 6),
        'benign_fpr': round(float(fpr), 6),
        'balanced_accuracy': round(float(bal_acc), 6),
        'roc_auc': round(float(roc_auc), 6) if roc_auc is not None else None,
        'partial_auc_max_fpr_0_1': round(float(partial_auc), 6) if partial_auc is not None else None
    }

def main():
    agent_dir = Path('/home/rornelas5/Wintap_Baseline/agent_evaluation')
    rf = pd.read_csv(agent_dir / 'rf_clean_2024_predictions.csv')
    zs = pd.read_csv(agent_dir / 'agent_zero_shot_predictions.csv')
    ra = pd.read_csv(agent_dir / 'agent_retrieval_assisted_predictions.csv')

    rf = rf.rename(columns={'prediction': 'pred_rf', 'malware_probability': 'prob_rf'})
    zs = zs.rename(columns={'prediction': 'pred_zs', 'malware_probability': 'prob_zs', 'status': 'status_zs'})
    ra = ra.rename(columns={'prediction': 'pred_ra', 'malware_probability': 'prob_ra', 'status': 'status_ra'})

    merged = rf.merge(zs, on=['experiment', 'label']).merge(ra, on=['experiment', 'label'])
    
    # 1. Matched subset: 5,613 samples where retrieval-assisted completed
    matched_ra = merged[
        merged['status_ra'].isin(['success', 'cached']) & 
        merged['pred_ra'].isin(['benign', 'malicious']) & 
        merged['prob_ra'].notna() &
        merged['status_zs'].isin(['success', 'cached']) & 
        merged['pred_zs'].isin(['benign', 'malicious']) & 
        merged['prob_zs'].notna()
    ].copy()
    
    print(f"=== MATCHED 3-WAY SUBSET (N = {len(matched_ra)}) ===")
    m_rf_5k = compute_metrics(matched_ra, 'pred_rf', 'prob_rf')
    m_zs_5k = compute_metrics(matched_ra, 'pred_zs', 'prob_zs')
    m_ra_5k = compute_metrics(matched_ra, 'pred_ra', 'prob_ra')
    
    df_5k = pd.DataFrame([
        {'Model': 'Random Forest (Original Baseline)', **m_rf_5k},
        {'Model': 'Zero-Shot LLM Agent (Qwen 3.8B)', **m_zs_5k},
        {'Model': 'Retrieval-Assisted LLM Agent (Qwen 3.8B)', **m_ra_5k},
    ])
    print(df_5k[['Model', 'accuracy', 'malware_recall', 'malware_precision', 'malware_f1', 'benign_fpr', 'roc_auc', 'partial_auc_max_fpr_0_1']].to_string(index=False))

    # 2. Matched subset: 14,610 samples where zero-shot completed
    matched_zs = merged[
        merged['status_zs'].isin(['success', 'cached']) & 
        merged['pred_zs'].isin(['benign', 'malicious']) & 
        merged['prob_zs'].notna()
    ].copy()
    print(f"\n=== MATCHED ZERO-SHOT SUBSET (N = {len(matched_zs)}) ===")
    m_rf_14k = compute_metrics(matched_zs, 'pred_rf', 'prob_rf')
    m_zs_14k = compute_metrics(matched_zs, 'pred_zs', 'prob_zs')
    df_14k = pd.DataFrame([
        {'Model': 'Random Forest (Original Baseline)', **m_rf_14k},
        {'Model': 'Zero-Shot LLM Agent (Qwen 3.8B)', **m_zs_14k},
    ])
    print(df_14k[['Model', 'accuracy', 'malware_recall', 'malware_precision', 'malware_f1', 'benign_fpr', 'roc_auc', 'partial_auc_max_fpr_0_1']].to_string(index=False))

    # 3. Full 14,619 reference
    print(f"\n=== FULL 2024 REFERENCE SET (N = {len(rf)}) ===")
    m_rf_full = compute_metrics(rf, 'pred_rf', 'prob_rf')
    df_full = pd.DataFrame([
        {'Model': 'Random Forest (Full 2024 Reference)', **m_rf_full},
    ])
    print(df_full[['Model', 'accuracy', 'malware_recall', 'malware_precision', 'malware_f1', 'benign_fpr', 'roc_auc', 'partial_auc_max_fpr_0_1']].to_string(index=False))

    # Save detailed comparison table to CSV
    combined_summary = pd.DataFrame([
        {'Evaluation Scope': f'Matched 3-Way (N={len(matched_ra)})', 'Method': 'Random Forest Baseline', **m_rf_5k},
        {'Evaluation Scope': f'Matched 3-Way (N={len(matched_ra)})', 'Method': 'LLM Zero-Shot Agent', **m_zs_5k},
        {'Evaluation Scope': f'Matched 3-Way (N={len(matched_ra)})', 'Method': 'LLM Retrieval-Assisted Agent', **m_ra_5k},
        {'Evaluation Scope': f'Matched Zero-Shot (N={len(matched_zs)})', 'Method': 'Random Forest Baseline', **m_rf_14k},
        {'Evaluation Scope': f'Matched Zero-Shot (N={len(matched_zs)})', 'Method': 'LLM Zero-Shot Agent', **m_zs_14k},
        {'Evaluation Scope': 'Full 2024 Set (N=14,619)', 'Method': 'Random Forest Baseline Reference', **m_rf_full}
    ])
    combined_summary.to_csv(agent_dir / 'matched_baseline_comparison.csv', index=False)
    print(f"\nSaved matched_baseline_comparison.csv to {agent_dir}")

    # Disagreement analysis on 5,613 subset
    print("\n=== ERROR / DISAGREEMENT BREAKDOWN (5,613 matched samples) ===")
    rf_correct = (matched_ra['pred_rf'] == matched_ra['label'])
    ra_correct = (matched_ra['pred_ra'] == matched_ra['label'])
    zs_correct = (matched_ra['pred_zs'] == matched_ra['label'])
    
    print(f"Both RF and Retrieval-Agent Correct: {(rf_correct & ra_correct).sum()} ({(rf_correct & ra_correct).mean():.2%})")
    print(f"RF Correct, Retrieval-Agent Wrong: {(rf_correct & ~ra_correct).sum()} ({(rf_correct & ~ra_correct).mean():.2%})")
    print(f"Retrieval-Agent Correct, RF Wrong: {(~rf_correct & ra_correct).sum()} ({(~rf_correct & ra_correct).mean():.2%})")
    print(f"Both Wrong: {(~rf_correct & ~ra_correct).sum()} ({(~rf_correct & ~ra_correct).mean():.2%})")
    
    # On malicious samples specifically:
    mal_subset = matched_ra[matched_ra['label'] == 'malicious']
    rf_mal_det = (mal_subset['pred_rf'] == 'malicious')
    ra_mal_det = (mal_subset['pred_ra'] == 'malicious')
    zs_mal_det = (mal_subset['pred_zs'] == 'malicious')
    
    print(f"\nMalicious samples specifically (N = {len(mal_subset)}):")
    print(f"  Detected by RF: {rf_mal_det.sum()} ({rf_mal_det.mean():.2%})")
    print(f"  Detected by LLM Zero-Shot: {zs_mal_det.sum()} ({zs_mal_det.mean():.2%})")
    print(f"  Detected by LLM Retrieval-Assisted: {ra_mal_det.sum()} ({ra_mal_det.mean():.2%})")
    print(f"  Detected by LLM Retrieval but MISSED by RF: {(ra_mal_det & ~rf_mal_det).sum()} ({(ra_mal_det & ~rf_mal_det).mean():.2%})")
    print(f"  Detected by RF but MISSED by LLM Retrieval: {(rf_mal_det & ~ra_mal_det).sum()} ({(rf_mal_det & ~ra_mal_det).mean():.2%})")

if __name__ == '__main__':
    main()
