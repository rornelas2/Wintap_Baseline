#!/usr/bin/env python3
"""
Reproduce clean temporal Random Forest baseline on Wintap DMBD dataset.

Protocol:
- Train set: all non-2024 training samples (truth_labels_train.json where year != 2024; n=32,996).
  Note: The 7,004 2024-benign samples in train are moved to the 2024 evaluation set.
  The non-2024 test samples (17,131 samples from 2017) are not part of the training set.
- Evaluation set: ALL 2024 samples from BOTH train and test splits (n=14,619: 12,252 benign, 2,367 malicious).
- Features: exactly the 9 baseline.FEATURES.
- Model: RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=4).
- Year is metadata only, never an input feature.
- Outputs:
  - rf_clean_2024_predictions.csv
  - rf_metrics.csv
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

FEATURES = [
    'process_create',
    'process_term',
    'reg_create_key',
    'reg_write',
    'file_create',
    'image_load',
    'reg_delete_key',
    'net_connect',
    'reg_delete_value'
]

ROOT = Path("/home/rornelas5/Wintap_Baseline")
FEATURES_CSV = ROOT / "year_evaluation" / "features.csv"
OUT_DIR = ROOT / "agent_evaluation"

def compute_metrics(actual_labels, pred_labels, pred_probs):
    actual = (actual_labels == 'malicious')
    pred = (pred_labels == 'malicious')
    
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[False, True]).ravel()
    both = (actual.nunique() == 2)
    n = len(actual)
    
    acc = (tp + tn) / n if n else 0.0
    mal_recall = tp / (tp + fn) if (tp + fn) else 0.0
    mal_prec = tp / (tp + fp) if (tp + fp) else 0.0
    mal_f1 = (2 * mal_prec * mal_recall / (mal_prec + mal_recall)) if (mal_prec + mal_recall) else 0.0
    benign_fpr = fp / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if both else 0.0
    
    roc_auc = float(roc_auc_score(actual, pred_probs)) if both else None
    partial_auc = float(roc_auc_score(actual, pred_probs, max_fpr=0.1)) if both else None
    
    return {
        "n": int(n),
        "benign": int(tn + fp),
        "malicious": int(tp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(acc),
        "malware_recall": float(mal_recall),
        "malware_precision": float(mal_prec),
        "malware_f1": float(mal_f1),
        "benign_fpr": float(benign_fpr),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": roc_auc,
        "partial_auc_max_fpr_0_1": partial_auc,
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading features from {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Total dataset shape: {df.shape}")
    
    is_2024 = df['year'].eq(2024)
    # The clean temporal protocol:
    # Train set = training split with year != 2024 (n = 32,996)
    # Evaluation set = all 2024 samples across train and test (n = 14,619)
    train_df = df.loc[(~df['test']) & (~is_2024)].copy()
    eval_df = df.loc[is_2024].copy()
    
    print(f"Train (non-2024 training split) samples: {len(train_df)}")
    print(f"  Benign: {train_df['label'].eq('benign').sum()}, Malicious: {train_df['label'].eq('malicious').sum()}")
    print(f"Evaluation (all 2024) samples: {len(eval_df)}")
    print(f"  Benign: {eval_df['label'].eq('benign').sum()}, Malicious: {eval_df['label'].eq('malicious').sum()}")
    
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df['label']
    
    X_eval = eval_df[FEATURES].fillna(0)
    y_eval = eval_df['label']
    
    print("Training RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=4)...")
    clf = RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=4)
    clf.fit(X_train, y_train)
    
    classes = list(clf.classes_)
    print(f"Model classes: {classes}")
    mal_idx = classes.index('malicious')
    
    eval_preds = clf.predict(X_eval)
    eval_probs = clf.predict_proba(X_eval)[:, mal_idx]
    
    # Save predictions
    pred_df = pd.DataFrame({
        "experiment": eval_df["experiment"],
        "label": y_eval.values,
        "prediction": eval_preds,
        "malware_probability": eval_probs
    })
    pred_path = OUT_DIR / "rf_clean_2024_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved RF 2024 predictions to {pred_path}")
    
    # Compute metrics
    m = compute_metrics(y_eval, eval_preds, eval_probs)
    print("\n=== Clean Temporal RF 2024 Results ===")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
            
    # Verify exact match with expected reference from year_evaluation/no_2024_training_evaluation.csv
    expected = {
        "n": 14619,
        "benign": 12252,
        "malicious": 2367,
        "tn": 11372,
        "fp": 880,
        "fn": 311,
        "tp": 2056,
        "accuracy": 0.918531,
        "malware_recall": 0.868610
    }
    
    print("\n=== Verification against expected reference ===")
    all_matched = True
    for k, exp_val in expected.items():
        actual_val = m[k]
        if isinstance(exp_val, float):
            diff = abs(actual_val - exp_val)
            ok = diff < 1e-4
        else:
            ok = (actual_val == exp_val)
        print(f"  {k}: actual={actual_val}, expected={exp_val} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_matched = False
            
    if all_matched:
        print("\n>>> ALL RF REFERENCE CHECKS PASSED EXACTLY! <<<")
    else:
        print("\n>>> WARNING: CHECKS DIFFERED! <<<")
        sys.exit(1)
        
    metrics_df = pd.DataFrame([m])
    metrics_df.to_csv(OUT_DIR / "rf_metrics.csv", index=False)

if __name__ == "__main__":
    main()
