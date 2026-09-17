#!/usr/bin/env python3
"""
Wintap DMBD Baseline Implementation
Based on LLNL Wintap-Analytics 2025-dmbd malware_data.ipynb baseline.

Dataset: /home/rornelas5/data/wintap/dmbd
Features: event counts per experiment per RuleName
Model: RandomForestClassifier(random_state=1, n_estimators=200)
Metric: accuracy, ROC AUC (max_fpr=0.1)
Reference baseline from README: 95.0% CV accuracy, 95.4% test accuracy
"""

import pandas as pd
from collections import Counter, OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

DATA_ROOT = "/home/rornelas5/data/wintap/dmbd"
TRAIN_LABELS = f"{DATA_ROOT}/truth_labels_train.json"
TEST_LABELS = f"{DATA_ROOT}/truth_labels_test.json"
TREE_FILES = [f"{DATA_ROOT}/trees_{i}.json" for i in range(8)]

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

def load_labels():
    df_train = pd.read_json(TRAIN_LABELS)
    df_train['test'] = False
    df_test = pd.read_json(TEST_LABELS)
    df_test['test'] = True
    df_truth = pd.concat([df_train, df_test], ignore_index=True)
    return df_truth

def build_feature_matrix():
    df_list = []
    for tf in TREE_FILES:
        df_tree = pd.read_json(tf)
        aggregate = OrderedDict([('RuleName', [Counter])])
        tree_agg = df_tree.groupby('experiment').agg(aggregate).reset_index()
        tree_agg.columns = ['experiment', 'event_counts']
        df_list.append(tree_agg)
    summary_df = pd.concat(df_list, ignore_index=True)
    # explode Counter dicts into columns
    summary_expanded = pd.concat(
        [summary_df.drop('event_counts', axis=1),
         pd.DataFrame(summary_df['event_counts'].tolist())],
        axis=1
    )
    return summary_expanded

def main():
    print("Loading labels...")
    df_truth = load_labels()
    print(f"Total experiments: {len(df_truth)}")

    print("Building feature matrix...")
    summary_expanded = build_feature_matrix()
    df = summary_expanded.merge(df_truth[['experiment', 'label', 'test']],
                                on='experiment', how='inner').fillna(0)
    print(f"Merged shape: {df.shape}")

    df_train = df[df['test'] == False]
    df_test = df[df['test'] == True]

    X_train = df_train[FEATURES]
    X_test = df_test[FEATURES]
    y_train = df_train['label']
    y_test = df_test['label']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 10-fold cross validation on training set
    clf = RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=-1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    y_prob_cv = cross_val_predict(clf, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1)
    y_prob_cv = [p[1] for p in y_prob_cv]
    y_pred_cv = ['malicious' if p >= 0.5 else 'benign' for p in y_prob_cv]

    acc_cv = sum(y_train == y_pred_cv) / len(y_train)
    recall_mal_cv = recall_score(y_train, y_pred_cv, pos_label='malicious')
    precision_mal_cv = precision_score(y_train, y_pred_cv, pos_label='malicious')
    roc_auc_cv = roc_auc_score(y_train, y_prob_cv, max_fpr=0.1)

    print("\n10-Fold Cross-Validation Results")
    print(f"Accuracy: {acc_cv*100:.3f}%")
    print(f"Malware Recall: {recall_mal_cv:.3f}")
    print(f"Malware Precision: {precision_mal_cv:.3f}")
    print(f"ROC AUC (max fpr 0.1): {roc_auc_cv:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_cv))

    # Test set evaluation
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    acc_test = sum(y_test == y_pred_test) / len(y_test)
    recall_mal_test = recall_score(y_test, y_pred_test, pos_label='malicious')
    precision_mal_test = precision_score(y_test, y_pred_test, pos_label='malicious')
    roc_auc_test = roc_auc_score(y_test, y_prob_test, max_fpr=0.1)

    print("\nTest Set Results")
    print(f"Accuracy: {acc_test*100:.3f}%")
    print(f"Malware Recall: {recall_mal_test:.3f}")
    print(f"Malware Precision: {precision_mal_test:.3f}")
    print(f"ROC AUC (max fpr 0.1): {roc_auc_test:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

if __name__ == "__main__":
    main()
