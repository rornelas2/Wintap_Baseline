"""Evaluate the unchanged local baseline on test and out-of-fold training data.

Run: python -u evaluate_by_year.py
Outputs are written to year_evaluation/. Year is metadata, never a feature.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import baseline


OUT = Path(__file__).resolve().parent / "year_evaluation"


def metrics(frame, evaluation, year):
    actual = frame.label.eq("malicious")
    predicted = frame.prediction.eq("malicious")
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    both = actual.nunique() == 2
    return dict(
        evaluation=evaluation, year=year, n=len(frame), benign=int(tn + fp),
        malicious=int(tp + fn), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        accuracy=float((tp + tn) / len(frame)),
        malware_recall=float(tp / (tp + fn)) if tp + fn else None,
        malware_precision=float(tp / (tp + fp)) if tp + fp else None,
        malware_f1=float(2 * tp / (2 * tp + fp + fn)) if tp + fn else None,
        benign_fpr=float(fp / (tn + fp)) if tn + fp else None,
        balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2) if both else None,
        roc_auc=float(roc_auc_score(actual, frame.malware_probability)) if both else None,
        partial_auc_max_fpr_0_1=float(roc_auc_score(actual, frame.malware_probability, max_fpr=0.1)) if both else None,
    )


def summarize(frame, evaluation):
    rows = [metrics(frame, evaluation, "all")]
    for year, group in frame.groupby("year", dropna=False, sort=True):
        rows.append(metrics(group, evaluation, "unknown" if pd.isna(year) else str(int(year))))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    truth = baseline.load_labels()
    assert truth.experiment.is_unique, "Duplicate experiment labels"
    assert set(truth.label) == {"benign", "malicious"}
    print("Building features using baseline.build_feature_matrix() ...", flush=True)
    features = baseline.build_feature_matrix()
    assert features.experiment.is_unique, "An experiment spans multiple feature rows"
    missing = set(truth.experiment) - set(features.experiment)
    assert not missing, f"Missing features for {len(missing)} labeled experiments"
    df = features.merge(truth, on="experiment", how="inner", validate="one_to_one")
    df[baseline.FEATURES] = df[baseline.FEATURES].fillna(0)
    assert np.isfinite(df[baseline.FEATURES].to_numpy()).all()
    df[["experiment", *baseline.FEATURES, "year", "label", "test"]].to_csv(OUT / "features.csv", index=False)
    train, test = df.loc[~df.test].copy(), df.loc[df.test].copy()
    assert len(train) == 40000 and len(test) == 24747
    print(f"Train: {len(train)}; test: {len(test)}; unlabeled feature rows: {len(features)-len(df)}", flush=True)

    # Same estimator, data ordering, features, and decision rule as baseline.py.
    # Bound CPU use; n_jobs changes execution only, not the estimator parameters.
    model = RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=4)
    model.fit(train[baseline.FEATURES], train.label)
    assert list(model.classes_) == ["benign", "malicious"]
    test["prediction"] = model.predict(test[baseline.FEATURES])
    test["malware_probability"] = model.predict_proba(test[baseline.FEATURES])[:, 1]
    columns = ["experiment", "year", "label", "prediction", "malware_probability"]
    test[columns].to_csv(OUT / "test_predictions.csv", index=False)
    rows = summarize(test, "held_out_test")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("Running local baseline's shuffled 10-fold stratified CV ...", flush=True)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    probs = cross_val_predict(model, train[baseline.FEATURES], train.label,
                             cv=folds, method="predict_proba", n_jobs=1)[:, 1]
    train["malware_probability"] = probs
    train["prediction"] = np.where(probs >= 0.5, "malicious", "benign")
    train["fold"] = -1
    for fold, (_, validation) in enumerate(folds.split(train[baseline.FEATURES], train.label)):
        train.iloc[validation, train.columns.get_loc("fold")] = fold
    assert train.fold.between(0, 9).all()
    train[[*columns, "fold"]].to_csv(OUT / "cv_predictions.csv", index=False)
    rows.extend(summarize(train, "training_out_of_fold"))
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    provenance = dict(
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, features=baseline.FEATURES,
        model=model.get_params(), cv=dict(n_splits=10, shuffle=True, random_state=1),
        train_n=len(train), test_n=len(test), feature_n=len(features),
        baseline_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        inputs=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                for p in map(Path, [*baseline.TREE_FILES, baseline.TRAIN_LABELS, baseline.TEST_LABELS])],
        notes=["Year is approximate first-seen year, not a model feature.",
               "CV uses mixed-year training folds; it is not a forward temporal evaluation.",
               "Single-class years have undefined ROC AUC and balanced accuracy.",
               "Test uses classifier.predict; CV uses malware probability >= 0.5, matching baseline.py."],
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved predictions, metrics, features, and provenance to {OUT}", flush=True)


if __name__ == "__main__":
    main()
