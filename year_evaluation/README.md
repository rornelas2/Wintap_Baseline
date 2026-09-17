# Wintap local baseline: evaluation by first-seen year

## Method

Run `python -u evaluate_by_year.py` from `/home/rornelas5/Wintap_Baseline`.
The evaluator imports the existing `baseline.py` feature builder and uses its nine event-count features and RandomForestClassifier(random_state=1, n_estimators=200). CPU parallelism is bounded to four workers. Year is used only to group results.

The model is fit on the provided training split and evaluated on the provided test split. Additional year-specific training results use predictions from shuffled stratified 10-fold cross-validation (random_state=1), with each prediction made by a model excluding that fold. These CV results are not forward-in-time validation. Test predictions use classifier.predict; CV predictions use malicious probability >= 0.5, matching the local baseline.

## Dataset and interpretation

The local labels contain 40,000 training and 24,747 test samples. Test years are 2017 (malware only), 2024 (both classes), and one benign sample with unknown year. There are no 2024 malware training samples, but there are 7,004 benign 2024 training samples. Earlier training years also exist before 2017. No experiment IDs overlap between train and test.

Year is the approximate first-seen year of the software, according to https://github.com/LLNL/Wintap-Analytics/tree/main/2025-dmbd . The published dataset description lists 25,416 test samples, so the published aggregate baseline is not treated as the result of this local run.

Compare malware recall across test years to avoid the different class mixtures obscuring the detection change. AUC is undefined for single-class subsets. The partial AUC column is sklearn's standardized ROC AUC with max_fpr=0.1. Small year groups are unstable and should be interpreted with their sample counts.

## Artifacts

- `metrics.csv`: overall and per-year metrics for held-out test and training cross-validation.
- `test_predictions.csv`, `cv_predictions.csv`: per-experiment labels, years, predictions, and probabilities; CV also records the fold.
- `features.csv`: event-count feature matrix and labels.
- `split_year_counts.csv`: sample counts by split, year, and class.
- `provenance.json`: package versions, estimator parameters, baseline hash, input sizes and timestamps.
- `slurm-340011.log`: evaluation output from this run.
