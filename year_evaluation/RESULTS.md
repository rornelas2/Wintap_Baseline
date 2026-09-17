# Local Wintap baseline results by year

The 2024 held-out test subset scores **90.19% accuracy**, **85.51% malware recall**, **83.36% malware precision**, **84.42% malware F1**, and **0.8691 standardized partial ROC AUC (max_fpr=0.1)**. Full ROC AUC is **0.9367**. It detects 2,024 of 2,367 malware samples, misses 343, and flags 404 of 5,248 benign samples (7.70% false-positive rate).

Held-out malware recall drops from 93.55% in 2017 to 85.51% in 2024, a decline of 8.04 percentage points. Malware miss rate increases from 6.45% to 14.49% (about 2.25 times). The class mixture differs between years, so this recall comparison is more informative than comparing overall accuracy. This is consistent with temporal deterioration, but does not by itself isolate age from other differences in the samples.

## Held-out test

| Year | Samples | Benign | Malware | Accuracy | Malware recall | Malware precision | Partial AUC (FPR ≤ 0.1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 24747 | 5249 | 19498 | 92.51% | 92.57% | 97.81% | 0.9315 |
| 2017 | 17131 | 0 | 17131 | 93.55% | 93.55% | 100.00% | — |
| 2024 | 7615 | 5248 | 2367 | 90.19% | 85.51% | 83.36% | 0.8691 |
| unknown | 1 | 1 | 0 | 0.00% | — | 0.00% | — |

## Training split: 10-fold out-of-fold results by year

Each sample is evaluated by a model that excluded its fold. These are mixed-year CV results, not forward temporal tests, and should not be combined with the held-out table as one chronological benchmark. The 2024 CV row contains benign samples only and says nothing about 2024 malware detection. Early years have very small sample sizes.

| Year | Samples | Benign | Malware | Accuracy | Malware recall | Malware precision | Partial AUC (FPR ≤ 0.1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 40000 | 20000 | 20000 | 91.66% | 90.35% | 92.77% | 0.9167 |
| 2007 | 1 | 1 | 0 | 0.00% | — | 0.00% | — |
| 2008 | 2 | 2 | 0 | 50.00% | — | 0.00% | — |
| 2009 | 3 | 2 | 1 | 100.00% | 100.00% | 100.00% | 1.0000 |
| 2010 | 5 | 3 | 2 | 60.00% | 50.00% | 50.00% | 0.7368 |
| 2011 | 4 | 3 | 1 | 75.00% | 100.00% | 50.00% | 0.4737 |
| 2012 | 4 | 3 | 1 | 50.00% | 0.00% | 0.00% | 0.4737 |
| 2013 | 14 | 7 | 7 | 64.29% | 71.43% | 62.50% | 0.4737 |
| 2014 | 43 | 28 | 15 | 67.44% | 86.67% | 52.00% | 0.6216 |
| 2015 | 94 | 34 | 60 | 78.72% | 85.00% | 82.26% | 0.5843 |
| 2016 | 117 | 47 | 70 | 80.34% | 91.43% | 79.01% | 0.7304 |
| 2017 | 15033 | 1227 | 13806 | 91.41% | 92.79% | 97.74% | 0.8572 |
| 2018 | 178 | 125 | 53 | 87.08% | 88.68% | 73.44% | 0.7716 |
| 2019 | 262 | 216 | 46 | 90.84% | 76.09% | 72.92% | 0.8387 |
| 2020 | 2634 | 695 | 1939 | 85.27% | 82.26% | 97.32% | 0.8665 |
| 2021 | 4005 | 2010 | 1995 | 88.96% | 85.21% | 92.04% | 0.8735 |
| 2022 | 3254 | 1906 | 1348 | 91.55% | 90.58% | 89.19% | 0.9052 |
| 2023 | 7343 | 6687 | 656 | 92.05% | 80.34% | 53.67% | 0.8450 |
| 2024 | 7004 | 7004 | 0 | 96.56% | — | 0.00% | — |

## Provenance and limitations

Evaluated the existing `/home/rornelas5/Wintap_Baseline/baseline.py` using `evaluate_by_year.py`, without changing the baseline. Nine event-count features; 200-tree random forest; random_state=1. Year is metadata only. Slurm job 340011 completed successfully in 4 minutes 43 seconds. All per-year counts, accuracies, and malware recalls were independently recomputed from saved prediction files and matched the metrics table.

No 2024 malware appears in training; 7,004 benign 2024 samples do. The local test labels have only 2017, 2024, and one unknown-year sample. Thus there are no official held-out scores for 2018–2023. Local label totals are 40,000 train and 24,747 test, whereas the published description lists 25,416 test samples. The 95.4% published accuracy is not this local baseline run: the locally reproduced aggregate test accuracy is 92.51%, and CV accuracy is 91.66%. The cause of the difference from published performance has not been established.

LLNL defines year as approximate first-seen year: https://github.com/LLNL/Wintap-Analytics/tree/main/2025-dmbd . The research challenge is described at https://gdo-wintap.llnl.gov/data/newdocs/DMBD/ .

See README.md for the method and artifact inventory, metrics.csv for all metrics and confusion counts, and provenance.json for exact package versions and input identifiers.
