# Evaluation Results: Random Forest Baseline vs. LLM Agent on Wintap DMBD 2024

This report evaluates behavioral malware detection on the Wintap DMBD held-out **year-2024** evaluation set under a strict, no-leakage temporal benchmark. We compare the clean 200-tree Random Forest baseline against the LLM Agent (`qwen3.8-flash`) in both **Zero-Shot** and **Retrieval-Assisted** configurations, with exact matched-subset analyses.

---

## 1. Executive Summary & Matched Benchmark Tables

All evaluations operate on the held-out year-2024 temporal distribution ($N = 14,619$ total samples: 12,252 benign, 2,367 malicious). Training candidates and retrieval memory are strictly restricted to samples from **years $\le$ 2023** (32,996 clean pre-2024 reference samples).

### A. Matched 3-Way Head-to-Head Comparison ($N = 5,613$ Identical Samples)
This table compares the **exact same 5,613 year-2024 samples** (4,699 benign, 914 malicious) evaluated across all three models under identical testing conditions:

| Metric | Random Forest (Original Baseline) | Zero-Shot LLM Agent (`qwen3.8-flash`) | Retrieval-Assisted LLM Agent (`qwen3.8-flash` + $k$-NN) | Relative Impact of Retrieval ($\Delta$) |
|---|---:|---:|---:|---:|
| **Evaluation Set ($n$)** | **5,613** | **5,613** | **5,613** | Matched |
| **Input Representation** | 9 scalar event counts | Bounded behavioral trace summary | Trace summary + 3 historical demos | +3 historical exemplars |
| **Accuracy** | **92.393%** | 82.843% | **88.545%** | **+5.702%** |
| **Malware Recall** | **87.418%** (799/914) | 21.991% (201/914) | **81.838%** (748/914) | **+59.847%** |
| **Malware Precision** | **71.917%** | 44.568% | **61.061%** | **+16.493%** |
| **Malware F1** | **78.914%** | 29.451% | **69.939%** | **+40.488%** |
| **Benign FPR** | 6.640% (312/4,699) | **5.320%** (250/4,699) | 10.151% (477/4,699) | +4.831% |
| **Balanced Accuracy** | **90.389%** | 58.336% | **85.843%** | **+27.507%** |
| **ROC AUC** | **0.9519** | 0.7467 | **0.9137** | **+0.1670** |
| **Partial ROC AUC (max FPR $\le$ 0.1)** | **0.8886** | 0.5712 | **0.7867** | **+0.2155** |
| **Coverage** | 100.0% (5,613/5,613) | 100.0% (5,613/5,613) | 100.0% (5,613/5,613) | Complete |
| **Average Prompt Tokens** | N/A (Tabular) | ~870 tokens | ~1,350 tokens | +480 tokens |
| **Average Completion Tokens** | N/A | ~830 tokens | ~850 tokens | +20 tokens |

---

### B. Matched Zero-Shot Head-to-Head Comparison ($N = 14,610$ Identical Samples)
The Zero-Shot LLM Agent ran to near-complete coverage across the entire year-2024 test split (14,610 valid outputs out of 14,619 total samples, 99.94% coverage; 12,243 benign, 2,367 malicious). Here is the exact head-to-head on those 14,610 samples:

| Metric | Random Forest (Original Baseline) | Zero-Shot LLM Agent (`qwen3.8-flash`) | Difference ($\Delta$) |
|---|---:|---:|---:|
| **Evaluation Set ($n$)** | **14,610** | **14,610** | Matched |
| **Accuracy** | **91.855%** | 83.025% | -8.830% |
| **Malware Recall** | **86.861%** (2,056/2,367) | 23.954% (567/2,367) | -62.907% |
| **Malware Precision** | **70.051%** | 45.469% | -24.582% |
| **Malware F1** | **77.556%** | 31.378% | -46.178% |
| **Benign FPR** | 7.180% (879/12,243) | **5.554%** (680/12,243) | **-1.626%** (LLM lower false alarm rate) |
| **Balanced Accuracy** | **89.840%** | 59.200% | -30.640% |
| **ROC AUC** | **0.9502** | 0.7458 | -0.2044 |
| **Partial ROC AUC (max FPR $\le$ 0.1)** | **0.8807** | 0.5757 | -0.3050 |
| **Coverage** | 100.0% | 99.94% (14,610 / 14,619) | 9 failures (network/schema) |

---

### C. Reference Random Forest Baseline on Full 2024 Dataset ($N = 14,619$)
For completeness, the clean 200-tree Random Forest baseline on all 14,619 year-2024 samples:
- **Accuracy**: 91.853% (13,428 / 14,619)
- **Malware Recall**: 86.861% (2,056 / 2,367)
- **Malware Precision**: 70.027% (2,056 / 2,936)
- **Malware F1**: 77.541%
- **Benign FPR**: 7.183% (880 / 12,252)
- **Balanced Accuracy**: 89.839%
- **ROC AUC**: 0.9502
- **Partial AUC (max FPR $\le$ 0.1)**: 0.8807

---

## 2. In-Depth Behavioral & Error Analysis

### A. The Transformative Role of In-Context Retrieval ($k$-NN)
The difference between Zero-Shot and Retrieval-Assisted agent performance is the central scientific finding of this benchmark:
1. **Recall Surge**: Zero-shot agent recall is only **21.99%** on the matched subset. Without domain context, the LLM defaults to extreme conservatism, classifying ambiguous or subtle behavioral traces as benign to avoid false alarms (yielding a very low 5.32% FPR).
2. **Retrieval Grounding**: By retrieving $k=3$ clean historical exemplars ($\le 2023$) similar in behavioral profile, malware recall leaps from **21.99% to 81.84%** (+59.85 percentage points).
3. **ROC AUC Convergence**: The retrieval-assisted agent achieves an ROC AUC of **0.9137**, approaching the Random Forest baseline (0.9519), despite having never been trained or fine-tuned on the Wintap dataset.

### B. Complementary Strengths & Disagreement Analysis
On the matched 5,613 samples, evaluating where the models agree and diverge reveals that the LLM agent captures qualitative structural dynamics that tabular feature counts miss:

```
Total Matched Samples: 5,613
  - Both RF and Retrieval-Agent Correct:  4,830 (86.05%)
  - RF Correct, Retrieval-Agent Wrong:      356  (6.34%)
  - Retrieval-Agent Correct, RF Wrong:      140  (2.49%)
  - Both Wrong:                             287  (5.11%)
```

#### Malicious Sample Breakdown ($N = 914$):
- **Detected by Random Forest**: 799 (87.42%)
- **Detected by Retrieval-Assisted LLM**: 748 (81.84%)
- **Unique Catches by LLM (Missed by RF)**: **27 malicious samples** (2.95% of all malware).
  - *Trace Characteristic*: These samples exhibited low total event counts (evading the Random Forest's numerical threshold triggers), but contained anomalous structural sequences (e.g. process injection, unexpected child process spawning from background utilities, or targeted registry modifications followed by immediate process exit). The LLM's behavioral parser flagged these qualitative red flags despite low scalar counts.
- **Unique Catches by RF (Missed by LLM)**: **78 malicious samples** (8.53%).
  - *Trace Characteristic*: High-volume event floods (e.g., thousands of registry queries or file writes) where the overall process hierarchy appeared standard, triggering tree splits in Random Forest but blending into standard administrative churn for the language model.
- **Ensemble Upper Bound**: If a simple union rule is applied (flagged if *either* RF or Retrieval LLM predicts malicious), total malware recall reaches **90.37%** (826 / 914), surpassing either standalone model.

---

## 3. Computational & Audit Integrity

- **Strict Temporal Isolation**: Audited via `test_evaluation_protocol.py`. Zero 2024 samples entered the retrieval pool (all 32,996 reference cases originated from $\le 2023$).
- **Deterministic Caching**: All evaluated samples are committed to `cache/llm_cache.db`.
- **Reproducibility Artifacts**:
  - `matched_baseline_comparison.csv`: Direct matched metric tables.
  - `official_wintap_test_comparison.csv`: Direct matched metrics on official Wintap test set.
  - `agent_zero_shot_predictions.csv`: 14,619 per-sample predictions for Zero-Shot.
  - `agent_retrieval_assisted_predictions.csv`: Per-sample predictions for Retrieval-Assisted.
  - `rf_clean_2024_predictions.csv`: Exact Random Forest benchmark predictions.
  - `per_sample_audit.csv`: Traceability log containing retrieved historical IDs and prompt hashes.

---

## 4. Comparison on the Official Wintap Test Split (`truth_labels_test.json`)

The original Wintap DMBD repository (`LLNL/Wintap-Analytics` 2025-dmbd) evaluated on the official test set of $N = 24,747$ samples from `truth_labels_test.json`, reporting **~95.4% test accuracy** (locally reproduced as **92.51%** aggregate test accuracy).

### Decomposition of the Official Test Set:
1. **2017 Malware Cohort ($N = 17,131$)**: **100% malicious**. The model was trained on contemporaneous 2017 malware samples ($N = 13,806$), achieving **93.55% recall** with zero benign samples to misclassify. This 69.2% majority cohort heavily inflates aggregate accuracy.
2. **2024 Test Cohort ($N = 7,615$)**: 5,248 benign, 2,367 malicious. This cohort represents genuine forward temporal generalization. On this cohort, the original Random Forest baseline drops to **90.19% accuracy**.

### Direct Head-to-Head on the Official 2024 Test Cohort:
Evaluating the models on the exact samples from the official test split:

| Evaluation Scope | Model | $n$ | Accuracy | Malware Recall | Malware Precision | Malware F1 | Benign FPR | ROC AUC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **Official Test Set (Full)** | Original Random Forest Baseline | 24,747 | **92.512%** | 92.574% | 97.805% | 95.118% | 7.716% | 0.9725 |
| **Official Test Set (2024 Cohort)** | Original Random Forest Baseline | 7,615 | **90.190%** | 85.509% | 83.361% | 84.421% | 7.698% | 0.9367 |
| **Official Test Set (2024 Cohort)** | Zero-Shot LLM Agent (`qwen3.8-flash`) | 7,613 | 71.050% | 23.954% | 58.393% | 33.972% | 7.701% | 0.7026 |
| **Official Test Set (Matched Cohort)** | Original Random Forest Baseline | 2,910 | **90.962%** | 85.449% | 85.730% | 85.589% | 6.513% | 0.9394 |
| **Official Test Set (Matched Cohort)** | Retrieval-Assisted LLM Agent (`qwen3.8-flash`) | 2,910 | **84.570%** | **81.838%** | 72.551% | 76.915% | 14.178% | **0.8934** |

On the matched 2,910 official test samples:
- The Retrieval-Assisted LLM Agent achieves **81.84% recall** (closely approaching Random Forest's 85.45%).
- The Retrieval LLM detected **37 malicious samples that Random Forest completely missed**.
- Combining both detectors yields an ensemble recall of **89.50%** (818 / 914).

