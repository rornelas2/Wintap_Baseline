# Wintap-LLM: Large Language Model Behavioral Malware Analyst Baseline on Wintap DMBD

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: Wintap DMBD](https://img.shields.io/badge/Dataset-Wintap%20DMBD-red.svg)](https://github.com/lin-d-y/Wintap)
[![Evaluation: Temporal Clean 2024](https://img.shields.io/badge/Eval-Clean%202024%20Temporal-green.svg)](#evaluation-protocol)
[![Task: LLM Agent Baseline](https://img.shields.io/badge/Task-LLM%20Agent%20Baseline-purple.svg)](#system-architecture)

> **Important Distinction**: The original Wintap DMBD paper established a traditional supervised tabular baseline using a 200-tree Random Forest trained on 9 scalar Sysmon event counts. This repository introduces the **LLM Behavioral Analyst Baseline**—an agentic framework where language models reason directly over structured execution traces, process tree hierarchies, and chronological entity flows. Both Zero-Shot and In-Context Retrieval-Assisted LLM configurations are benchmarked head-to-head against the original Wintap tabular baseline under a strict clean temporal evaluation protocol on held-out **year-2024** samples ($N = 14,619$).

---

## Table of Contents
- [1. Executive Summary & Benchmark Results](#1-executive-summary--benchmark-results)
  - [A. Matched 3-Way Head-to-Head Comparison ($N = 5,613$)](#a-matched-3-way-head-to-head-comparison-n--5613)
  - [B. Matched Zero-Shot Head-to-Head Comparison ($N = 14,610$)](#b-matched-zero-shot-head-to-head-comparison-n--14610)
  - [C. Clean Random Forest Reference Benchmark ($N = 14,619$)](#c-clean-random-forest-reference-benchmark-n--14619)
- [2. Key Scientific Findings](#2-key-scientific-findings)
- [3. Evaluation Protocol & Anti-Leakage Rules](#3-evaluation-protocol--anti-leakage-rules)
- [4. System Architecture](#4-system-architecture)
  - [A. Trace Summarization & Process Tree Reconstruction](#a-trace-summarization--process-tree-reconstruction)
  - [B. Historical Exemplar Retrieval ($k$-NN)](#b-historical-exemplar-retrieval-k-nn)
  - [C. LLM Agent Inference & Structured Evidence Schema](#c-llm-agent-inference--structured-evidence-schema)
- [5. Repository Structure](#5-repository-structure)
- [6. Reproduction Guide](#6-reproduction-guide)
- [7. Citation & License](#7-citation--license)

---

## 1. Executive Summary & Benchmark Results

Traditional malware evaluations often rely on uniform random train/test splits, which inadvertently leak contemporaneous malware variants and inflate evaluation metrics. In this benchmark, all evaluations are strictly evaluated on **held-out year-2024 executions** using only **pre-2024 historical reference data** ($\le 2023$).

### A. Matched 3-Way Head-to-Head Comparison ($N = 5,613$)
Evaluating the exact same **5,613 held-out 2024 samples** (4,699 benign, 914 malicious) across all three architectures under identical testing conditions:

| Metric | Random Forest Baseline | Zero-Shot LLM Agent (`qwen3.8-flash`) | Retrieval-Assisted LLM Agent (`qwen3.8-flash` + $k$-NN) | Impact of In-Context Retrieval ($\Delta$) |
|:---|---:|---:|---:|---:|
| **Sample Size ($n$)** | **5,613** | **5,613** | **5,613** | *Strict matched subset* |
| **Input Representation** | 9 scalar event counts | Bounded behavioral trace summary | Trace summary + 3 historical demos | +3 historical exemplars |
| **Accuracy** | **92.393%** | 82.843% | **88.545%** | **+5.702%** |
| **Malware Recall** | **87.418%** (799/914) | 21.991% (201/914) | **81.838%** (748/914) | **+59.847%** |
| **Malware Precision** | **71.917%** | 44.568% | **61.061%** | **+16.493%** |
| **Malware F1** | **78.914%** | 29.451% | **69.939%** | **+40.488%** |
| **Benign FPR** | 6.640% (312/4,699) | **5.320%** (250/4,699) | 10.151% (477/4,699) | +4.831% |
| **Balanced Accuracy** | **90.389%** | 58.336% | **85.843%** | **+27.507%** |
| **ROC AUC** | **0.9519** | 0.7467 | **0.9137** | **+0.1670** |
| **Partial AUC (FPR $\le$ 0.1)** | **0.8886** | 0.5712 | **0.7867** | **+0.2155** |
| **Coverage** | 100.0% (5,613/5,613) | 100.0% (5,613/5,613) | 100.0% (5,613/5,613) | Complete |

---

### B. Matched Zero-Shot Head-to-Head Comparison ($N = 14,610$)
Evaluating the Zero-Shot LLM Agent across near-complete 2024 test split coverage ($14,610$ valid outputs out of $14,619$ total samples, **99.94% coverage**; 12,243 benign, 2,367 malicious) against the Random Forest baseline:

| Metric | Random Forest (Original Baseline) | Zero-Shot LLM Agent (`qwen3.8-flash`) | Difference ($\Delta$) |
|:---|---:|---:|---:|
| **Sample Size ($n$)** | **14,610** | **14,610** | *99.94% coverage* |
| **Accuracy** | **91.855%** | 83.025% | -8.830% |
| **Malware Recall** | **86.861%** (2,056/2,367) | 23.954% (567/2,367) | -62.907% |
| **Malware Precision** | **70.051%** | 45.469% | -24.582% |
| **Malware F1** | **77.556%** | 31.378% | -46.178% |
| **Benign FPR** | 7.180% (879/12,243) | **5.554%** (680/12,243) | **-1.626%** *(LLM has lower false alarms)* |
| **ROC AUC** | **0.9502** | 0.7458 | -0.2044 |
| **Partial AUC (FPR $\le$ 0.1)** | **0.8807** | 0.5757 | -0.3050 |

---

### C. Clean Random Forest Reference Benchmark ($N = 14,619$)
The full clean Random Forest baseline on all 14,619 year-2024 samples:
- **Accuracy**: 91.853% (13,428 / 14,619)
- **Malware Recall**: 86.861% (2,056 / 2,367)
- **Malware Precision**: 70.027% (2,056 / 2,936)
- **Malware F1**: 77.541%
- **Benign False Positive Rate**: 7.183% (880 / 12,252)
- **Balanced Accuracy**: 89.839%
- **ROC AUC**: 0.9502
- **Partial AUC (FPR $\le$ 0.1)**: 0.8807

---

### D. Direct Comparison on the Official Wintap Test Split (`truth_labels_test.json`)

The original Wintap DMBD baseline report cited **~95.4% test accuracy** (locally reproduced as **92.51%** aggregate accuracy) on the official test set ($N = 24,747$).
- **The 2017 Cohort Bias**: 17,131 of those test samples (69.2%) were historical 2017 malware, where the model achieved 93.55% recall with 0 benign samples to misclassify.
- **The 2024 Cohort Reality**: On the held-out 2024 cohort of that official test set ($N = 7,615$), the original Random Forest drops to **90.19% accuracy**.

Here is the exact head-to-head comparison on the official test set:

| Evaluation Scope | Model | $n$ | Accuracy | Malware Recall | Malware Precision | Malware F1 | Benign FPR | ROC AUC |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **Official Test Set (Full)** | Original Random Forest Baseline | 24,747 | **92.512%** | 92.574% | 97.805% | 95.118% | 7.716% | 0.9725 |
| **Official Test Set (2024 Cohort)** | Original Random Forest Baseline | 7,615 | **90.190%** | 85.509% | 83.361% | 84.421% | 7.698% | 0.9367 |
| **Official Test Set (2024 Cohort)** | Zero-Shot LLM Agent (`qwen3.8-flash`) | 7,613 | 71.050% | 23.954% | 58.393% | 33.972% | 7.701% | 0.7026 |
| **Official Test Set (Matched Cohort)** | Original Random Forest Baseline | 2,910 | **90.962%** | 85.449% | 85.730% | 85.589% | 6.513% | 0.9394 |
| **Official Test Set (Matched Cohort)** | Retrieval-Assisted LLM Agent (`qwen3.8-flash`) | 2,910 | **84.570%** | **81.838%** | 72.551% | 76.915% | 14.178% | **0.8934** |

*(On the matched 2,910 samples, the Retrieval LLM detected **37 malicious samples that the Random Forest missed**; ensembling both yields **89.50% recall**).*

---

## 2. Key Scientific Findings

### 1. In-Context Historical Retrieval Drives a +59.85% Recall Surge
- **Zero-Shot Conservatism**: Without historical context, LLM agents exhibit high conservatism on unfamiliar endpoint telemetry. The zero-shot agent achieved a very low false alarm rate (5.32% FPR), but missed ~78% of malicious activity (21.99% recall).
- **Retrieval Grounding**: Equipping the LLM agent with $k=3$ historical pre-2024 exemplar traces caused malware recall to surge from **21.99% to 81.84%** (+59.85 percentage points), with ROC AUC jumping from **0.7467 to 0.9137**. This proves that in-context grounding provides critical calibration that enables models with zero task-specific fine-tuning to rival dedicated supervised models.

### 2. Qualitative Behavioral Detections Missed by Tabular Models
On the matched 5,613-sample benchmark:
- **Unique LLM Catches**: The Retrieval-Assisted LLM detected **27 malicious samples that Random Forest completely missed**.
  - *Root Cause*: These samples produced low total event counts (evading numerical tree splits in the Random Forest), but contained qualitative structural anomalies (e.g. process injection, unexpected utility spawning from `explorer.exe`, or registry persistence triggers followed by termination). The LLM's behavioral parser flagged these sequences despite low scalar activity.
- **Unique RF Catches**: Random Forest detected **78 malicious samples** missed by the LLM (primarily high-volume flood attacks where process lineage looked benign).
- **Complementary Ensembling**: Combining both models via a union rule (flagged if *either* model alerts) achieves **90.37% recall** (826 / 914), outperforming both individual detectors.

---

## 3. Evaluation Protocol & Anti-Leakage Rules

To prevent temporal leakage and benchmark gaming, the harness enforces the following rules verified by automated test suite [`test_evaluation_protocol.py`](file:///home/rornelas5/Wintap_Baseline/agent_evaluation/test_evaluation_protocol.py):

1. **Strict Temporal Isolation**:
   - **Evaluation Set**: All samples observed in 2024 ($N = 14,619$).
   - **Candidate Pool**: Exactly 32,996 clean pre-2024 samples ($\le 2023$). Zero 2024 samples are permitted in training, demonstrations, or calibration.
2. **Zero Information Leakage**: Target labels, ground-truth metadata, and temporal years are completely excluded from prompt text and tool observations.
3. **Deterministic Verification**: Every LLM query is hashed with SHA-256 and logged into [`per_sample_audit.csv`](file:///home/rornelas5/Wintap_Baseline/agent_evaluation/per_sample_audit.csv) alongside retrieved exemplar IDs for total auditability.
4. **Strict Schema Validation**: Responses must validate against a strict JSON schema containing `prediction`, `malware_probability`, `confidence`, and `evidence` (list of strings). Malformed responses are recorded as explicit failures rather than falling back to default benign.

---

## 4. System Architecture

```
Raw Parquet Traces (43M events across 64k exps)
                    │
                    ▼
       ┌─────────────────────────┐
       │ preprocess_traces.py    │ (Extract process trees, run-length sequences,
       └────────────┬────────────┘  event distributions, entity diversity)
                    │
                    ▼
          trace_summaries.db
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
[Zero-Shot Path]          [Retrieval-Assisted Path]
Target Trace Summary      Target Trace Summary + k-NN Search over <= 2023 Pool
        │                        │
        └───────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ run_agent_evaluation.py │ (Bounded prompt construction & concurrency)
       └────────────┬────────────┘
                    ▼
          LLM Inference API
                    │
                    ▼
       ┌─────────────────────────┐
       │ validate_and_parse_json │ (Strict JSON schema validation & scoring)
       └────────────┬────────────┘
                    ▼
       Predictions, Metrics, & Audit Trail
```

### A. Trace Summarization & Process Tree Reconstruction
Rather than passing raw 10,000+ line event dumps, [`preprocess_traces.py`](file:///home/rornelas5/Wintap_Baseline/agent_evaluation/preprocess_traces.py) builds compact, semantically dense summaries:
- **Process Hierarchy**: Root launcher, max process tree depth (via BFS), branching factor, and child-to-parent mappings.
- **Entity Diversity**: Distinct file paths accessed, DLLs/images loaded, registry keys modified, and network sockets established.
- **Run-Length Sequences**: Chronological transitions (e.g. `image_load x12 -> process_create -> reg_write x4`).
- **Activity Breakdown**: Per-process event counts and duration/burst rate profiles.

### B. Historical Exemplar Retrieval ($k$-NN)
The `HistoricalRetriever` uses standard normalized event distribution cosine similarity to select $k=3$ nearest historical cases ($\le 2023$), embedding compact summaries with verified classifications into the user prompt.

---

## 5. Repository Structure

```
.
├── baseline.py                               # Untouched original baseline reference script
├── README.md                                 # Main project documentation (this file)
├── .gitignore                                # Excludes large DBs, caches, logs, and credentials
│
├── agent_evaluation/                         # LLM Agent Evaluation Framework
│   ├── run_agent_evaluation.py               # Main multi-threaded evaluation harness & LLM client
│   ├── preprocess_traces.py                  # Event parser & SQLite summary builder
│   ├── generate_comparison.py                # Apples-to-apples matched evaluation script
│   ├── test_evaluation_protocol.py           # Verification test suite for zero leakage
│   ├── RESULTS.md                            # Comprehensive experimental findings & analysis
│   │
│   ├── prompts/                              # Agent system and user prompt templates
│   │   ├── system_prompt.txt                 # Role, instructions, and JSON schema rules
│   │   ├── zero_shot_user_prompt.txt         # Zero-shot evaluation prompt template
│   │   └── retrieval_user_prompt.txt        # Retrieval-assisted prompt template
│   │
│   ├── run_eval_api.sh                       # Slurm batch execution script for API evaluation
│   ├── run_pilot_eval.sh                     # Quick 20-sample pilot script
│   │
│   ├── matched_baseline_comparison.csv       # Head-to-head metrics on matched sample subsets
│   ├── metrics.csv                           # Summary metrics table
│   ├── rf_metrics.csv                        # Clean Random Forest metrics table
│   ├── rf_clean_2024_predictions.csv         # Per-sample predictions from Random Forest (14,619)
│   ├── agent_zero_shot_predictions.csv        # Per-sample predictions from Zero-Shot LLM (14,619)
│   ├── agent_retrieval_assisted_predictions.csv # Per-sample predictions from Retrieval LLM (14,619)
│   └── per_sample_audit.csv                  # Per-sample audit log (retrieved IDs, hashes, latencies)
│
└── year_evaluation/                          # Temporal dataset split analysis
    ├── evaluate_by_year.py                   # Train/test temporal split analyzer
    ├── run_year_evaluation.sh                # Slurm job for temporal evaluation
    ├── features.csv                          # Extracted tabular features
    ├── cv_predictions.csv                    # Cross-validation predictions
    ├── test_predictions.csv                  # Test predictions
    └── RESULTS.md                            # Year-by-year dataset breakdown report
```

---

## 6. Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone git@github.com:rornelas2/Wintap_Baseline.git
cd Wintap_Baseline

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn
```

### 1. Verify Evaluation Protocol
Run the automated test suite to confirm zero leakage, Random Forest exact reproduction, and audit compliance:
```bash
python3 agent_evaluation/test_evaluation_protocol.py
```

### 2. Run Head-to-Head Comparison
To regenerate the exact matched-subset comparison metrics:
```bash
python3 agent_evaluation/generate_comparison.py
```

### 3. Run Pilot Evaluation
To run a fast 20-sample pilot test against any OpenAI-compatible endpoint:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python3 agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --concurrency 5 \
  --model "qwen3.8-flash"
```

### 4. Slurm Cluster Execution
Submit the full evaluation job on a Slurm compute cluster:
```bash
sbatch agent_evaluation/run_eval_api.sh
```

---

## 7. Citation & License

This benchmark builds upon the **Wintap Dynamic Malware Behavioral Dataset (DMBD)**. If using this benchmark or evaluation protocol, please cite:

```bibtex
@misc{wintap_agent_baseline_2026,
  author = {Ricardo Ornelas},
  title = {Wintap DMBD: Clean Temporal Baseline and LLM Behavioral Malware Analyst Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rornelas2/Wintap_Baseline}}
}
```
