# Wintap DMBD: LLM Behavioral Analyst Baseline & Clean Temporal Evaluation
This directory (`/home/rornelas5/Wintap_Baseline/agent_evaluation`) implements the **LLM Behavioral Analyst Baseline** for the Wintap DMBD benchmark under a clean temporal evaluation protocol. It evaluates:
1. **The LLM Behavioral Analyst Baseline**: Zero-Shot and In-Context Retrieval-Assisted LLM agents reasoning over structured, deterministic execution traces and process tree hierarchies.
2. **The Original Tabular Baseline (Reference)**: Clean 200-tree Random Forest trained on 9 scalar event counts strictly on non-2024 data ($\le 2023$).

---

## 1. Strict Temporal Evaluation Protocol & No-Leakage Guarantees

To ensure rigorous temporal evaluation without data leakage:

1. **Held-Out Evaluation Set (Year == 2024)**:
   - Contains **every sample with `year == 2024`** from **both** the supplied training and test label files.
   - Total evaluation samples: **14,619** (12,252 benign, 2,367 malicious).
2. **Strict Removal of All 2024 Samples from Development**:
   - ALL 2024 samples (benign and malicious) are strictly excluded from model training, demonstration pools, nearest-neighbor retrieval, prompt examples, calibration, and threshold tuning.
3. **Training & Demonstration Pool (Years <= 2023)**:
   - The clean training set consists strictly of the non-2024 samples from `truth_labels_train.json` (**32,996** samples).
   - The 7,004 year-2024 benign samples in the training file are reassigned to the 2024 evaluation set.
   - The candidate retrieval pool for few-shot demonstrations is strictly filtered by `year < 2024`.
4. **No Target Sample Label or Year Leakage**:
   - The target sample's ground truth label is **never** included in its prompt, retrieved examples, filename, output, logs, or model-accessible tool responses.
   - The calendar year is **never** provided in the model prompt (timestamps are converted into relative elapsed execution offsets).
   - No internet access or external malware reputation lookups (VirusTotal, hash queries) are used.
5. **Anonymized Reasoned Evidence**:
   - Behavioral analysis relies solely on dynamic trace patterns: process tree depth, branching factors, rule frequencies, entity diversity, per-process action profiles, and chronological event transitions.
   - Anonymized tokens (e.g. `P123`, `F456`, `R789`) are not assigned synthetic meanings.

---

## 2. Reproduced Clean Random Forest Baseline

Trained `RandomForestClassifier(random_state=1, n_estimators=200, n_jobs=4)` on all 32,996 non-2024 training samples using `baseline.FEATURES`. Evaluated on all 14,619 year-2024 samples.

| Metric | Reference / Reproduced Value | Status |
|---|---:|:---:|
| **Evaluation Samples ($n$)** | **14,619** | Exact Match |
| **Benign Samples** | 12,252 | Exact Match |
| **Malicious Samples** | 2,367 | Exact Match |
| **True Negatives ($TN$)** | 11,372 | Exact Match |
| **False Positives ($FP$)** | 880 | Exact Match |
| **False Negatives ($FN$)** | 311 | Exact Match |
| **True Positives ($TP$)** | 2,056 | Exact Match |
| **Accuracy** | **91.853%** (0.918531) | Exact Match |
| **Malware Recall** | **86.861%** (0.868610) | Exact Match |
| **Malware Precision** | **70.027%** (0.700272) | Exact Match |
| **Malware F1** | **77.541%** (0.775410) | Exact Match |
| **Benign False Positive Rate** | **7.183%** (0.071825) | Exact Match |
| **Balanced Accuracy** | **89.839%** (0.898393) | Exact Match |
| **ROC AUC** | **0.950240** | Exact Match |
| **Partial ROC AUC (max FPR ≤ 0.1)** | **0.880654** | Exact Match |

*Artifact*: `rf_clean_2024_predictions.csv` and `rf_metrics.csv`.

---

## 3. Dynamic Trace Representation & Preprocessing

The raw Wintap tree files (`trees_0.json` through `trees_7.json`, ~8 GB and 43M events) are preprocessed into compact, deterministic behavioral trace summaries:

- **Rule Counts**: Counts of all events across the 9 baseline rules plus detonation launcher.
- **Entity Diversity**: Unique process GUIDs, unique files created, unique images/DLLs loaded, unique registry keys touched, unique network connection targets.
- **Process Tree Structure**:
  - Root process identified from detonation launcher.
  - Linear-time BFS process depth calculation ($O(V + E)$, immune to cycle hangs).
  - Maximum branching factor and root direct child process count.
- **Per-Process Profiles**: Top 5 most active processes, their role, actions breakdown, and child processes spawned.
- **Chronological Event Transitions**: Run-length encoded event stream (e.g. `image_load x15 -> process_create -> reg_create_key x4 -> ...`), bounded to prevent token explosion.
- **Temporal Dynamics**: Relative duration in seconds, average event rate, and peak 1-second burst rate.
- **Storage**: Indexed in `trace_summaries.db` (SQLite) and `trace_summaries/shard_{0..7}.jsonl` for instant sub-millisecond retrieval.

---

## 4. Local LLM Agent Architecture

The evaluator (`run_agent_evaluation.py`) supports any OpenAI-compatible local model server (such as `muse-glimmer`):

### Model Configuration Environment Variables
```bash
export WINTAP_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export WINTAP_LLM_API_KEY="EMPTY"
export WINTAP_LLM_MODEL="muse-glimmer-30b-dynamic"
```

### Supported Modes
1. **`zero_shot`**: Single-turn evaluation of the target trace summary with expert behavioral analyst instructions.
2. **`retrieval_assisted`**: $k$-NN retrieval ($k=3$) from strictly pre-2024 historical samples using cosine similarity on normalized event features. Appends verified historical classifications and trace summaries as reference cases. Retrieved IDs are recorded for full auditability.
3. **`tool_agent`**: Interactive loop enabling the agent to inspect bounded trace views.

### Resilience & Auditability
- **Resumable Cache**: Every API response is cached in SQLite (`cache/llm_cache.db`) by `experiment_id` and SHA-256 hash of the prompt and configuration. Runs can be stopped and resumed safely without re-querying the model.
- **Strict JSON Enforcement**: Responses are validated against schema (`prediction`, `malware_probability`, `confidence`, `evidence`, `uncertainty_reason`).
- **Explicit Failure Handling**: API timeouts or malformed JSON trigger retries (exponential backoff); failed samples are marked with explicit error status and **never** silently defaulted to benign.
- **Audit Log**: `per_sample_audit.csv` records experiment ID, configuration, prompt hash, retrieved IDs, status, prediction, probability, confidence, latency, and token counts.

---

## 5. Execution Guide

### Step 1: Run Verification & Unit Tests
Run on a Slurm compute node or submit via sbatch:
```bash
sbatch /home/rornelas5/Wintap_Baseline/agent_evaluation/run_tests_slurm.sh
```
*Proves*:
- No 2024 sample enters retrieval or demonstrations.
- All 14,619 year-2024 samples are evaluated once.
- Random forest exactly matches reference metrics.
- Malformed model responses fail explicitly.

### Step 2: Launch Local Muse-Glimmer Server
Allocate a GPU node and launch the pinned `muse-glimmer` server:
```bash
# On an A100 GPU:
srun --partition=gpu --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G \
  --time=04:00:00 --pty bash

# Inside the GPU compute shell:
cd ~/OpenSource_Agentic_Model_Setup
source scripts/activate-muse-gguf.sh
MUSE_GPU=a100 bash scripts/serve-muse-gguf.sh
```
*Or submit as a background Slurm job*:
```bash
sbatch /home/rornelas5/Wintap_Baseline/agent_evaluation/run_muse_server_gpu.sh
```

### Step 3: Run the 20-Sample Pilot Evaluation
Once the server is running on `127.0.0.1:8000`:
```bash
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --base-url http://127.0.0.1:8000/v1 \
  --model muse-glimmer-30b-dynamic
```
*Offline verification without running GPU*:
```bash
python3 /home/rornelas5/Wintap_Baseline/agent_evaluation/run_agent_evaluation.py \
  --config all \
  --split pilot_20 \
  --mock
```

### Step 4: Run the Full 2024 Evaluation
```bash
sbatch /home/rornelas5/Wintap_Baseline/agent_evaluation/run_full_eval.sh
```
The run is fully cached and resumable; if interrupted, it resumes from the last completed sample.

---

## 6. Artifact Inventory

- `reproduce_rf.py`: Reproduces clean Random Forest baseline on 2024 split.
- `generate_all_summaries.py`: Precomputes trace summaries from raw trees into SQLite and JSONL.
- `run_agent_evaluation.py`: Resumable CLI evaluator for local LLM agents.
- `test_evaluation_protocol.py`: Self-check unit tests for no-leakage, RF reproduction, and JSON validation.
- `rf_clean_2024_predictions.csv`: Clean RF predictions on all 14,619 year-2024 samples.
- `rf_metrics.csv`: Full metric suite for RF temporal reference.
- `agent_zero_shot_predictions.csv`: Pilot predictions for zero-shot agent.
- `agent_retrieval_assisted_predictions.csv`: Pilot predictions for retrieval-assisted agent.
- `per_sample_audit.csv`: Complete audit log with prompt hashes, retrieval IDs, latencies, tokens.
- `metrics.csv`: Combined metric comparison table.
- `RESULTS.md`: Detailed comparison and behavioral findings.
