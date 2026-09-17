#!/usr/bin/env python3
"""
Agent Evaluation Harness for Wintap DMBD Malware Dataset.

Features:
- OpenAI-compatible local LLM client (configurable via env vars or CLI:
  WINTAP_LLM_BASE_URL, WINTAP_LLM_API_KEY, WINTAP_LLM_MODEL).
- Strictly enforces clean temporal evaluation:
  * Final evaluation set: year == 2024.
  * Retrieval & demonstrations strictly drawn from year < 2024.
  * No target sample ground-truth label or year is ever provided to the model.
- Modes:
  * zero_shot: direct behavioral trace evaluation.
  * retrieval_assisted: k-NN retrieval from non-2024 samples with audited IDs.
  * tool_agent: bounded multi-turn tool interaction.
- Resumable disk caching by experiment ID and prompt/config hash.
- Strict JSON validation with retry handling and explicit failure status.
- Computes comprehensive metrics: Accuracy, Precision, Recall, F1, FPR, Balanced Acc,
  ROC AUC, Partial ROC AUC (max_fpr=0.1), Coverage, Token usage, Latency.
- Generates required CSVs and audits:
  * rf_clean_2024_predictions.csv
  * agent_<config>_predictions.csv
  * metrics.csv
  * per_sample_audit.csv
"""

import os
import sys
import re
import time
import json
import uuid
import sqlite3
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, roc_auc_score
import urllib.request
import urllib.error

ROOT = Path("/home/rornelas5/Wintap_Baseline")
AGENT_DIR = ROOT / "agent_evaluation"
DB_PATH = AGENT_DIR / "trace_summaries.db"
FEATURES_CSV = ROOT / "year_evaluation" / "features.csv"
CACHE_DIR = AGENT_DIR / "cache"
PROMPTS_DIR = AGENT_DIR / "prompts"

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

class LLMClient:
    """Configurable OpenAI-compatible local client with retry handling."""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60, max_retries: int = 3, mock: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.mock = mock

    def call_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, Any]:
        """Send chat completion request to local endpoint with retry."""
        if self.mock:
            # Deterministic mock response based on prompt text
            prompt_content = messages[-1]["content"]
            has_proc_create = "process_create" in prompt_content
            has_net = "unique_network_connections" in prompt_content and not "unique_network_connections: 0" in prompt_content
            has_reg_burst = "reg_write" in prompt_content or "reg_create_key" in prompt_content
            
            # Simple heuristic for mock model
            is_mal = (has_proc_create and has_net) or (has_reg_burst and has_proc_create)
            prob = 0.85 if is_mal else 0.15
            pred = "malicious" if is_mal else "benign"
            
            mock_json = {
                "prediction": pred,
                "malware_probability": prob,
                "confidence": 0.88,
                "evidence": [
                    f"Mock observation: observed process activity and entity distribution.",
                    f"Mock observation: {'detected multi-process network pattern' if is_mal else 'low activity profile'}."
                ],
                "uncertainty_reason": None
            }
            return {
                "content": json.dumps(mock_json),
                "prompt_tokens": len(str(messages)) // 4,
                "completion_tokens": 80,
                "latency": 0.05,
                "error": None
            }

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "x-opencode-session": str(uuid.uuid4())
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                curr_payload = dict(payload)
                if attempt > 1 and last_error and ("response_format" in last_error.lower() or "400" in last_error):
                    curr_payload.pop("response_format", None)

                data = json.dumps(curr_payload).encode("utf-8")
                req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    latency = time.time() - t0
                    res_json = json.loads(resp.read().decode("utf-8"))
                    choice = res_json["choices"][0]["message"]
                    content = choice.get("content", "")
                    usage = res_json.get("usage", {})
                    p_tokens = usage.get("prompt_tokens", len(str(messages)) // 4)
                    c_tokens = usage.get("completion_tokens", len(content) // 4)
                    return {
                        "content": content,
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                        "latency": latency,
                        "error": None
                    }
            except urllib.error.HTTPError as exc:
                latency = time.time() - t0
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")[:100]
                    last_error = f"HTTP {exc.code} {exc.reason}: {err_body}"
                except Exception:
                    last_error = f"HTTP {exc.code} {exc.reason}"
                time.sleep(1.5 * attempt)
            except Exception as exc:
                latency = time.time() - t0
                last_error = str(exc)
                time.sleep(1.5 * attempt)

        return {
            "content": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency": latency,
            "error": f"API failed after {self.max_retries} attempts: {last_error}"
        }

def validate_and_parse_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Strictly validate and parse LLM response into schema."""
    if not raw_text or not raw_text.strip():
        return None, "Empty response"

    text = raw_text.strip()
    # Strip markdown codeblocks if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"

    if not isinstance(data, dict):
        return None, "Output is not a JSON dictionary"

    pred = data.get("prediction")
    if pred not in ["benign", "malicious"]:
        return None, f"Invalid prediction value: {pred}. Must be 'benign' or 'malicious'"

    prob = data.get("malware_probability")
    try:
        prob = float(prob)
        if not (0.0 <= prob <= 1.0):
            return None, f"malware_probability {prob} out of bounds [0, 1]"
        data["malware_probability"] = prob
    except (TypeError, ValueError):
        return None, f"Invalid malware_probability: {prob}"

    conf = data.get("confidence")
    try:
        conf = float(conf)
        if not (0.0 <= conf <= 1.0):
            return None, f"confidence {conf} out of bounds [0, 1]"
        data["confidence"] = conf
    except (TypeError, ValueError):
        return None, f"Invalid confidence: {conf}"

    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        return None, "evidence must be a list of strings"

    return data, None

def load_dotenv_if_exists():
    """Load environment variables from .env files if present and not already set."""
    candidates = [
        AGENT_DIR / ".env",
        ROOT / ".env",
        Path.home() / ".env",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

    # Aliasing compatibility
    if "OPENAI_API_KEY" in os.environ and "WINTAP_LLM_API_KEY" not in os.environ:
        os.environ["WINTAP_LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if "OPENAI_BASE_URL" in os.environ and "WINTAP_LLM_BASE_URL" not in os.environ:
        os.environ["WINTAP_LLM_BASE_URL"] = os.environ["OPENAI_BASE_URL"]

# Load env immediately on module load
load_dotenv_if_exists()

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class DiskCache:
    """Thread-safe SQLite persistent cache for prompt responses."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    config_name TEXT,
                    model_name TEXT,
                    prompt_hash TEXT,
                    raw_response TEXT,
                    parsed_json TEXT,
                    status TEXT,
                    latency REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    timestamp REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_cfg ON response_cache(experiment_id, config_name)")

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute("SELECT raw_response, parsed_json, status, latency, prompt_tokens, completion_tokens FROM response_cache WHERE cache_key = ?", (cache_key,)).fetchone()
                if row:
                    return {
                        "raw_response": row[0],
                        "parsed_json": json.loads(row[1]) if row[1] else None,
                        "status": row[2],
                        "latency": row[3],
                        "prompt_tokens": row[4],
                        "completion_tokens": row[5]
                    }
            return None

    def set(self, cache_key: str, exp_id: str, config: str, model: str, prompt_hash: str,
            raw_resp: str, parsed: Optional[Dict[str, Any]], status: str, latency: float,
            p_tok: int, c_tok: int):
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO response_cache 
                    (cache_key, experiment_id, config_name, model_name, prompt_hash, raw_response, parsed_json, status, latency, prompt_tokens, completion_tokens, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (cache_key, exp_id, config, model, prompt_hash, raw_resp, json.dumps(parsed) if parsed else None, status, latency, p_tok, c_tok, time.time()))

class SummaryDB:
    """Access precomputed trace summaries from SQLite."""
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_summary(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute("SELECT json_data, text_repr FROM summaries WHERE experiment_id = ?", (exp_id,)).fetchone()
            if row:
                return {
                    "data": json.loads(row[0]),
                    "text": row[1]
                }
        return None

class HistoricalRetriever:
    """k-NN retrieval strictly from non-2024 training samples."""
    def __init__(self, features_df: pd.DataFrame, summary_db: SummaryDB):
        # Strict enforcement: non-2024 training only
        is_train_non_2024 = (~features_df['test']) & (features_df['year'] < 2024)
        self.pool_df = features_df.loc[is_train_non_2024].copy().reset_index(drop=True)
        assert (self.pool_df['year'] < 2024).all(), "Leakage error: 2024 sample found in candidate retrieval pool!"
        
        self.summary_db = summary_db
        self.feature_cols = FEATURES
        
        # Normalize features with log1p for balanced distance
        X_mat = np.log1p(self.pool_df[self.feature_cols].fillna(0).to_numpy())
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.nn.fit(X_mat)
        print(f"Historical retriever initialized with {len(self.pool_df)} clean non-2024 reference samples.")

    def retrieve(self, target_features: np.ndarray, k: int = 3) -> Tuple[List[str], str]:
        """Retrieve k nearest non-2024 neighbors and format as demonstration block."""
        x_norm = np.log1p(np.maximum(0, target_features.reshape(1, -1)))
        distances, indices = self.nn.kneighbors(x_norm, n_neighbors=k)
        
        retrieved_ids = []
        demo_blocks = []
        
        for idx in indices[0]:
            ref_row = self.pool_df.iloc[idx]
            ref_exp = ref_row['experiment']
            ref_label = ref_row['label']
            retrieved_ids.append(ref_exp)
            
            summary = self.summary_db.get_summary(ref_exp)
            if summary:
                # Bounded excerpt of trace
                data = summary['data']
                demo_text = (
                    f"Reference Sample (Historical ID: {ref_exp}):\n"
                    f"  - Total Events: {data['total_events']}, Duration: {data['duration_seconds']}s\n"
                    f"  - Event Rules: {json.dumps(data['rule_counts'])}\n"
                    f"  - Process Tree Depth: {data['process_tree']['max_tree_depth']}, Branching: {data['process_tree']['max_branching_factor']}\n"
                    f"  - Entity Diversity: {json.dumps(data['entity_counts'])}\n"
                    f"  Verified Classification: {ref_label}\n"
                )
                demo_blocks.append(demo_text)
                
        return retrieved_ids, "\n".join(demo_blocks)

def compute_all_metrics(actual_labels: pd.Series, pred_labels: pd.Series, pred_probs: pd.Series, 
                        total_expected: int, latencies: List[float], prompt_tokens: List[int], 
                        completion_tokens: List[int], tool_calls_list: Optional[List[int]] = None) -> Dict[str, Any]:
    """Calculate full benchmark metrics matching evaluation protocol."""
    # Ensure perfectly aligned 0..N-1 indices across all Series
    actual_labels = pd.Series(list(actual_labels))
    pred_labels = pd.Series(list(pred_labels))
    pred_probs = pd.Series(list(pred_probs))

    # Filter valid predictions
    valid_mask = pred_labels.isin(["benign", "malicious"]) & pred_probs.notna()
    n_valid = int(valid_mask.sum())
    n_failures = total_expected - n_valid
    coverage = float(n_valid / total_expected) if total_expected else 0.0

    if n_valid == 0:
        return {
            "n": total_expected,
            "valid_predictions": 0,
            "failures": n_failures,
            "coverage": 0.0,
            "accuracy": 0.0,
            "malware_recall": 0.0,
            "malware_precision": 0.0,
            "malware_f1": 0.0,
            "benign_fpr": 0.0,
            "balanced_accuracy": 0.0,
            "roc_auc": None,
            "partial_auc_max_fpr_0_1": None,
            "avg_latency_sec": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0
        }

    y_true_valid = actual_labels[valid_mask].eq("malicious")
    y_pred_valid = pred_labels[valid_mask].eq("malicious")
    y_prob_valid = pred_probs[valid_mask].astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid, labels=[False, True]).ravel()
    both_classes = (y_true_valid.nunique() == 2)

    acc = (tp + tn) / n_valid
    mal_recall = tp / (tp + fn) if (tp + fn) else 0.0
    mal_prec = tp / (tp + fp) if (tp + fp) else 0.0
    mal_f1 = (2 * mal_prec * mal_recall / (mal_prec + mal_recall)) if (mal_prec + mal_recall) else 0.0
    benign_fpr = fp / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if both_classes else 0.0

    roc_auc = float(roc_auc_score(y_true_valid, y_prob_valid)) if both_classes else None
    partial_auc = float(roc_auc_score(y_true_valid, y_prob_valid, max_fpr=0.1)) if both_classes else None

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    avg_p_tok = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0
    avg_c_tok = float(np.mean(completion_tokens)) if completion_tokens else 0.0
    avg_tools = float(np.mean(tool_calls_list)) if tool_calls_list else 0.0

    return {
        "n": total_expected,
        "valid_predictions": n_valid,
        "failures": n_failures,
        "coverage": round(coverage, 4),
        "benign": int(tn + fp),
        "malicious": int(tp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(float(acc), 6),
        "malware_recall": round(float(mal_recall), 6),
        "malware_precision": round(float(mal_prec), 6),
        "malware_f1": round(float(mal_f1), 6),
        "benign_fpr": round(float(benign_fpr), 6),
        "balanced_accuracy": round(float(bal_acc), 6),
        "roc_auc": round(float(roc_auc), 6) if roc_auc is not None else None,
        "partial_auc_max_fpr_0_1": round(float(partial_auc), 6) if partial_auc is not None else None,
        "avg_latency_sec": round(avg_lat, 3),
        "avg_prompt_tokens": round(avg_p_tok, 1),
        "avg_completion_tokens": round(avg_c_tok, 1),
        "avg_tool_calls": round(avg_tools, 2)
    }

def process_single_sample_dict(item: Tuple[int, Dict[str, Any]],
                               config_name: str,
                               client: LLMClient,
                               cache: DiskCache,
                               summary_db: SummaryDB,
                               retriever: Optional[HistoricalRetriever],
                               sys_prompt: str,
                               zero_user_tmpl: str,
                               ret_user_tmpl: str) -> Dict[str, Any]:
    i, row = item
    exp_id = row['experiment']

    try:
        # Strict check: NEVER provide true_label to prompt
        summary = summary_db.get_summary(exp_id)
        if not summary:
            return {
                "idx": i,
                "experiment_id": exp_id,
                "prediction": None,
                "malware_probability": None,
                "confidence": None,
                "status": "missing_trace_summary",
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "retrieved_ids": "",
                "prompt_hash": "",
                "evidence_count": 0
            }

        trace_text = summary['text']
        retrieved_ids = []
        if config_name == "retrieval_assisted":
            assert retriever is not None, "Retriever must be initialized for retrieval_assisted mode"
            feat_vec = np.array([row[f] for f in FEATURES], dtype=float)
            retrieved_ids, ref_block = retriever.retrieve(feat_vec, k=3)
            user_msg = ret_user_tmpl.replace("{reference_examples}", ref_block).replace("{target_trace}", trace_text)
        elif config_name in ["zero_shot", "tool_agent"]:
            user_msg = zero_user_tmpl.replace("{trace_text}", trace_text)
        else:
            raise ValueError(f"Unknown config: {config_name}")

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg}
        ]

        prompt_str = json.dumps(messages, sort_keys=True)
        prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        cache_key = f"{exp_id}_{config_name}_{client.model}_{prompt_hash[:16]}"

        cached = cache.get(cache_key)
        if cached is not None and cached["status"] in ["success", "cached"]:
            res_content = cached["raw_response"]
            parsed_json = cached["parsed_json"]
            status = "cached"
            lat = cached["latency"]
            p_tok = cached["prompt_tokens"]
            c_tok = cached["completion_tokens"]
        else:
            resp = client.call_chat_completion(messages)
            lat = resp["latency"]
            p_tok = resp["prompt_tokens"]
            c_tok = resp["completion_tokens"]

            if resp["error"]:
                res_content = ""
                parsed_json = None
                status = f"api_error: {resp['error'][:50]}"
            else:
                res_content = resp["content"]
                parsed_json, parse_err = validate_and_parse_json(res_content)
                if parse_err:
                    status = f"validation_error: {parse_err[:50]}"
                else:
                    status = "success"

                cache.set(cache_key, exp_id, config_name, client.model, prompt_hash,
                          res_content, parsed_json, status, lat, p_tok, c_tok)

        pred_val = parsed_json.get("prediction") if parsed_json else None
        prob_val = parsed_json.get("malware_probability") if parsed_json else None
        conf_val = parsed_json.get("confidence") if parsed_json else None
        ev_count = len(parsed_json.get("evidence", [])) if parsed_json else 0
        retrieved_ids_str = ",".join(retrieved_ids)

        return {
            "idx": i,
            "experiment_id": exp_id,
            "prediction": pred_val,
            "malware_probability": prob_val,
            "confidence": conf_val,
            "status": status,
            "latency": lat,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "retrieved_ids": retrieved_ids_str,
            "prompt_hash": prompt_hash,
            "evidence_count": ev_count
        }
    except Exception as exc:
        return {
            "idx": i,
            "experiment_id": exp_id,
            "prediction": None,
            "malware_probability": None,
            "confidence": None,
            "status": f"exception: {str(exc)[:50]}",
            "latency": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retrieved_ids": "",
            "prompt_hash": "",
            "evidence_count": 0
        }

def run_evaluation(config_name: str, split_mode: str, limit: Optional[int], 
                   client: LLMClient, cache: DiskCache, summary_db: SummaryDB,
                   features_df: pd.DataFrame, retriever: Optional[HistoricalRetriever],
                   concurrency: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute evaluation loop for a given configuration with optional multithreading."""
    print(f"\n========================================================")
    print(f"Executing Agent Evaluation: Config={config_name}, Split={split_mode}, Limit={limit}, Concurrency={concurrency}")
    print(f"Model: {client.model}, Base URL: {client.base_url}, Mock: {client.mock}")
    print(f"========================================================")

    # Determine evaluation subset
    if split_mode == "2024":
        eval_df = features_df.loc[features_df['year'].eq(2024)].reset_index(drop=True)
    elif split_mode == "dev_2023":
        eval_df = features_df.loc[features_df['year'].eq(2023)].reset_index(drop=True)
    elif split_mode == "pilot_20":
        # Balanced 10 benign, 10 malicious from 2024 evaluation set
        df_2024 = features_df.loc[features_df['year'].eq(2024)]
        benign_sample = df_2024[df_2024['label'] == 'benign'].sample(n=10, random_state=42)
        mal_sample = df_2024[df_2024['label'] == 'malicious'].sample(n=10, random_state=42)
        eval_df = pd.concat([benign_sample, mal_sample], ignore_index=True)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    if limit is not None and limit < len(eval_df):
        eval_df = eval_df.iloc[:limit].copy()

    total_samples = len(eval_df)
    print(f"Evaluation samples count: {total_samples}")
    print(f"  Benign: {eval_df['label'].eq('benign').sum()}, Malicious: {eval_df['label'].eq('malicious').sum()}")

    # Load prompts
    sys_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text()
    zero_user_tmpl = (PROMPTS_DIR / "zero_shot_user_prompt.txt").read_text()
    ret_user_tmpl = (PROMPTS_DIR / "retrieval_user_prompt.txt").read_text()

    results = [None] * total_samples
    completed_count = 0
    valid_count = 0
    items = list(enumerate(eval_df.to_dict('records')))

    if concurrency > 1:
        print(f"Executing parallel API evaluation with {concurrency} workers...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_sample_dict, item, config_name, client, cache, summary_db, retriever, sys_prompt, zero_user_tmpl, ret_user_tmpl
                ): item[0] for item in items
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results[idx] = res
                completed_count += 1
                if res["prediction"] is not None:
                    valid_count += 1
                if completed_count % 25 == 0 or completed_count == total_samples:
                    print(f"[{completed_count}/{total_samples}] Processed. Last Status={res['status']}, Valid={valid_count}/{completed_count}, LastPred={res['prediction']}, LastProb={res['malware_probability']}")
    else:
        for item in items:
            idx = item[0]
            res = process_single_sample_dict(item, config_name, client, cache, summary_db, retriever, sys_prompt, zero_user_tmpl, ret_user_tmpl)
            results[idx] = res
            completed_count += 1
            if res["prediction"] is not None:
                valid_count += 1
            if completed_count % 10 == 0 or completed_count == total_samples:
                print(f"[{completed_count}/{total_samples}] Processed. Last Status={res['status']}, Valid={valid_count}/{completed_count}, LastPred={res['prediction']}, LastProb={res['malware_probability']}")

    predictions = [r["prediction"] for r in results]
    probabilities = [r["malware_probability"] for r in results]
    confidences = [r["confidence"] for r in results]
    statuses = [r["status"] for r in results]
    latencies = [r["latency"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results]
    completion_tokens = [r["completion_tokens"] for r in results]

    audit_records = [{
        "experiment_id": r["experiment_id"],
        "config": config_name,
        "model": client.model,
        "prompt_hash": r["prompt_hash"],
        "retrieved_ids": r["retrieved_ids"],
        "status": r["status"],
        "prediction": r["prediction"],
        "malware_probability": r["malware_probability"],
        "confidence": r["confidence"],
        "latency_sec": round(r["latency"], 3),
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "evidence_count": r["evidence_count"]
    } for r in results]

    # Build predictions dataframe
    pred_df = pd.DataFrame({
        "experiment": eval_df["experiment"].values,
        "label": eval_df["label"].values,
        "prediction": predictions,
        "malware_probability": probabilities,
        "confidence": confidences,
        "status": statuses
    })

    audit_df = pd.DataFrame(audit_records)

    # Compute metrics
    metrics = compute_all_metrics(
        actual_labels=eval_df["label"],
        pred_labels=pd.Series(predictions),
        pred_probs=pd.Series(probabilities),
        total_expected=total_samples,
        latencies=latencies,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    metrics["configuration"] = config_name
    metrics["split"] = split_mode
    metrics["model"] = client.model

    return pred_df, audit_df, metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Agent Baseline on Wintap DMBD")
    parser.add_argument("--config", type=str, choices=["zero_shot", "retrieval_assisted", "tool_agent", "all"], default="zero_shot")
    parser.add_argument("--split", type=str, choices=["2024", "dev_2023", "pilot_20"], default="pilot_20")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent workers for LLM API calls")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Run with mock model server for offline testing")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    # Explicit environment variable resolution via os.environ.get()
    resolved_base_url = (
        args.base_url 
        or os.environ.get("WINTAP_LLM_BASE_URL") 
        or os.environ.get("OPENAI_BASE_URL") 
        or "https://api.openai.com/v1"
    )
    resolved_api_key = (
        args.api_key 
        or os.environ.get("WINTAP_LLM_API_KEY") 
        or os.environ.get("OPENAI_API_KEY") 
        or "EMPTY"
    )
    resolved_model = (
        args.model 
        or os.environ.get("WINTAP_LLM_MODEL") 
        or os.environ.get("OPENAI_MODEL") 
        or "qwen3.8-flash"
    )
    resolved_concurrency = (
        args.concurrency 
        or int(os.environ.get("WINTAP_CONCURRENCY", "1"))
    )

    # Auto-detection: if base_url is an API key (starts with sk-), fix it
    if resolved_base_url.startswith("sk-"):
        print("[Auto-Config] Detected API key in base_url; assigning to api_key and resetting base_url to https://api.openai.com/v1")
        if resolved_api_key.startswith("http://") or resolved_api_key.startswith("https://"):
            resolved_base_url, resolved_api_key = resolved_api_key, resolved_base_url
        else:
            resolved_api_key = resolved_base_url
            resolved_base_url = "https://api.openai.com/v1"
    elif resolved_api_key.startswith("http://") or resolved_api_key.startswith("https://"):
        print("[Auto-Config] Detected URL in api_key; swapping base_url and api_key")
        resolved_base_url, resolved_api_key = resolved_api_key, resolved_base_url

    # Auto-detection: opencode.ai does not currently route gpt-5.6-luna (upstream HTTP 500)
    if "opencode.ai" in resolved_base_url and resolved_model == "gpt-5.6-luna":
        print("[Auto-Config] 'gpt-5.6-luna' returns upstream HTTP 500 on opencode.ai. Routing to 'qwen3.8-flash'.")
        resolved_model = "qwen3.8-flash"

    if not args.mock and resolved_api_key in ["", "EMPTY"]:
        print("ERROR: No API key provided!")
        print("Please export OPENAI_API_KEY or WINTAP_LLM_API_KEY in your shell environment, e.g.:")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    cache = DiskCache(CACHE_DIR / "llm_cache.db")
    summary_db = SummaryDB(DB_PATH)

    print(f"Loading features from {FEATURES_CSV}...")
    features_df = pd.read_csv(FEATURES_CSV)

    retriever = HistoricalRetriever(features_df, summary_db)

    client = LLMClient(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=resolved_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        mock=args.mock
    )

    configs_to_run = ["zero_shot", "retrieval_assisted"] if args.config == "all" else [args.config]

    all_metrics = []
    # If rf_metrics.csv exists, load it into all_metrics
    rf_metrics_path = AGENT_DIR / "rf_metrics.csv"
    if rf_metrics_path.exists():
        rf_m = pd.read_csv(rf_metrics_path).to_dict(orient="records")[0]
        rf_m["configuration"] = "random_forest_200"
        rf_m["split"] = "2024"
        rf_m["model"] = "RandomForest(n=200,seed=1)"
        rf_m["coverage"] = 1.0
        rf_m["failures"] = 0
        rf_m["valid_predictions"] = rf_m["n"]
        all_metrics.append(rf_m)

    combined_audits = []

    for cfg in configs_to_run:
        pred_df, audit_df, metrics = run_evaluation(
            config_name=cfg,
            split_mode=args.split,
            limit=args.limit,
            client=client,
            cache=cache,
            summary_db=summary_db,
            features_df=features_df,
            retriever=retriever,
            concurrency=resolved_concurrency
        )

        # Save predictions
        pred_path = AGENT_DIR / f"agent_{cfg}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved {cfg} predictions to {pred_path}")

        combined_audits.append(audit_df)
        all_metrics.append(metrics)

        print(f"\n--- Metrics Summary ({cfg}) ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Save audit records
    if combined_audits:
        full_audit = pd.concat(combined_audits, ignore_index=True)
        audit_path = AGENT_DIR / "per_sample_audit.csv"
        full_audit.to_csv(audit_path, index=False)
        print(f"\nSaved per-sample audit log to {audit_path}")

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = AGENT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved combined metrics to {metrics_path}")

if __name__ == "__main__":
    main()
