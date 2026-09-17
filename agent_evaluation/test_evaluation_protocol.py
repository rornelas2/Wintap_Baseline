#!/usr/bin/env python3
"""
Self-check and verification test suite for Wintap DMBD Agent Evaluation Protocol.

Verifies:
1. No 2024 sample enters retrieval / demonstrations / calibration.
2. All 2024 samples are evaluated exactly once with zero duplicates.
3. Random forest results exactly reproduce the clean temporal reference.
4. Malformed LLM responses fail explicitly and cannot silently count as benign.
"""

import sys
import json
import sqlite3
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/rornelas5/Wintap_Baseline")
AGENT_DIR = ROOT / "agent_evaluation"
FEATURES_CSV = ROOT / "year_evaluation" / "features.csv"
DB_PATH = AGENT_DIR / "trace_summaries.db"
RF_PREDS_CSV = AGENT_DIR / "rf_clean_2024_predictions.csv"
RF_METRICS_CSV = AGENT_DIR / "rf_metrics.csv"

# Import evaluation components
sys.path.insert(0, str(AGENT_DIR))
from run_agent_evaluation import validate_and_parse_json, compute_all_metrics, HistoricalRetriever, SummaryDB

class TestEvaluationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_df = pd.read_csv(FEATURES_CSV)
        cls.is_2024 = cls.features_df['year'].eq(2024)
        cls.eval_2024 = cls.features_df.loc[cls.is_2024]
        cls.train_non_2024 = cls.features_df.loc[(~cls.features_df['test']) & (~cls.is_2024)]

    def test_rf_reproduction(self):
        """Verify RF clean temporal reference metrics exact reproduction."""
        self.assertTrue(RF_PREDS_CSV.exists(), "rf_clean_2024_predictions.csv missing")
        self.assertTrue(RF_METRICS_CSV.exists(), "rf_metrics.csv missing")

        preds_df = pd.read_csv(RF_PREDS_CSV)
        metrics_df = pd.read_csv(RF_METRICS_CSV)
        m = metrics_df.iloc[0].to_dict()

        # Check sample counts
        self.assertEqual(len(preds_df), 14619, "Must evaluate exactly 14,619 samples")
        self.assertTrue(preds_df['experiment'].is_unique, "Predictions contain duplicate experiments")

        # Check classes
        self.assertEqual(m['n'], 14619)
        self.assertEqual(m['benign'], 12252)
        self.assertEqual(m['malicious'], 2367)

        # Check confusion matrix
        self.assertEqual(m['tn'], 11372)
        self.assertEqual(m['fp'], 880)
        self.assertEqual(m['fn'], 311)
        self.assertEqual(m['tp'], 2056)

        # Check accuracy and recall
        self.assertAlmostEqual(m['accuracy'], 0.918530679, places=5)
        self.assertAlmostEqual(m['malware_recall'], 0.868610055, places=5)
        self.assertAlmostEqual(m['malware_precision'], 0.700272480, places=5)
        self.assertAlmostEqual(m['benign_fpr'], 0.071825008, places=5)
        print("\n[PASS] RF temporal reference exact match confirmed.")

    def test_no_2024_leakage_in_retrieval(self):
        """Verify retrieval candidate pool strictly excludes all 2024 samples."""
        summary_db = SummaryDB(DB_PATH)
        retriever = HistoricalRetriever(self.features_df, summary_db)

        # 1. Pool must not contain any 2024 samples
        self.assertTrue((retriever.pool_df['year'] < 2024).all(), "2024 sample found in retriever pool!")
        self.assertEqual(len(retriever.pool_df), 32996, "Pool size must equal non-2024 training samples (32,996)")

        # 2. Test actual retrieval on 10 random 2024 targets
        sample_targets = self.eval_2024.sample(n=10, random_state=7)
        for _, target_row in sample_targets.iterrows():
            feat_vec = target_row[['process_create', 'process_term', 'reg_create_key',
                                   'reg_write', 'file_create', 'image_load',
                                   'reg_delete_key', 'net_connect', 'reg_delete_value']].to_numpy(dtype=float)
            ret_ids, block = retriever.retrieve(feat_vec, k=5)
            self.assertEqual(len(ret_ids), 5)
            # Verify every retrieved ID has year < 2024
            for r_id in ret_ids:
                r_year = self.features_df.loc[self.features_df['experiment'] == r_id, 'year'].values[0]
                self.assertLess(r_year, 2024, f"Leakage! Retrieved sample {r_id} has year {r_year} >= 2024")
                self.assertNotIn(r_id, self.eval_2024['experiment'].values, f"Retrieved ID {r_id} is in 2024 eval set!")

        print("\n[PASS] Strict absence of 2024 samples in retrieval pool confirmed.")

    def test_malformed_llm_responses_cannot_count_as_benign(self):
        """Verify that malformed/invalid JSON responses fail explicitly and never count as benign."""
        test_cases = [
            ("", "Empty string"),
            ("Not JSON at all", "Plain text"),
            ('{"prediction": "benign"', "Unterminated JSON"),
            ('{"prediction": "unknown", "malware_probability": 0.5, "confidence": 0.5, "evidence": []}', "Invalid prediction value"),
            ('{"malware_probability": 0.5, "confidence": 0.5, "evidence": []}', "Missing prediction"),
            ('{"prediction": "benign", "confidence": 0.5, "evidence": []}', "Missing malware_probability"),
            ('{"prediction": "benign", "malware_probability": 1.5, "confidence": 0.5, "evidence": []}', "Probability out of range > 1.0"),
            ('{"prediction": "benign", "malware_probability": -0.2, "confidence": 0.5, "evidence": []}', "Probability out of range < 0.0"),
            ('{"prediction": "benign", "malware_probability": "low", "confidence": 0.5, "evidence": []}', "Non-numeric probability"),
            ('{"prediction": "benign", "malware_probability": 0.2, "confidence": 0.8, "evidence": "not a list"}', "Evidence not a list"),
            ('{"prediction": "malicious", "malware_probability": 0.9, "confidence": 2.0, "evidence": []}', "Confidence > 1.0"),
        ]

        for raw, desc in test_cases:
            parsed, err = validate_and_parse_json(raw)
            self.assertIsNone(parsed, f"Failed for case '{desc}': parsed should be None but was {parsed}")
            self.assertIsNotNone(err, f"Failed for case '{desc}': err should not be None")

        # Verify valid JSON parses properly
        valid_json = '{"prediction": "malicious", "malware_probability": 0.85, "confidence": 0.9, "evidence": ["high process creation", "network burst"]}'
        parsed, err = validate_and_parse_json(valid_json)
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["prediction"], "malicious")
        self.assertAlmostEqual(parsed["malware_probability"], 0.85)
        self.assertEqual(len(parsed["evidence"]), 2)

        # Test metric calculation behavior with failures
        y_true = pd.Series(["benign", "malicious", "benign", "malicious"])
        y_pred = pd.Series(["benign", None, "benign", None]) # 2 failures
        y_prob = pd.Series([0.1, None, 0.2, None])

        metrics = compute_all_metrics(
            actual_labels=y_true,
            pred_labels=y_pred,
            pred_probs=y_prob,
            total_expected=4,
            latencies=[0.1, 0.1, 0.1, 0.1],
            prompt_tokens=[10, 10, 10, 10],
            completion_tokens=[10, 10, 10, 10]
        )

        self.assertEqual(metrics["failures"], 2, "Failures must be counted explicitly")
        self.assertEqual(metrics["valid_predictions"], 2)
        self.assertEqual(metrics["coverage"], 0.5, "Coverage must reflect valid prediction ratio")
        # Ensure failures did not get treated as benign (which would have made tp=0, fp=0, fn=2, tn=2)
        # Instead, only valid samples (2 benign) are evaluated
        self.assertEqual(metrics["benign"], 2)
        self.assertEqual(metrics["malicious"], 0)
        print("\n[PASS] Malformed LLM response rejection and explicit failure metrics confirmed.")

    def test_2024_evaluation_set_completeness(self):
        """Verify that exactly 14,619 evaluation samples exist with correct class balance."""
        self.assertEqual(len(self.eval_2024), 14619)
        self.assertEqual(self.eval_2024['label'].eq('benign').sum(), 12252)
        self.assertEqual(self.eval_2024['label'].eq('malicious').sum(), 2367)
        self.assertTrue(self.eval_2024['experiment'].is_unique)
        print("\n[PASS] 2024 evaluation set completeness confirmed.")

    def test_audit_log_structure_and_no_leakage(self):
        """Verify per_sample_audit.csv contains required fields and no ground truth leakage."""
        audit_path = AGENT_DIR / "per_sample_audit.csv"
        self.assertTrue(audit_path.exists(), "per_sample_audit.csv must exist")

        audit_df = pd.read_csv(audit_path)
        required_cols = [
            "experiment_id", "config", "model", "prompt_hash",
            "retrieved_ids", "status", "prediction", "malware_probability",
            "confidence", "latency_sec", "prompt_tokens", "completion_tokens"
        ]
        for col in required_cols:
            self.assertIn(col, audit_df.columns, f"Missing column {col} in per_sample_audit.csv")

        # Verify no ground truth label in audit fields consumed by model
        self.assertNotIn("label", audit_df.columns)
        self.assertNotIn("ground_truth", audit_df.columns)
        self.assertNotIn("true_label", audit_df.columns)

        # Verify prompt hashes are 64-character SHA256 hex strings
        valid_hashes = audit_df['prompt_hash'].str.match(r"^[0-9a-f]{64}$")
        self.assertTrue(valid_hashes.all(), "Prompt hashes must be 64-character hex strings")

        # Verify all retrieved IDs are strictly pre-2024
        for _, row in audit_df.iterrows():
            r_ids = str(row['retrieved_ids'])
            if r_ids and r_ids != "nan" and r_ids.strip():
                for exp_id in r_ids.split(","):
                    exp_id = exp_id.strip()
                    if exp_id:
                        m = self.features_df.loc[self.features_df['experiment'] == exp_id, 'year']
                        self.assertTrue(len(m) > 0, f"Retrieved ID {exp_id} not found in features.csv")
                        self.assertLess(m.values[0], 2024, f"Leakage: Retrieved sample {exp_id} has year >= 2024")

        print("\n[PASS] Audit log structure and strict no-leakage verification confirmed.")

if __name__ == "__main__":
    unittest.main()
