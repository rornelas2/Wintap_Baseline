#!/usr/bin/env python3
"""
Inspect raw tree structure and experiment distribution across trees_0..trees_7.json.
Runs on compute node via Slurm.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_ROOT = Path("/home/rornelas5/data/wintap/dmbd")
TREE_FILES = [DATA_ROOT / f"trees_{i}.json" for i in range(8)]

def main():
    print(f"Reading trees_0.json sample...")
    df0 = pd.read_json(TREE_FILES[0])
    print(f"trees_0.json shape: {df0.shape}")
    print(f"trees_0.json columns: {df0.columns.tolist()}")
    print("\nSample 5 rows:")
    print(df0.head(5).to_string())
    
    print("\nUnique RuleNames in trees_0:")
    print(df0['RuleName'].value_counts())
    
    # Inspect a couple of experiments with various RuleNames
    sample_exps = df0['experiment'].unique()[:5]
    for exp in sample_exps:
        sub = df0[df0['experiment'] == exp].sort_values('timestamp')
        print(f"\n--- Experiment {exp} ({len(sub)} events) ---")
        print(f"Rule counts: {dict(Counter(sub['RuleName']))}")
        print(f"Timestamp range: min={sub['timestamp'].min()}, max={sub['timestamp'].max()}")
        print(f"Unique parent_name: {sub['parent_name'].nunique()}, unique child_name: {sub['child_name'].nunique()}")
        print(f"Unique parent_guid: {sub['parent_guid'].nunique()}, unique child_guid: {sub['child_guid'].nunique()}")
        print("First 5 events:")
        print(sub[['timestamp', 'RuleName', 'parent_name', 'child_name', 'parent_guid', 'child_guid']].head(5).to_string())

if __name__ == "__main__":
    main()
