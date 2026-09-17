#!/usr/bin/env python3
import time
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

def test():
    t0 = time.time()
    print("Reading trees_0.json...")
    df = pd.read_json("/home/rornelas5/data/wintap/dmbd/trees_0.json")
    print(f"Read in {time.time() - t0:.2f}s, rows={len(df)}")
    
    t1 = time.time()
    # Sort entire dataframe once
    df['ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['experiment', 'ts']).reset_index(drop=True)
    print(f"Sorted in {time.time() - t1:.2f}s")
    
    t2 = time.time()
    exps = df['experiment'].unique()[:500]
    print(f"Testing summary on 500 experiments...")
    
    processed = 0
    for exp in exps:
        sub = df[df['experiment'] == exp]
        n_events = len(sub)
        rule_counts = dict(Counter(sub['RuleName']))
        
        # Temporal
        t_start = sub['ts'].iloc[0]
        t_end = sub['ts'].iloc[-1]
        duration_sec = max(0.0, (t_end - t_start).total_seconds())
        
        # Entities
        proc_creates = sub[sub['RuleName'] == 'process_create']
        children_by_parent = defaultdict(list)
        for _, row in proc_creates.iterrows():
            children_by_parent[row['parent_guid']].append(row['child_guid'])
            
        det_event = sub[sub['RuleName'] == 'detonation']
        root_proc = det_event.iloc[0]['child_guid'] if not det_event.empty else sub.iloc[0]['parent_guid']
        
        # Linear BFS for depth
        depth = {root_proc: 1}
        queue = [root_proc]
        max_depth = 1
        visited = {root_proc}
        while queue:
            curr = queue.pop(0)
            curr_d = depth[curr]
            if curr_d > max_depth:
                max_depth = curr_d
            for child in children_by_parent.get(curr, []):
                if child not in visited:
                    visited.add(child)
                    depth[child] = curr_d + 1
                    queue.append(child)
                    
        processed += 1
        
    print(f"Processed 500 experiments in {time.time() - t2:.2f}s! ({processed / (time.time() - t2):.1f} exps/sec)")

if __name__ == "__main__":
    test()
