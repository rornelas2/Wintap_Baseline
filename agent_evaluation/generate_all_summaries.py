#!/usr/bin/env python3
"""
Precompute trace summaries for all 8 tree shards (64,747 experiments).
Extracts bounded, label-free, year-free behavioral summaries.
Stores results in:
  - SQLite database: agent_evaluation/trace_summaries.db
  - Shard JSONL files: agent_evaluation/trace_summaries/shard_{i}.jsonl
  - Sample JSON files in: agent_evaluation/trace_summaries/samples/{experiment}.json
"""

import sys
import time
import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

DATA_ROOT = Path("/home/rornelas5/data/wintap/dmbd")
ROOT = Path("/home/rornelas5/Wintap_Baseline")
OUT_DIR = ROOT / "agent_evaluation" / "trace_summaries"
DB_PATH = ROOT / "agent_evaluation" / "trace_summaries.db"

def format_summary_text(summary: dict) -> str:
    lines = [
        f"=== Dynamic Execution Trace Summary (Experiment: {summary['experiment_id']}) ===",
        f"Execution Metrics: {summary['total_events']} total events | Duration: {summary['duration_seconds']}s | "
        f"Avg Rate: {summary['avg_events_per_sec']} ev/s | Peak Rate: {summary['peak_events_per_sec']} ev/s",
        "",
        "--- Event Counts by Rule ---",
    ]
    for rule, cnt in sorted(summary['rule_counts'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {rule}: {cnt}")
        
    lines.append("\n--- Entity Diversity Counts ---")
    for entity, cnt in summary['entity_counts'].items():
        lines.append(f"  - {entity}: {cnt}")
        
    lines.append("\n--- Process Tree Structure ---")
    pt = summary['process_tree']
    lines.append(f"  - Total Processes Involved: {pt['total_processes']}")
    lines.append(f"  - Max Process Tree Depth: {pt['max_tree_depth']}")
    lines.append(f"  - Max Branching Factor: {pt['max_branching_factor']}")
    lines.append(f"  - Root Process Direct Children: {pt['root_spawned_children']}")
    
    lines.append("\n--- Top Active Process Profiles ---")
    for p in summary['top_processes']:
        root_str = " (Root Detonation Binary)" if p['is_root'] else ""
        lines.append(f"  * {p['process_id']}{root_str}: {p['total_actions']} actions, spawned {p['children_spawned']} child processes")
        action_str = ", ".join(f"{k}: {v}" for k, v in p['action_breakdown'].items())
        lines.append(f"    Actions: {action_str}")
        
    lines.append("\n--- Chronological Event Flow (Run-Length Encoded) ---")
    flow_str = " -> ".join(summary['event_sequence_compressed'])
    lines.append(f"  {flow_str}")
    
    return "\n".join(lines)

def summarize_experiment(exp_id: str, events: list) -> dict:
    events.sort(key=lambda x: str(x[0]))
    n_events = len(events)
    
    rules_list = [e[1] for e in events]
    rule_counts = dict(Counter(rules_list))
    
    # Duration (strictly relative seconds, no calendar year)
    ts_start = str(events[0][0])
    ts_end = str(events[-1][0])
    try:
        t_s = pd.Timestamp(ts_start)
        t_e = pd.Timestamp(ts_end)
        duration_sec = max(0.0, (t_e - t_s).total_seconds())
    except Exception:
        duration_sec = 0.0
        
    # Burst rates
    if duration_sec > 0:
        avg_rate = round(n_events / max(1.0, duration_sec), 2)
        # 1-second bin peak approximation
        sec_counts = Counter()
        for e in events:
            try:
                sec_offset = int((pd.Timestamp(str(e[0])) - t_s).total_seconds())
                sec_counts[sec_offset] += 1
            except Exception:
                pass
        peak_rate = max(sec_counts.values()) if sec_counts else n_events
    else:
        avg_rate = float(n_events)
        peak_rate = n_events
        
    proc_guids = set()
    unique_files = set()
    unique_images = set()
    unique_reg = set()
    unique_net = set()
    children_by_parent = defaultdict(list)
    root_proc = None
    
    for ts, rule, p_g, c_g, c_n in events:
        if pd.notna(p_g):
            proc_guids.add(p_g)
        if rule == 'detonation':
            root_proc = c_g
        elif rule == 'process_create':
            if pd.notna(c_g):
                proc_guids.add(c_g)
            if pd.notna(p_g) and pd.notna(c_g):
                children_by_parent[p_g].append(c_g)
        elif rule == 'file_create':
            unique_files.add(c_n)
        elif rule == 'image_load':
            unique_images.add(c_n)
        elif rule.startswith('reg_'):
            unique_reg.add(c_n)
        elif rule == 'net_connect':
            unique_net.add(c_n)
            
    if not root_proc and events:
        root_proc = events[0][2]
        
    # BFS depth from root
    depth = {root_proc: 1}
    queue = [root_proc]
    max_depth = 1
    visited = {root_proc}
    while queue:
        curr = queue.pop(0)
        curr_d = depth.get(curr, 1)
        if curr_d > max_depth:
            max_depth = curr_d
        for kid in children_by_parent.get(curr, []):
            if kid not in visited:
                visited.add(kid)
                depth[kid] = curr_d + 1
                queue.append(kid)
                
    branching_factors = [len(kids) for kids in children_by_parent.values()]
    max_branching = max(branching_factors) if branching_factors else 0
    
    # Run-length compressed sequence
    compressed_runs = []
    curr_r = rules_list[0]
    run_len = 1
    for r in rules_list[1:]:
        if r == curr_r:
            run_len += 1
        else:
            compressed_runs.append(f"{curr_r} x{run_len}" if run_len > 1 else curr_r)
            curr_r = r
            run_len = 1
    compressed_runs.append(f"{curr_r} x{run_len}" if run_len > 1 else curr_r)
    
    if len(compressed_runs) > 40:
        display_runs = compressed_runs[:40] + [f"... [{len(compressed_runs)-40} further event runs omitted]"]
    else:
        display_runs = compressed_runs
        
    p_act = Counter([e[2] for e in events if pd.notna(e[2])])
    top_procs = p_act.most_common(5)
    
    proc_profiles = []
    for rank, (p_g, act_count) in enumerate(top_procs, 1):
        is_root = (p_g == root_proc)
        tag = "Process_Root" if is_root else f"Process_{rank}"
        p_rules = Counter([e[1] for e in events if e[2] == p_g])
        proc_profiles.append({
            "process_id": tag,
            "is_root": is_root,
            "total_actions": act_count,
            "children_spawned": len(children_by_parent.get(p_g, [])),
            "action_breakdown": dict(p_rules)
        })
        
    summary = {
        "experiment_id": exp_id,
        "total_events": n_events,
        "duration_seconds": round(duration_sec, 3),
        "avg_events_per_sec": avg_rate,
        "peak_events_per_sec": peak_rate,
        "rule_counts": rule_counts,
        "entity_counts": {
            "processes": len(proc_guids),
            "unique_files_created": len(unique_files),
            "unique_images_loaded": len(unique_images),
            "unique_registry_keys": len(unique_reg),
            "unique_network_connections": len(unique_net),
        },
        "process_tree": {
            "total_processes": len(proc_guids),
            "max_tree_depth": max_depth,
            "max_branching_factor": max_branching,
            "root_spawned_children": len(children_by_parent.get(root_proc, []))
        },
        "event_sequence_compressed": display_runs,
        "top_processes": proc_profiles
    }
    return summary

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing SQLite database at {DB_PATH}...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            experiment_id TEXT PRIMARY KEY,
            shard_id INTEGER,
            total_events INTEGER,
            duration_seconds REAL,
            json_data TEXT,
            text_repr TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp ON summaries(experiment_id)")
    conn.commit()
    
    total_processed = 0
    t_global_start = time.time()
    
    for shard_id in range(8):
        tree_file = DATA_ROOT / f"trees_{shard_id}.json"
        print(f"\n--- Processing Shard {shard_id} ({tree_file.name}) ---", flush=True)
        t_shard = time.time()
        df = pd.read_json(tree_file)
        print(f"Loaded {len(df)} events in {time.time() - t_shard:.2f}s.", flush=True)
        
        t_partition = time.time()
        exp_events = defaultdict(list)
        exps = df['experiment'].values
        tss = df['timestamp'].values
        rules = df['RuleName'].values
        p_guids = df['parent_guid'].values
        c_guids = df['child_guid'].values
        c_names = df['child_name'].values
        
        for i in range(len(df)):
            exp_events[exps[i]].append((tss[i], rules[i], p_guids[i], c_guids[i], c_names[i]))
            
        print(f"Partitioned {len(df)} rows into {len(exp_events)} experiments in {time.time() - t_partition:.2f}s.", flush=True)
        
        jsonl_path = OUT_DIR / f"shard_{shard_id}.jsonl"
        db_rows = []
        
        with open(jsonl_path, "w") as jf:
            for exp_id, events in exp_events.items():
                summ = summarize_experiment(exp_id, events)
                text_repr = format_summary_text(summ)
                json_str = json.dumps(summ)
                
                # Write to jsonl
                jf.write(json.dumps({"experiment_id": exp_id, "summary": summ, "text": text_repr}) + "\n")
                
                db_rows.append((
                    exp_id,
                    shard_id,
                    summ["total_events"],
                    summ["duration_seconds"],
                    json_str,
                    text_repr
                ))
                
                # Save a few samples to disk as individual files for easy inspection
                if total_processed < 50 or total_processed % 500 == 0:
                    sample_path = samples_dir / f"{exp_id}.json"
                    with open(sample_path, "w") as sf:
                        json.dump(summ, sf, indent=2)
                        
                total_processed += 1
                
        # Bulk insert to SQLite
        cursor.executemany("""
            INSERT OR REPLACE INTO summaries 
            (experiment_id, shard_id, total_events, duration_seconds, json_data, text_repr)
            VALUES (?, ?, ?, ?, ?, ?)
        """, db_rows)
        conn.commit()
        print(f"Saved {len(db_rows)} experiments to DB and {jsonl_path.name} in {time.time() - t_shard:.2f}s.", flush=True)
        
    conn.close()
    print(f"\n==================================================")
    print(f"Successfully processed and indexed ALL {total_processed} experiments in {time.time() - t_global_start:.2f}s!")
    print(f"Database: {DB_PATH}")
    print(f"Directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
