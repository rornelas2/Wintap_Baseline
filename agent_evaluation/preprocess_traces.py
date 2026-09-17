#!/usr/bin/env python3
"""
Precompute deterministic, label-free, year-free trace summaries from raw trees_*.json.
Can run for a single shard (trees_i.json) or all shards in parallel.

Guarantees:
- NO ground-truth labels are included.
- NO calendar year or absolute timestamps are included (relative offsets only).
- Bounded context: compressed event sequences, process tree metrics, entity counts, top processes.
- Saved to agent_evaluation/trace_summaries/{experiment}.json
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

DATA_ROOT = Path("/home/rornelas5/data/wintap/dmbd")
ROOT = Path("/home/rornelas5/Wintap_Baseline")
OUT_DIR = ROOT / "agent_evaluation" / "trace_summaries"

def build_experiment_summary(exp_id: str, events: pd.DataFrame) -> dict:
    events = events.sort_values('timestamp').reset_index(drop=True)
    n_events = len(events)
    
    # 1. Event counts by RuleName
    rule_counts = dict(Counter(events['RuleName']))
    
    # 2. Temporal metrics (elapsed time from start in seconds, strictly NO calendar year)
    ts_series = pd.to_datetime(events['timestamp'])
    t_start = ts_series.iloc[0]
    t_end = ts_series.iloc[-1]
    duration_sec = max(0.0, (t_end - t_start).total_seconds())
    
    if duration_sec > 0:
        elapsed_sec = (ts_series - t_start).dt.total_seconds()
        sec_bins = elapsed_sec.astype(int)
        events_per_sec = sec_bins.value_counts()
        peak_events_per_sec = int(events_per_sec.max())
        avg_events_per_sec = round(n_events / max(1.0, duration_sec), 2)
    else:
        peak_events_per_sec = n_events
        avg_events_per_sec = float(n_events)
        
    # 3. Entity counts
    proc_guids = set(events['parent_guid'].dropna())
    created_procs = set(events.loc[events['RuleName'] == 'process_create', 'child_guid'].dropna())
    proc_guids.update(created_procs)
    
    unique_files = int(events.loc[events['RuleName'] == 'file_create', 'child_name'].nunique())
    unique_images = int(events.loc[events['RuleName'] == 'image_load', 'child_name'].nunique())
    unique_reg_keys = int(events.loc[events['RuleName'].isin(['reg_create_key', 'reg_delete_key', 'reg_write', 'reg_delete_value']), 'child_name'].nunique())
    unique_net_conns = int(events.loc[events['RuleName'] == 'net_connect', 'child_name'].nunique())
    
    # 4. Process Tree Analysis
    det_event = events[events['RuleName'] == 'detonation']
    if not det_event.empty:
        det_parent = det_event.iloc[0]['parent_guid']
        root_proc = det_event.iloc[0]['child_guid']
    else:
        root_proc = events.iloc[0]['parent_guid']
        det_parent = None
        
    proc_creates = events[events['RuleName'] == 'process_create']
    children_by_parent = defaultdict(list)
    for _, row in proc_creates.iterrows():
        p = row['parent_guid']
        c = row['child_guid']
        children_by_parent[p].append(c)
        
    def get_max_depth(node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return 0
        visited.add(node)
        kids = children_by_parent.get(node, [])
        if not kids:
            return 1
        return 1 + max(get_max_depth(k, visited.copy()) for k in kids)
        
    tree_depth = get_max_depth(root_proc)
    branching_factors = [len(kids) for kids in children_by_parent.values()]
    max_branching = max(branching_factors) if branching_factors else 0
    total_processes = len(proc_guids)
    
    # 5. Ordered Event-Type Sequence (Run-Length Encoded)
    rule_seq = events['RuleName'].tolist()
    compressed_runs = []
    current_rule = rule_seq[0]
    current_run = 1
    for r in rule_seq[1:]:
        if r == current_rule:
            current_run += 1
        else:
            compressed_runs.append(f"{current_rule} x{current_run}" if current_run > 1 else current_rule)
            current_rule = r
            current_run = 1
    compressed_runs.append(f"{current_rule} x{current_run}" if current_run > 1 else current_rule)
    
    # Keep up to first 40 runs to bound token count
    if len(compressed_runs) > 40:
        display_runs = compressed_runs[:40] + [f"... [{len(compressed_runs)-40} further event runs omitted]"]
    else:
        display_runs = compressed_runs
        
    # 6. Per-Process Activity Summaries (top 5 most active processes)
    proc_activity = events.groupby('parent_guid')['RuleName'].agg(list).to_dict()
    top_procs = sorted(proc_activity.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    process_summaries = []
    
    guid_to_tag = {}
    if root_proc:
        guid_to_tag[root_proc] = "Process_Root"
    tag_counter = 1
    for p_guid, _ in top_procs:
        if p_guid not in guid_to_tag:
            guid_to_tag[p_guid] = f"Process_{tag_counter}"
            tag_counter += 1
            
    for p_guid, act_rules in top_procs:
        counts = dict(Counter(act_rules))
        tag = guid_to_tag.get(p_guid, f"Process_{tag_counter}")
        is_root = (p_guid == root_proc)
        children_spawned = len(children_by_parent.get(p_guid, []))
        process_summaries.append({
            "process_id": tag,
            "is_root": is_root,
            "total_actions": len(act_rules),
            "children_spawned": children_spawned,
            "action_breakdown": counts
        })
        
    summary = {
        "experiment_id": exp_id,
        "total_events": n_events,
        "duration_seconds": round(duration_sec, 3),
        "peak_events_per_sec": peak_events_per_sec,
        "avg_events_per_sec": avg_events_per_sec,
        "rule_counts": rule_counts,
        "entity_counts": {
            "processes": total_processes,
            "unique_files_created": unique_files,
            "unique_images_loaded": unique_images,
            "unique_registry_keys": unique_reg_keys,
            "unique_network_connections": unique_net_conns,
        },
        "process_tree": {
            "total_processes": total_processes,
            "max_tree_depth": tree_depth,
            "max_branching_factor": max_branching,
            "root_spawned_children": len(children_by_parent.get(root_proc, []))
        },
        "event_sequence_compressed": display_runs,
        "top_processes": process_summaries
    }
    return summary

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

def process_shard(shard_id: int):
    tree_file = DATA_ROOT / f"trees_{shard_id}.json"
    print(f"[Shard {shard_id}] Loading {tree_file} ...", flush=True)
    df = pd.read_json(tree_file)
    print(f"[Shard {shard_id}] Loaded {len(df)} events across {df['experiment'].nunique()} experiments.", flush=True)
    
    shard_out_dir = OUT_DIR / f"shard_{shard_id}"
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a combined jsonlines file for this shard as well as individual files
    jsonl_path = OUT_DIR / f"summaries_shard_{shard_id}.jsonl"
    
    count = 0
    with open(jsonl_path, "w") as jf:
        for exp_id, grp in df.groupby('experiment', sort=False):
            summary = build_experiment_summary(exp_id, grp)
            summary['text_representation'] = format_summary_text(summary)
            
            # Write to combined jsonl
            jf.write(json.dumps(summary) + "\n")
            
            # Write individual file
            exp_file = shard_out_dir / f"{exp_id}.json"
            with open(exp_file, "w") as ef:
                json.dump(summary, ef, indent=2)
                
            count += 1
            if count % 1000 == 0:
                print(f"[Shard {shard_id}] Processed {count} experiments...", flush=True)
                
    print(f"[Shard {shard_id}] Complete! Processed {count} experiments. Saved to {jsonl_path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=None, help="Shard id 0..7")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.shard is not None:
        process_shard(args.shard)
    else:
        for i in range(8):
            process_shard(i)

if __name__ == "__main__":
    main()
