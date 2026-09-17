#!/usr/bin/env python3
import time
import json
from collections import Counter, defaultdict
import pandas as pd

def test():
    t0 = time.time()
    print("Reading trees_0.json...", flush=True)
    df = pd.read_json("/home/rornelas5/data/wintap/dmbd/trees_0.json")
    print(f"Read in {time.time() - t0:.2f}s, rows={len(df)}", flush=True)
    
    t1 = time.time()
    print("Partitioning events by experiment using zip...", flush=True)
    exp_events = defaultdict(list)
    
    # Extract arrays
    exps = df['experiment'].values
    tss = df['timestamp'].values
    rules = df['RuleName'].values
    p_guids = df['parent_guid'].values
    c_guids = df['child_guid'].values
    c_names = df['child_name'].values
    
    for i in range(len(df)):
        exp_events[exps[i]].append((tss[i], rules[i], p_guids[i], c_guids[i], c_names[i]))
        
    print(f"Partitioned {len(df)} events into {len(exp_events)} experiments in {time.time() - t1:.2f}s!", flush=True)
    
    t2 = time.time()
    print("Summarizing all experiments in trees_0...", flush=True)
    summaries = []
    
    for exp_id, events in exp_events.items():
        # Sort events by timestamp
        events.sort(key=lambda x: str(x[0]))
        n_events = len(events)
        
        # Rule counts
        rules_list = [e[1] for e in events]
        rule_counts = dict(Counter(rules_list))
        
        # Timestamps (relative duration, no year)
        ts_start = str(events[0][0])
        ts_end = str(events[-1][0])
        # Simple duration approximation if strings
        try:
            # Format: 'YYYY-MM-DD HH:MM:SS.mmm'
            # Parse time of day or timedelta
            t_s = pd.Timestamp(ts_start)
            t_e = pd.Timestamp(ts_end)
            duration_sec = max(0.0, (t_e - t_s).total_seconds())
        except Exception:
            duration_sec = 0.0
            
        # Entities
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
            
        # Linear BFS for tree depth
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
        
        # Compressed runs
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
            
        # Top processes by activity
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
        summaries.append(summary)
        
    print(f"Processed ALL {len(summaries)} experiments in {time.time() - t2:.2f}s! ({len(summaries) / (time.time() - t2):.1f} exps/sec)", flush=True)
    print(f"Total time: {time.time() - t0:.2f}s", flush=True)

if __name__ == "__main__":
    test()
