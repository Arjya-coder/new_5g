import os
import pandas as pd
import multiprocessing
from functools import partial
from datasetgen.run_pipeline import evaluate_algo1_from_csv

def process_file(args):
    scenario_id, pattern, seed, local_baseline_dir = args
    local_tick = os.path.join(local_baseline_dir, f"s{scenario_id}_{pattern}_{seed}_tick.csv")
    
    if not os.path.exists(local_tick):
        return None
        
    metrics = evaluate_algo1_from_csv(local_tick)
    print(f"Processed S{scenario_id}_{pattern} Seed {seed}", flush=True)
    return {
        "Scenario": scenario_id,
        "Pattern": pattern,
        "Seed": seed,
        "Handovers": metrics['handovers'],
        "RLF": metrics['rlf_events'],
        "PingPongs": metrics['ping_pongs']
    }

def fast_reevaluate():
    local_baseline_dir = "E:\\5g_handover\\dataset_phase1"
    
    scenarios = range(1, 8)
    patterns = ['A', 'B']
    seeds = range(1, 6)
    
    tasks = [(s, p, seed, local_baseline_dir) for s in scenarios for p in patterns for seed in seeds]
    
    print("Starting Multi-Core Offline Re-evaluation using saved datasets...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_file, tasks)
    
    global_results = [r for r in results if r is not None]
    
    results_df = pd.DataFrame(global_results)
    results_df.to_csv(os.path.join(local_baseline_dir, "algo1_stabilized_validation_results.csv"), index=False)
    
    print("\nRE-EVALUATION COMPLETE!")
    summary = results_df.groupby(['Scenario', 'Pattern'])[['Handovers', 'RLF', 'PingPongs']].mean()
    print(summary)
    
    # Dump cleanly to a markdown style for the AI to parse instantly
    summary.to_csv(os.path.join(local_baseline_dir, "algo1_stabilized_summary_dump.csv"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    fast_reevaluate()
