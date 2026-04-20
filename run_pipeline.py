import subprocess
import pandas as pd
import os
import shutil
import json
from typing import Tuple

from phase_1.algo_1 import Algorithm1


WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_CPP_PATH = os.path.join(WORKSPACE_ROOT, "dataset.cpp")
LOCAL_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "dataset")


def _unc_join_posix(unc_root: str, posix_path: str) -> str:
    """Join a Windows UNC root with a POSIX-style relative path."""
    parts = [p for p in posix_path.split('/') if p]
    return os.path.join(unc_root, *parts)


def deploy_and_run_ns3(
    scenario_id: int,
    pattern: str,
    seed: int,
    wsl_unc_ns3_root: str = "\\\\wsl$\\Ubuntu\\home\\arjyadeep\\ns-3-dev",
    wsl_ns3_root: str = "~/ns-3-dev",
    duration: int = 0,
) -> Tuple[str, str, str]:
    """
    Deploys dataset.cpp to the ns-3 scratch folder and executes the simulation for a specific seed.
    """
    dest_path = os.path.join(wsl_unc_ns3_root, "scratch", "md_scenarios.cc")
    
    # Ensure C++ file is present in ns3
    shutil.copy(DATASET_CPP_PATH, dest_path)

    run_prefix = f"s{scenario_id}_p{pattern}_seed{seed}"
    output_prefix = f"phase1_baseline/{run_prefix}"
    
    # We instruct ns-3 to save it directly into the phase1_baseline folder inside ns-3-dev
    bash_cmd = (
        f"cd {wsl_ns3_root} && "
        f"mkdir -p phase1_baseline && "
        f"./ns3 run 'scratch/md_scenarios --scenarioId={scenario_id} --pattern={pattern} "
        f"--seed={seed} --duration={duration} --outputPrefix={output_prefix}'"
    )
    
    print(f"Executing: {run_prefix}...")
    
    cmd_args = ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", bash_cmd]
    result = subprocess.run(
        cmd_args, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            "ns-3 command failed.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    
    # Paths in WSL
    wsl_tick_path = _unc_join_posix(wsl_unc_ns3_root, f"{output_prefix}_tick.csv")
    wsl_event_path = _unc_join_posix(wsl_unc_ns3_root, f"{output_prefix}_events.csv")
    wsl_summary_path = _unc_join_posix(wsl_unc_ns3_root, f"{output_prefix}_summary.json")
    
    return wsl_tick_path, wsl_event_path, wsl_summary_path


def _scenario_rlf_threshold_dbm(scenario_id: int) -> float:
    # Mirrors the per-scenario thresholds in dataset.cpp.
    return {
        1: -122.0,
        2: -121.0,
        3: -124.0,
        4: -122.0,
        5: -125.0,
        6: -123.0,
        7: -118.0,
    }.get(int(scenario_id), -122.0)

def evaluate_algo1_from_csv(csv_path):
    """
    Evaluates Algorithm 1 offline and returns metrics.
    """
    df = pd.read_csv(csv_path)
    ues = df['ue_id'].unique()
    algo_instances = {ue: Algorithm1() for ue in ues}
    
    metrics = {
        "handovers": 0,
        "emergency_hos": 0,
        "rlf_events": 0,
        "ping_pongs": 0
    }
    
    ue_sim_states = {ue: {"serving_cell": None, "low_rsrp_ticks": 0, "last_ho_to": None, "last_ho_time": -100} for ue in ues}
    
    for idx, row in df.iterrows():
        ue = row['ue_id']
        algo = algo_instances[ue]
        sim_state = ue_sim_states[ue]
        
        rsrp_neighbors = []
        sinr_neighbors = []
        neighbor_ids = []
        distance_neighbors = []
        
        for n_i in range(1, 7):
            n_id = row.get(f'n{n_i}_id', -1)
            if pd.isna(n_id) or n_id == -1: continue
            neighbor_ids.append(int(n_id))
            rsrp_neighbors.append(row[f'n{n_i}_rsrp_dbm'])
            sinr_neighbors.append(row[f'n{n_i}_sinr_db'])
            distance_neighbors.append(row[f'n{n_i}_d_m'])
            
        if sim_state["serving_cell"] is None:
            sim_state["serving_cell"] = int(row['serving_cell'])
            
        serving_rsrp = -140.0
        serving_sinr = -20.0
        distance_serving = 200.0
        
        if int(row['serving_cell']) == sim_state["serving_cell"]:
            serving_rsrp = row['serving_rsrp_dbm']
            serving_sinr = row['serving_sinr_db']
            for i, nid in enumerate(neighbor_ids):
                if nid == sim_state["serving_cell"]:
                    distance_serving = float(distance_neighbors[i])
                    break
            else:
                if distance_neighbors:
                    distance_serving = float(min(distance_neighbors))
        else:
            found = False
            for i, nid in enumerate(neighbor_ids):
                if nid == sim_state["serving_cell"]:
                    serving_rsrp = rsrp_neighbors[i]
                    serving_sinr = sinr_neighbors[i]
                    distance_serving = float(distance_neighbors[i])
                    found = True
                    break
            if not found:
                serving_rsrp, serving_sinr = -140.0, -20.0

        decision = algo.step(
            rsrp_serving=serving_rsrp,
            sinr_serving=serving_sinr,
            cqi_serving=10, 
            distance_serving=distance_serving,
            rsrp_neighbors=rsrp_neighbors,
            neighbor_ids=neighbor_ids,
            distance_neighbors=distance_neighbors,
            velocity=row['speed_mps'],
            now_s=row['time_s'],
            TTT_eff=250,
            HYS_eff=4.5
        )
        
        if decision['action'] != 0:
            target_id = decision['target_cell_id']
            metrics["handovers"] += 1
            if "MUST-HO" in decision['reason']:
                metrics["emergency_hos"] += 1
            if sim_state["last_ho_to"] == sim_state["serving_cell"] and (row['time_s'] - sim_state["last_ho_time"] < 5.0):
                metrics["ping_pongs"] += 1
                
            sim_state["last_ho_to"] = target_id
            sim_state["last_ho_time"] = row['time_s']
            sim_state["serving_cell"] = target_id
            
        scenario_id = int(row.get('scenario_id', 1))
        if serving_rsrp < _scenario_rlf_threshold_dbm(scenario_id):
            sim_state["low_rsrp_ticks"] += 1
        else:
            sim_state["low_rsrp_ticks"] = 0
            
        if sim_state["low_rsrp_ticks"] >= 2:
            metrics["rlf_events"] += 1
            sim_state["low_rsrp_ticks"] = 0 
            
    return metrics

def run_phase1_validation():
    # Setup local organized directory
    local_baseline_dir = LOCAL_DATASET_DIR
    if not os.path.exists(local_baseline_dir):
        os.makedirs(local_baseline_dir)
        
    scenarios = range(1, 8) # 1 through 7
    patterns = ['A', 'B', 'C']
    seeds = range(1, 6) # 1 through 5

    evaluate_algo1 = False  # Dataset generation is the priority; enable if you want the offline Algo1 metrics.
    
    total_runs = len(scenarios) * len(patterns) * len(seeds)
    current_run = 0
    
    global_results = []
    
    print(f"Starting Phase 1 Full Validation: {total_runs} runs.")
    
    for scenario_id in scenarios:
        for pattern in patterns:
            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running S{scenario_id} Pattern {pattern} Seed {seed}...")
                
                try:
                    # Execute in WSL
                    wsl_tick, wsl_event, wsl_summary = deploy_and_run_ns3(scenario_id, pattern, seed)

                    run_prefix = f"s{scenario_id}_p{pattern}_seed{seed}"
                    
                    # Store locally in workspace
                    local_tick = os.path.join(local_baseline_dir, f"{run_prefix}_tick.csv")
                    local_event = os.path.join(local_baseline_dir, f"{run_prefix}_events.csv")
                    local_summary = os.path.join(local_baseline_dir, f"{run_prefix}_summary.json")
                    
                    # Copy from \\wsl$\... to e:\5g_handover\phase1_baseline\
                    if os.path.exists(wsl_tick):
                        shutil.copy(wsl_tick, local_tick)
                        shutil.copy(wsl_event, local_event)
                        shutil.copy(wsl_summary, local_summary)
                    else:
                        print("Error: WSL files were not generated.")
                        continue

                    # Always record the ns-3 generator summary (fast).
                    with open(local_summary, 'r', encoding='utf-8') as f:
                        summary = json.load(f)

                    row_out = {
                        "Scenario": scenario_id,
                        "Pattern": pattern,
                        "Seed": seed,
                        "Ns3_TotalHandovers": summary.get("total_handovers"),
                        "Ns3_TotalRlf": summary.get("total_rlf"),
                        "Ns3_TotalPingPong": summary.get("total_ping_pong"),
                        "Ns3_HandoverPerMin": summary.get("handover_per_min"),
                        "DurationS": summary.get("duration_s"),
                        "UeCount": summary.get("ue_count"),
                    }

                    if evaluate_algo1:
                        algo1_metrics = evaluate_algo1_from_csv(local_tick)
                        print(f"  > Algo1 Handovers: {algo1_metrics['handovers']}")
                        print(f"  > Algo1 RLFs:      {algo1_metrics['rlf_events']}")
                        print(f"  > Algo1 PingPongs: {algo1_metrics['ping_pongs']}")
                        row_out.update({
                            "Algo1_Handovers": algo1_metrics['handovers'],
                            "Algo1_RLF": algo1_metrics['rlf_events'],
                            "Algo1_PingPongs": algo1_metrics['ping_pongs'],
                        })

                    global_results.append(row_out)
                    
                except Exception as e:
                    print(f"Error executing S{scenario_id}_{pattern}_{seed}: {e}")
                    
    # Export full metrics
    results_df = pd.DataFrame(global_results)
    results_df.to_csv(os.path.join(local_baseline_dir, "algo1_validation_results.csv"), index=False)
    print("\nPhase 1 Validation Full Generation Complete!")
    print(f"All datasets stored under `{local_baseline_dir}` alongside `algo1_validation_results.csv`!")

if __name__ == "__main__":
    run_phase1_validation()
