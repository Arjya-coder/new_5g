"""
PHASE 2 TRAINING RUNNER - RLF FIX ONLY
Minimal changes: Only critical fix #5 applied
"""

import os
import glob
import random
import re
import json
import pandas as pd
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from phase_2.algo_2 import Algorithm1, RLModule, PPOAgent, TrainingEnv, train_ppo, evaluate_agent
from phase_2.plot_metrics import plot_metrics

class OfflineNs3Env:
    """Offline wrapper simulating ns-3 (UNCHANGED)"""
    def __init__(self, dataset_dir, csv_files=None):
        self.csv_files = list(csv_files) if csv_files is not None else glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
        if not self.csv_files:
            raise RuntimeError(f"No *_tick.csv files found in {dataset_dir}")
        self.current_df = None
        self.row_idx = 0
        self.current_ue = None
        
    def reset(self, algo1):
        file_path = random.choice(self.csv_files)
        self.current_df = pd.read_csv(file_path)
        ues = self.current_df['ue_id'].unique()
        self.current_ue = random.choice(ues)
        self.current_df = self.current_df[self.current_df['ue_id'] == self.current_ue].reset_index(drop=True)
        min_remaining = 50
        max_start = max(0, len(self.current_df) - min_remaining)
        self.row_idx = random.randint(0, max_start) if max_start > 0 else 0

        out = self._process_current_row(algo1, 160, 3.0)
        self.row_idx = min(self.row_idx + 1, len(self.current_df) - 1)
        return out
        
    def step(self, ttt_eff, hys_eff, algo1):
        if self.is_done():
            self.row_idx = len(self.current_df) - 1
        output = self._process_current_row(algo1, ttt_eff, hys_eff)
        self.row_idx += 1
        return output
        
    def is_done(self):
        return self.current_df is None or self.row_idx >= len(self.current_df) - 1
        
    def _process_current_row(self, algo1, ttt_eff, hys_eff):
        row = self.current_df.iloc[self.row_idx]
        
        rsrp_neighbors = []
        sinr_neighbors = []
        neighbor_ids = []
        distance_neighbors = []
        
        for n_i in range(1, 7):
            n_id = row.get(f'n{n_i}_id', -1)
            if pd.isna(n_id) or n_id == -1: continue
            neighbor_ids.append(int(n_id))
            rsrp_neighbors.append(float(row[f'n{n_i}_rsrp_dbm']))
            sinr_neighbors.append(float(row[f'n{n_i}_sinr_db']))
            distance_neighbors.append(float(row[f'n{n_i}_d_m']))
            
        if algo1.serving_cell_id == 0:
            algo1.serving_cell_id = int(row['serving_cell'])
            
        serving_rsrp = -140.0
        serving_sinr = -20.0
        distance_serving = 200.0

        def _approx_cqi_from_sinr(sinr_db: float) -> int:
            cqi = int(round((sinr_db + 20.0) / 30.0 * 15.0))
            return max(0, min(15, cqi))
        
        baseline_match = int(row['serving_cell']) == algo1.serving_cell_id

        if baseline_match:
            serving_rsrp = float(row['serving_rsrp_dbm'])
            serving_sinr = float(row['serving_sinr_db'])
        else:
            found = False
            for i, nid in enumerate(neighbor_ids):
                if nid == algo1.serving_cell_id:
                    serving_rsrp = rsrp_neighbors[i]
                    serving_sinr = float(sinr_neighbors[i])
                    distance_serving = float(distance_neighbors[i])
                    found = True
                    break
            if not found:
                serving_rsrp, serving_sinr = -140.0, -30.0

        if baseline_match:
            if 'serving_d_m' in row and not pd.isna(row['serving_d_m']):
                distance_serving = float(row['serving_d_m'])
            if 'serving_cqi' in row and not pd.isna(row.get('serving_cqi')):
                cqi_serving = int(row['serving_cqi'])
            else:
                cqi_serving = _approx_cqi_from_sinr(serving_sinr)
        else:
            cqi_serving = _approx_cqi_from_sinr(serving_sinr)
                
        decision = algo1.step(
            rsrp_serving=serving_rsrp,
            sinr_serving=serving_sinr,
            cqi_serving=cqi_serving,
            distance_serving=distance_serving,
            rsrp_neighbors=rsrp_neighbors,
            neighbor_ids=neighbor_ids,
            distance_neighbors=distance_neighbors,
            velocity=float(row.get('speed_mps', 15.0)),
            now_s=float(row['time_s']),
            TTT_eff=ttt_eff,
            HYS_eff=hys_eff
        )
        decision['scenario_id'] = int(row.get('scenario_id', 1))
        return decision


_TICK_RE = re.compile(r"^s(?P<scenario>\d+)_p(?P<pattern>[A-C])_seed(?P<seed>\d+)_tick\.csv$", re.IGNORECASE)

def _parse_seed_from_tick_filename(path: str) -> int | None:
    base = os.path.basename(path)
    m = _TICK_RE.match(base)
    if not m:
        return None
    return int(m.group('seed'))

def _parse_triplet_from_tick_filename(path: str) -> tuple[int, str, int] | None:
    base = os.path.basename(path)
    m = _TICK_RE.match(base)
    if not m:
        return None
    scenario_id = int(m.group('scenario'))
    pattern = str(m.group('pattern')).upper()
    seed = int(m.group('seed'))
    return scenario_id, pattern, seed

def _validate_split_full_grid(*, split_name: str, files: list[str], expected_seeds: list[int]) -> None:
    expected_scenarios = list(range(1, 8))
    expected_patterns = ["A", "B", "C"]
    expected_pairs = {(s, p) for s in expected_scenarios for p in expected_patterns}

    parsed = []
    skipped = []
    for f in files:
        t = _parse_triplet_from_tick_filename(f)
        if t is None:
            skipped.append(f)
            continue
        parsed.append((f, t[0], t[1], t[2]))

    by_seed: dict[int, set[tuple[int, str]]] = {int(s): set() for s in expected_seeds}
    for _, scenario_id, pattern, seed in parsed:
        if seed in by_seed:
            by_seed[int(seed)].add((int(scenario_id), str(pattern)))

    problems: list[str] = []
    for seed in expected_seeds:
        pairs = by_seed.get(int(seed), set())
        missing = sorted(expected_pairs - pairs)
        if missing:
            missing_str = ", ".join([f"s{s}_p{p}" for s, p in missing])
            problems.append(f"seed {seed} missing {len(missing)}/21: {missing_str}")

    if problems:
        raise RuntimeError(
            f"Dataset split '{split_name}' is incomplete.\n"
            + "\n".join(f"- {p}" for p in problems)
        )

def split_tick_files_by_seed(tick_files, train_seeds, val_seeds, test_seeds):
    train, val, test, skipped = [], [], [], []
    train_seeds = set(int(s) for s in train_seeds)
    val_seeds = set(int(s) for s in val_seeds)
    test_seeds = set(int(s) for s in test_seeds)

    for f in tick_files:
        seed = _parse_seed_from_tick_filename(f)
        if seed is None:
            skipped.append(f)
            continue
        if seed in train_seeds:
            train.append(f)
        elif seed in val_seeds:
            val.append(f)
        elif seed in test_seeds:
            test.append(f)
        else:
            skipped.append(f)

    return train, val, test, skipped

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 2 RETRAINING - RLF CRITICAL FIX ONLY")
    print("=" * 80)
    print("\n✅ CRITICAL FIX #5:")
    print("   Velocity-Aware RLF Penalty")
    print("   └─ -2.0 × (1 + velocity/50)")
    print("\n✅ PRESERVED (100% backward compatible):")
    print("   ├─ HO penalty: -0.5 (unchanged)")
    print("   ├─ PP penalty: -1.0 (unchanged)")
    print("   ├─ State: 23-dim (unchanged)")
    print("   └─ Actions: 15-dim (unchanged)")
    print("\n" + "=" * 80 + "\n")
    
    dataset_dir = r"E:\5g_handover\dataset"
    model_dir = os.path.join(THIS_DIR, "models_rlf_fix")
    log_dir = os.path.join(THIS_DIR, "logs_rlf_fix")
    plot_dir = os.path.join(THIS_DIR, "plots_rlf_fix")

    train_seeds = [1, 2, 3]
    val_seeds = [4]
    test_seeds = [5]

    all_tick = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    train_files, val_files, test_files, skipped = split_tick_files_by_seed(
        all_tick, train_seeds=train_seeds, val_seeds=val_seeds, test_seeds=test_seeds
    )

    if skipped:
        print(f"Warning: {len(skipped)} tick files skipped (unexpected name).")

    print("Dataset split:")
    print(f"  Train files: {len(train_files)} (seeds {train_seeds})")
    print(f"  Val files:   {len(val_files)} (seeds {val_seeds})")
    print(f"  Test files:  {len(test_files)} (seeds {test_seeds})\n")

    if not train_files or not val_files or not test_files:
        raise RuntimeError("Split produced an empty set.")

    _validate_split_full_grid(split_name="train", files=train_files, expected_seeds=train_seeds)
    _validate_split_full_grid(split_name="val", files=val_files, expected_seeds=val_seeds)
    _validate_split_full_grid(split_name="test", files=test_files, expected_seeds=test_seeds)

    # Build environments
    rl_module = RLModule()
    control_interval_steps = 5

    ns3_env_train = OfflineNs3Env(dataset_dir, csv_files=train_files)
    ns3_env_val = OfflineNs3Env(dataset_dir, csv_files=val_files)
    ns3_env_test = OfflineNs3Env(dataset_dir, csv_files=test_files)

    train_env = TrainingEnv(ns3_env_train, Algorithm1(), rl_module)
    val_env = TrainingEnv(ns3_env_val, Algorithm1(), rl_module)
    test_env = TrainingEnv(ns3_env_test, Algorithm1(), rl_module)

    agent = PPOAgent(state_dim=23, action_dim=15)

    # Train
    train_ppo(
        agent,
        rl_module,
        train_env,
        num_episodes=350,
        rollout_horizon=512,
        save_dir=model_dir,
        log_dir=log_dir,
        control_interval_steps=control_interval_steps,
    )

    # Evaluate
    print("\nEvaluating on validation split...")
    val_metrics = evaluate_agent(agent, rl_module, val_env, num_episodes=10, greedy=True,
                                 control_interval_steps=control_interval_steps)

    print("Evaluating on test split...")
    test_metrics = evaluate_agent(agent, rl_module, test_env, num_episodes=10, greedy=True,
                                  control_interval_steps=control_interval_steps)

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "eval_val_latest.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(log_dir, "eval_test_latest.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Generate plots from the latest metrics JSON in this run
    try:
        plot_metrics(log_dir=log_dir, save_dir=plot_dir)
        print(f"Plots saved to: {plot_dir}")
    except Exception as e:
        print(f"Warning: failed to generate plots: {e}")

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Models saved to: {model_dir}")
    print(f"Logs saved to: {log_dir}")
    print("\nNext: Run Phase 3 evaluation on the new RLF-fixed models")
    print("=" * 80 + "\n")