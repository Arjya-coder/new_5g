import os
import glob
import json
import random
import re
from typing import Optional, Tuple, List

import pandas as pd

from phase_2.algo_2 import (
    Algorithm1,
    RLModule,
    PPOAgent,
    TrainingEnv,
    train_ppo,
    evaluate_agent,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Offline ns-3 Environment Wrapper
# -----------------------------------------------------------------------------

class OfflineNs3Env:
    """
    Offline wrapper simulating the ns-3 environment using pre-generated CSVs.
    """

    def __init__(self, dataset_dir: str, csv_files: Optional[List[str]] = None):
        self.dataset_dir = dataset_dir
        self.csv_files = list(csv_files) if csv_files is not None else glob.glob(os.path.join(dataset_dir, "*_tick.csv"))

        if not self.csv_files:
            raise RuntimeError(f"No *_tick.csv files found in {dataset_dir}")

        self.current_df = None
        self.row_idx = 0
        self.current_ue = None

    def reset(self, algo1: Algorithm1):
        file_path = random.choice(self.csv_files)
        self.current_df = pd.read_csv(file_path)

        if "ue_id" not in self.current_df.columns:
            raise RuntimeError(f"Missing ue_id column in {file_path}")

        ues = self.current_df["ue_id"].unique()
        self.current_ue = random.choice(ues)

        self.current_df = self.current_df[self.current_df["ue_id"] == self.current_ue].reset_index(drop=True)

        # randomize episode start point to avoid overfitting to trace prefix
        min_remaining = 50
        max_start = max(0, len(self.current_df) - min_remaining)
        self.row_idx = random.randint(0, max_start) if max_start > 0 else 0

        out = self._process_current_row(algo1, 160, 3.0)

        # ensure first observation not repeated on first step()
        self.row_idx = min(self.row_idx + 1, len(self.current_df) - 1)
        return out

    def step(self, ttt_eff: int, hys_eff: float, algo1: Algorithm1):
        if self.is_done():
            self.row_idx = len(self.current_df) - 1  # clamp

        output = self._process_current_row(algo1, ttt_eff, hys_eff)
        self.row_idx += 1
        return output

    def is_done(self):
        return self.current_df is None or self.row_idx >= len(self.current_df) - 1

    def _process_current_row(self, algo1: Algorithm1, ttt_eff: int, hys_eff: float):
        row = self.current_df.iloc[self.row_idx]

        rsrp_neighbors = []
        sinr_neighbors = []
        neighbor_ids = []
        distance_neighbors = []

        for n_i in range(1, 7):
            n_id = row.get(f"n{n_i}_id", -1)
            if pd.isna(n_id) or int(n_id) == -1:
                continue

            neighbor_ids.append(int(n_id))
            rsrp_neighbors.append(float(row.get(f"n{n_i}_rsrp_dbm", -140.0)))
            sinr_neighbors.append(float(row.get(f"n{n_i}_sinr_db", -20.0)))
            distance_neighbors.append(float(row.get(f"n{n_i}_d_m", 200.0)))

        # initialize serving cell from row on first call
        if algo1.serving_cell_id == 0:
            algo1.serving_cell_id = int(row["serving_cell"])

        serving_rsrp = -140.0
        serving_sinr = -20.0
        distance_serving = 200.0

        def _approx_cqi_from_sinr(sinr_db: float) -> int:
            # rough mapping -20..+10 dB -> CQI 0..15
            cqi = int(round((sinr_db + 20.0) / 30.0 * 15.0))
            return max(0, min(15, cqi))

        # if algorithm serving cell matches row serving cell, use direct serving measurements
        baseline_match = int(row["serving_cell"]) == int(algo1.serving_cell_id)

        if baseline_match:
            serving_rsrp = float(row.get("serving_rsrp_dbm", -140.0))
            serving_sinr = float(row.get("serving_sinr_db", -20.0))
            if "serving_d_m" in row and not pd.isna(row.get("serving_d_m")):
                distance_serving = float(row["serving_d_m"])
        else:
            # otherwise reconstruct from neighbor columns
            found = False
            for i, nid in enumerate(neighbor_ids):
                if nid == int(algo1.serving_cell_id):
                    serving_rsrp = float(rsrp_neighbors[i])
                    serving_sinr = float(sinr_neighbors[i])
                    distance_serving = float(distance_neighbors[i])
                    found = True
                    break
            if not found:
                serving_rsrp, serving_sinr = -140.0, -30.0

        # CQI
        if baseline_match and "serving_cqi" in row and not pd.isna(row.get("serving_cqi")):
            cqi_serving = int(row["serving_cqi"])
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
            velocity=float(row.get("speed_mps", 15.0)),
            now_s=float(row.get("time_s", 0.0)),
            TTT_eff=int(ttt_eff),
            HYS_eff=float(hys_eff),
        )

        decision["scenario_id"] = int(row.get("scenario_id", 1))
        return decision


# -----------------------------------------------------------------------------
# Dataset split helpers
# -----------------------------------------------------------------------------

_TICK_RE = re.compile(r"^s(?P<scenario>\d+)_p(?P<pattern>[A-C])_seed(?P<seed>\d+)_tick\.csv$", re.IGNORECASE)


def _parse_seed_from_tick_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = _TICK_RE.match(base)
    if not m:
        return None
    return int(m.group("seed"))


def _parse_triplet_from_tick_filename(path: str) -> Optional[Tuple[int, str, int]]:
    base = os.path.basename(path)
    m = _TICK_RE.match(base)
    if not m:
        return None
    scenario_id = int(m.group("scenario"))
    pattern = str(m.group("pattern")).upper()
    seed = int(m.group("seed"))
    return scenario_id, pattern, seed


def _validate_split_full_grid(split_name: str, files: List[str], expected_seeds: List[int]) -> None:
    """
    Ensure each expected seed has full scenario-pattern coverage:
      scenarios 1..7 × patterns A/B/C = 21 files per seed
    """
    expected_scenarios = list(range(1, 8))
    expected_patterns = ["A", "B", "C"]
    expected_pairs = {(s, p) for s in expected_scenarios for p in expected_patterns}

    by_seed = {int(s): set() for s in expected_seeds}
    skipped = []

    for f in files:
        triplet = _parse_triplet_from_tick_filename(f)
        if triplet is None:
            skipped.append(f)
            continue
        s, p, seed = triplet
        if seed in by_seed:
            by_seed[int(seed)].add((int(s), str(p)))

    problems = []
    for seed in expected_seeds:
        pairs = by_seed.get(int(seed), set())
        missing = sorted(expected_pairs - pairs)
        if missing:
            missing_str = ", ".join([f"s{s}_p{p}" for s, p in missing])
            problems.append(f"seed {seed} missing {len(missing)}/21: {missing_str}")

    if problems:
        msg = (
            f"Dataset split '{split_name}' incomplete.\n"
            + "\n".join(f"- {p}" for p in problems)
        )
        if skipped:
            msg += "\nAlso skipped files with unexpected names:\n" + "\n".join(skipped)
        raise RuntimeError(msg)


def split_tick_files_by_seed(tick_files: List[str], train_seeds: List[int], val_seeds: List[int], test_seeds: List[int]):
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # IMPORTANT: keep this dataset path as your environment setup uses it
    dataset_dir = r"E:\5g_handover\dataset"

    model_dir = os.path.join(THIS_DIR, "models_rlf_fix")
    log_dir = os.path.join(THIS_DIR, "logs_rlf_fix")
    plot_dir = os.path.join(THIS_DIR, "plots_rlf_fix")  # reserved for plotting scripts

    # leakage-safe split
    train_seeds = [1, 2, 3]
    val_seeds = [4]
    test_seeds = [5]

    all_tick = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    train_files, val_files, test_files, skipped = split_tick_files_by_seed(
        all_tick,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )

    if skipped:
        print(f"Warning: {len(skipped)} files skipped due to unexpected names.")

    print("Dataset split:")
    print(f"  Train files: {len(train_files)} (seeds {train_seeds})")
    print(f"  Val files:   {len(val_files)} (seeds {val_seeds})")
    print(f"  Test files:  {len(test_files)} (seeds {test_seeds})")

    if not train_files or not val_files or not test_files:
        raise RuntimeError("Split produced empty train/val/test. Check filenames and seed lists.")

    _validate_split_full_grid("train", train_files, train_seeds)
    _validate_split_full_grid("val", val_files, val_seeds)
    _validate_split_full_grid("test", test_files, test_seeds)

    # Build environments
    rl_module = RLModule()
    control_interval_steps = 5  # hold selected TTT/HYS for 0.5s

    ns3_env_train = OfflineNs3Env(dataset_dir, csv_files=train_files)
    ns3_env_val = OfflineNs3Env(dataset_dir, csv_files=val_files)
    ns3_env_test = OfflineNs3Env(dataset_dir, csv_files=test_files)

    train_env = TrainingEnv(ns3_env_train, Algorithm1(), rl_module)
    val_env = TrainingEnv(ns3_env_val, Algorithm1(), rl_module)
    test_env = TrainingEnv(ns3_env_test, Algorithm1(), rl_module)

    # Must match algo_2.py action_dim
    agent = PPOAgent(state_dim=23, action_dim=15)

    print("\n" + "=" * 80)
    print("STARTING PHASE 2 TRAINING (RLF-FIX CONFIG)")
    print("=" * 80)
    print("State dim: 23 | Action dim: 15")
    print("Control interval steps:", control_interval_steps)
    print("=" * 80 + "\n")

    train_ppo(
        agent=agent,
        rl_module=rl_module,
        training_env=train_env,
        num_episodes=350,
        rollout_horizon=512,
        save_dir=model_dir,
        log_dir=log_dir,
        control_interval_steps=control_interval_steps,
    )

    print("\nEvaluating on validation split (greedy policy)...")
    val_metrics = evaluate_agent(
        agent=agent,
        rl_module=rl_module,
        training_env=val_env,
        num_episodes=10,
        greedy=True,
        control_interval_steps=control_interval_steps,
    )

    print("Evaluating on test split (greedy policy)...")
    test_metrics = evaluate_agent(
        agent=agent,
        rl_module=rl_module,
        training_env=test_env,
        num_episodes=10,
        greedy=True,
        control_interval_steps=control_interval_steps,
    )

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "eval_val_latest.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    with open(os.path.join(log_dir, "eval_test_latest.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Models saved to: {model_dir}")
    print(f"Logs saved to:   {log_dir}")
    print("=" * 80 + "\n")