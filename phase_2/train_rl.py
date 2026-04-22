import os
import glob
import json
import random
from typing import Optional, List

import pandas as pd

from phase_1.algo_1 import Algorithm1
from phase_2.algo_2 import (
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
    One episode = one random UE trace from one random tick file, starting from a random row.
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

        # randomize episode start to avoid overfitting to trace prefix
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
            cqi = int(round((sinr_db + 20.0) / 30.0 * 15.0))
            return max(0, min(15, cqi))

        baseline_match = int(row["serving_cell"]) == int(algo1.serving_cell_id)

        if baseline_match:
            serving_rsrp = float(row.get("serving_rsrp_dbm", -140.0))
            serving_sinr = float(row.get("serving_sinr_db", -20.0))
            if "serving_d_m" in row and not pd.isna(row.get("serving_d_m")):
                distance_serving = float(row["serving_d_m"])
        else:
            # reconstruct from neighbor columns
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

        decision["scenario_id"] = int(row.get("scenario_id", 0))
        return decision


# -----------------------------------------------------------------------------
# Split helpers (filename-agnostic)
# -----------------------------------------------------------------------------

def make_split_manifest(all_files: List[str], total_files: int, seed: int = 123):
    rng = random.Random(seed)
    files = list(all_files)
    rng.shuffle(files)

    if len(files) < total_files:
        raise RuntimeError(f"Need at least {total_files} tick files, but found {len(files)}")

    files = files[:total_files]

    # Good default for 105: 80/15/10
    n_train, n_val = 80, 15
    n_test = total_files - n_train - n_val
    assert n_test > 0

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return {"train": train, "val": val, "test": test}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_dir = r"E:\5g_handover\dataset"

    model_dir = os.path.join(THIS_DIR, "models_rlf_fix")
    log_dir = os.path.join(THIS_DIR, "logs_rlf_fix")
    os.makedirs(log_dir, exist_ok=True)

    manifest_path = os.path.join(log_dir, "split_manifest.json")

    all_tick = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    if not all_tick:
        raise RuntimeError(f"No *_tick.csv files found in {dataset_dir}")

    TOTAL_FILES = 105
    split = make_split_manifest(all_tick, total_files=TOTAL_FILES, seed=123)

    print("Dataset split:")
    print("  Train files:", len(split["train"]))
    print("  Val files:  ", len(split["val"]))
    print("  Test files: ", len(split["test"]))
    print("  Total:      ", sum(len(v) for v in split.values()))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print("Saved split manifest to:", manifest_path)

    rl_module = RLModule()
    control_interval_steps = 5  # hold selected TTT/HYS for 0.5s

    ns3_env_train = OfflineNs3Env(dataset_dir, csv_files=split["train"])
    ns3_env_val = OfflineNs3Env(dataset_dir, csv_files=split["val"])
    ns3_env_test = OfflineNs3Env(dataset_dir, csv_files=split["test"])

    train_env = TrainingEnv(ns3_env_train, Algorithm1(), rl_module)
    val_env = TrainingEnv(ns3_env_val, Algorithm1(), rl_module)
    test_env = TrainingEnv(ns3_env_test, Algorithm1(), rl_module)

    agent = PPOAgent(state_dim=23, action_dim=15)

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

    with open(os.path.join(log_dir, "eval_val_latest.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    with open(os.path.join(log_dir, "eval_test_latest.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nDone.")
    print("Models:", model_dir)
    print("Logs:  ", log_dir)