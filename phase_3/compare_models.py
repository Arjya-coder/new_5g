import argparse
import glob
import json
import os
import sys
from collections import deque

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Append parent dir so we can import models from phase_2 and phase_1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from phase_1.algo_1 import Algorithm1 as Phase1Algo
from phase_2.algo_2 import Algorithm1 as Algo2Baseline
from phase_2.algo_2 import PPOAgent, RLModule


def _collect_dataset_files(dataset_dir):
    """Collect supported Phase 3 files (*_tick.csv and *_test_noise.csv)."""
    tick_files = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    noise_files = glob.glob(os.path.join(dataset_dir, "*_test_noise.csv"))
    files = sorted(set(tick_files + noise_files))
    return files


def _parse_neighbors(row, max_neighbors=6):
    neighbor_ids = []
    neighbor_rsrp = []
    neighbor_sinr = []
    neighbor_distances = []

    for n_i in range(1, max_neighbors + 1):
        n_id = row.get(f"n{n_i}_id", -1)
        if pd.isna(n_id) or int(n_id) == -1:
            continue

        neighbor_ids.append(int(n_id))
        neighbor_rsrp.append(float(row.get(f"n{n_i}_rsrp_dbm", -140.0)))
        neighbor_sinr.append(float(row.get(f"n{n_i}_sinr_db", -20.0)))
        neighbor_distances.append(float(row.get(f"n{n_i}_d_m", 200.0)))

    return neighbor_ids, neighbor_rsrp, neighbor_sinr, neighbor_distances


def _find_neighbor_index(neighbor_ids, target_cell):
    for i, nid in enumerate(neighbor_ids):
        if nid == target_cell:
            return i
    return None


def _resolve_serving_measurements(row,
                                  serving_cell,
                                  neighbor_ids,
                                  neighbor_rsrp,
                                  neighbor_sinr,
                                  neighbor_distances):
    """
    Resolve serving metrics for the currently connected cell.
    If the current serving cell differs from row['serving_cell'], we try to
    reconstruct from neighbor columns.
    """
    row_serving_cell = int(row["serving_cell"])

    if row_serving_cell == serving_cell:
        serving_rsrp = float(row["serving_rsrp_dbm"])
        serving_sinr = float(row["serving_sinr_db"])
    else:
        idx = _find_neighbor_index(neighbor_ids, serving_cell)
        if idx is None:
            serving_rsrp = -140.0
            serving_sinr = -20.0
        else:
            serving_rsrp = float(neighbor_rsrp[idx])
            serving_sinr = float(neighbor_sinr[idx])

    idx_dist = _find_neighbor_index(neighbor_ids, serving_cell)
    if idx_dist is None:
        serving_distance = float(row.get("serving_d_m", 200.0))
    else:
        serving_distance = float(neighbor_distances[idx_dist])

    return serving_rsrp, serving_sinr, serving_distance


def _simulate_trajectory(ue_df,
                         algo,
                         mode,
                         baseline_ttt,
                         baseline_hys,
                         rl_agent=None,
                         rl_module=None):
    """
    Simulate one UE trajectory.

    mode='baseline': static TTT/HYS baseline.
    mode='ppo':      use trained PPO policy with 1-step lag, matching training loop behavior.
    """
    if mode not in {"baseline", "ppo"}:
        raise ValueError("mode must be 'baseline' or 'ppo'")

    total_handovers = 0
    total_rlfs = 0
    total_ping_pongs = 0

    prev_serving_cell = None
    serving_cell = None
    last_ho_time = -float("inf")
    rlf_counter = 0
    time_since_last_ho = 0.0
    recent_rlf_events = deque(maxlen=10)
    recent_pp_events = deque(maxlen=10)
    rsrp_prev = None

    # PPO parameters are carried forward step-by-step from selected actions.
    ttt_rule = 160
    hys_rule = 3.0

    for _, row in ue_df.iterrows():
        now_s = float(row["time_s"])

        neighbor_ids, neighbor_rsrp, neighbor_sinr, neighbor_distances = _parse_neighbors(row)

        if serving_cell is None:
            serving_cell = int(row["serving_cell"])
            if hasattr(algo, "serving_cell_id"):
                algo.serving_cell_id = serving_cell

        serving_rsrp, serving_sinr, serving_distance = _resolve_serving_measurements(
            row,
            serving_cell,
            neighbor_ids,
            neighbor_rsrp,
            neighbor_sinr,
            neighbor_distances,
        )

        velocity = float(row.get("speed_mps", 15.0))

        if mode == "baseline":
            ttt_eff, hys_eff = int(baseline_ttt), float(baseline_hys)
        else:
            ttt_eff, hys_eff = int(ttt_rule), float(hys_rule)

        decision = algo.step(
            rsrp_serving=serving_rsrp,
            sinr_serving=serving_sinr,
            cqi_serving=10,
            distance_serving=serving_distance,
            rsrp_neighbors=neighbor_rsrp,
            neighbor_ids=neighbor_ids,
            distance_neighbors=neighbor_distances,
            velocity=velocity,
            now_s=now_s,
            TTT_eff=ttt_eff,
            HYS_eff=hys_eff,
        )

        if mode == "ppo":
            recent_rlf_count = sum(recent_rlf_events)
            recent_pp_count = sum(recent_pp_events)
            state = rl_module.build_state_vector(
                decision,
                time_since_last_ho,
                recent_rlf_count,
                recent_pp_count,
                rsrp_prev,
            )
            logits = rl_agent.actor(state.reshape(1, -1), training=False)
            action = int(np.argmax(logits[0].numpy()))
            delta_ttt, delta_hys = rl_agent.decode_action(action)
            ttt_rule, hys_rule = rl_agent.compute_effective_parameters(
                ttt_rule,
                hys_rule,
                delta_ttt,
                delta_hys,
            )

        rsrp_prev = float(decision.get("rsrp_serving_dbm", serving_rsrp))

        if serving_sinr < -20.0:
            rlf_counter += 1
        else:
            rlf_counter = 0

        rlf_event = (rlf_counter >= 2)
        if rlf_event:
            total_rlfs += 1
        recent_rlf_events.append(1 if rlf_event else 0)

        action_triggered = int(decision.get("action", 0))
        target_cell = decision.get("target_cell_id", None)
        ho_occurred = action_triggered > 0 and target_cell is not None

        ping_pong_event = False
        if ho_occurred and serving_cell is not None:
            if (
                prev_serving_cell == target_cell
                and serving_cell != prev_serving_cell
                and (now_s - last_ho_time) < 1.0
            ):
                ping_pong_event = True
                total_ping_pongs += 1
        recent_pp_events.append(1 if ping_pong_event else 0)

        time_since_last_ho += 0.1
        if ho_occurred:
            total_handovers += 1
            prev_serving_cell = serving_cell
            serving_cell = int(target_cell)
            last_ho_time = now_s
            time_since_last_ho = 0.0
            if hasattr(algo, "serving_cell_id"):
                algo.serving_cell_id = int(target_cell)

    return {
        "HOs": int(total_handovers),
        "RLFs": int(total_rlfs),
        "PingPongs": int(total_ping_pongs),
    }


def _safe_improvement(baseline_value, ppo_value):
    if baseline_value == 0:
        return 0.0
    return ((baseline_value - ppo_value) / baseline_value) * 100.0


def _plot_summary(results, output_png, baseline_ttt, baseline_hys):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    baseline_vals = [results["Baseline"]["HOs"], results["Baseline"]["RLFs"], results["Baseline"]["PingPongs"]]
    ppo_vals = [results["PPO_Agent"]["HOs"], results["PPO_Agent"]["RLFs"], results["PPO_Agent"]["PingPongs"]]
    metric_labels = ["Handovers", "RLF", "Ping-Pong"]

    improvements = [
        _safe_improvement(results["Baseline"]["HOs"], results["PPO_Agent"]["HOs"]),
        _safe_improvement(results["Baseline"]["RLFs"], results["PPO_Agent"]["RLFs"]),
        _safe_improvement(results["Baseline"]["PingPongs"], results["PPO_Agent"]["PingPongs"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Phase 3 Deployment Validation: Baseline vs Trained PPO",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(metric_labels))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label=f"Baseline (TTT={int(baseline_ttt)}, HYS={baseline_hys:.1f})", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, ppo_vals, width, label="Trained PPO", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Total Count")
    ax.set_title("Absolute Metrics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(1.0, h * 0.01), f"{int(h)}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in improvements]
    bars = ax2.bar(metric_labels, improvements, color=colors)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Improvement over Baseline (%)")
    ax2.set_title("Relative Improvement")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, improvements):
        y = bar.get_height()
        offset = 0.8 if y >= 0 else -1.2
        ax2.text(bar.get_x() + bar.get_width() / 2.0, y + offset, f"{val:+.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparisons(dataset_dir, model_dir, output_dir, baseline_ttt=250, baseline_hys=4.5):
    csv_files = _collect_dataset_files(dataset_dir)
    if not csv_files:
        raise RuntimeError(f"No supported dataset files found in: {dataset_dir}")

    actor_path = os.path.join(model_dir, "actor.h5")
    critic_path = os.path.join(model_dir, "critic.h5")
    if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
        raise FileNotFoundError(
            f"Trained model not found at {model_dir}. Expected actor.h5 and critic.h5."
        )

    print(f"Loaded {len(csv_files)} Phase 3 dataset files from: {dataset_dir}")
    print(f"Using trained model from: {model_dir}")

    rl_agent = PPOAgent(state_dim=20, action_dim=15)
    rl_agent.load(model_dir)
    rl_module = RLModule()

    summary = {
        "Baseline": {"HOs": 0, "RLFs": 0, "PingPongs": 0},
        "PPO_Agent": {"HOs": 0, "RLFs": 0, "PingPongs": 0},
    }
    per_file_rows = []

    for idx, file_path in enumerate(csv_files, start=1):
        df = pd.read_csv(file_path)
        if "ue_id" not in df.columns:
            print(f"Skipping {os.path.basename(file_path)} (missing ue_id column)")
            continue

        base_metrics = {"HOs": 0, "RLFs": 0, "PingPongs": 0}
        ppo_metrics = {"HOs": 0, "RLFs": 0, "PingPongs": 0}

        for _, ue_df in df.groupby("ue_id"):
            ue_df = ue_df.sort_values("time_s").reset_index(drop=True)

            base_algo = Phase1Algo()
            m_base = _simulate_trajectory(
                ue_df,
                base_algo,
                mode="baseline",
                baseline_ttt=baseline_ttt,
                baseline_hys=baseline_hys,
            )

            ppo_algo = Algo2Baseline()
            m_ppo = _simulate_trajectory(
                ue_df,
                ppo_algo,
                mode="ppo",
                baseline_ttt=baseline_ttt,
                baseline_hys=baseline_hys,
                rl_agent=rl_agent,
                rl_module=rl_module,
            )

            for k in base_metrics:
                base_metrics[k] += m_base[k]
                ppo_metrics[k] += m_ppo[k]

        for k in summary["Baseline"]:
            summary["Baseline"][k] += base_metrics[k]
            summary["PPO_Agent"][k] += ppo_metrics[k]

        per_file_rows.append(
            {
                "file": os.path.basename(file_path),
                "algorithm": "Baseline",
                "HOs": base_metrics["HOs"],
                "RLFs": base_metrics["RLFs"],
                "PingPongs": base_metrics["PingPongs"],
            }
        )
        per_file_rows.append(
            {
                "file": os.path.basename(file_path),
                "algorithm": "PPO_Agent",
                "HOs": ppo_metrics["HOs"],
                "RLFs": ppo_metrics["RLFs"],
                "PingPongs": ppo_metrics["PingPongs"],
            }
        )

        print(f"[{idx}/{len(csv_files)}] Processed {os.path.basename(file_path)}")

    os.makedirs(output_dir, exist_ok=True)

    per_file_csv = os.path.join(output_dir, "per_file_metrics_v3.csv")
    pd.DataFrame(per_file_rows).to_csv(per_file_csv, index=False)

    improvements = {
        "HO_reduction_percent": _safe_improvement(summary["Baseline"]["HOs"], summary["PPO_Agent"]["HOs"]),
        "RLF_reduction_percent": _safe_improvement(summary["Baseline"]["RLFs"], summary["PPO_Agent"]["RLFs"]),
        "PingPong_reduction_percent": _safe_improvement(summary["Baseline"]["PingPongs"], summary["PPO_Agent"]["PingPongs"]),
    }

    summary_payload = {
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "baseline": {
            "ttt_ms": int(baseline_ttt),
            "hys_db": float(baseline_hys),
            "metrics": summary["Baseline"],
        },
        "ppo_agent": {
            "metrics": summary["PPO_Agent"],
        },
        "improvements": improvements,
        "files_evaluated": len(csv_files),
    }

    summary_json = os.path.join(output_dir, "comparison_summary_v3.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    plot_path = os.path.join(output_dir, "comparison_results_v3.png")
    _plot_summary(summary, plot_path, baseline_ttt=baseline_ttt, baseline_hys=baseline_hys)

    print("\n--- PHASE 3 SUMMARY ---")
    print("Baseline:")
    print(f"  HOs:       {summary['Baseline']['HOs']}")
    print(f"  RLFs:      {summary['Baseline']['RLFs']}")
    print(f"  PingPongs: {summary['Baseline']['PingPongs']}")
    print("PPO Agent:")
    print(f"  HOs:       {summary['PPO_Agent']['HOs']}")
    print(f"  RLFs:      {summary['PPO_Agent']['RLFs']}")
    print(f"  PingPongs: {summary['PPO_Agent']['PingPongs']}")
    print("Improvements over baseline:")
    print(f"  HO reduction:        {improvements['HO_reduction_percent']:+.2f}%")
    print(f"  RLF reduction:       {improvements['RLF_reduction_percent']:+.2f}%")
    print(f"  PingPong reduction:  {improvements['PingPong_reduction_percent']:+.2f}%")
    print(f"\nSaved per-file metrics: {per_file_csv}")
    print(f"Saved summary JSON:     {summary_json}")
    print(f"Saved plot:             {plot_path}")


def _build_arg_parser():
    default_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset")
    default_model_dir = os.path.join(REPO_ROOT, "phase_2", "models", "final")
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    parser = argparse.ArgumentParser(
        description="Phase 3 evaluator: compare Phase 1 baseline vs Phase 2 trained PPO model"
    )
    parser.add_argument("--dataset-dir", default=default_dataset_dir, help="Directory containing Phase 3 CSV files")
    parser.add_argument("--model-dir", default=default_model_dir, help="Directory containing actor.h5 and critic.h5")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save Phase 3 outputs")
    parser.add_argument("--baseline-ttt", type=int, default=250, help="Baseline static TTT (ms)")
    parser.add_argument("--baseline-hys", type=float, default=4.5, help="Baseline static HYS (dB)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_comparisons(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        baseline_ttt=args.baseline_ttt,
        baseline_hys=args.baseline_hys,
    )
