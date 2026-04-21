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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from phase_1.algo_1 import Algorithm1 as Phase1Algo
from phase_2.algo_2 import Algorithm1 as Algo2Baseline
from phase_2.algo_2 import PPOAgent, RLModule


class _NumpyActorPolicy:
    """Fast NumPy forward-pass for Dense MLP actor models."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray, str]]):
        self.layers = [
            (np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32), str(act))
            for w, b, act in layers
        ]

    @classmethod
    def from_keras_model(cls, model) -> "_NumpyActorPolicy":
        layers = list(getattr(model, "layers", []))

        dense_layers: list[tuple[np.ndarray, np.ndarray, str]] = []
        for layer in layers:
            layer_type = layer.__class__.__name__

            if layer_type == "Dense":
                weights = layer.get_weights()
                if len(weights) != 2:
                    raise ValueError(
                        f"Dense layer weights unexpected: expected 2 arrays, got {len(weights)}"
                    )

                w, b = weights
                activation = getattr(layer, "activation", None)
                act_name = (
                    getattr(activation, "__name__", "linear")
                    if activation is not None
                    else "linear"
                )
                dense_layers.append((w, b, act_name))
                continue

            get_weights = getattr(layer, "get_weights", None)
            if callable(get_weights) and get_weights():
                raise ValueError(
                    f"Unsupported actor layer type '{layer_type}' with weights; "
                    "NumPy policy only supports Dense MLP actors."
                )

        if not dense_layers:
            raise ValueError("Actor model has no Dense layers; cannot build NumPy policy.")

        return cls(dense_layers)

    def argmax_action(self, state: np.ndarray) -> int:
        x = np.asarray(state, dtype=np.float32).reshape(-1)
        for w, b, act in self.layers:
            x = x @ w + b

            if act == "relu":
                np.maximum(x, 0.0, out=x)
            elif act in {"linear", "identity"}:
                pass
            elif act == "tanh":
                np.tanh(x, out=x)
            else:
                raise ValueError(
                    f"Unsupported activation '{act}' in actor; supported: relu, linear, tanh"
                )

        return int(np.argmax(x))


def _approx_cqi_from_sinr(sinr_db: float) -> int:
    cqi = int(round((float(sinr_db) + 20.0) / 30.0 * 15.0))
    return max(0, min(15, cqi))


def _scenario_rlf_threshold_dbm(scenario_id: int) -> float:
    return {
        1: -122.0,
        2: -121.0,
        3: -124.0,
        4: -122.0,
        5: -125.0,
        6: -123.0,
        7: -118.0,
        8: -120.0,
        9: -123.0,
        10: -126.0,
        11: -124.0,
        12: -123.0,
        13: -127.0,
        14: -121.0,
        15: -125.0,
    }.get(int(scenario_id), -122.0)


def _infer_state_dim(rl_module: RLModule) -> int:
    dummy = {
        "rsrp_serving_dbm": -100.0,
        "sinr_serving_db": 0.0,
        "cqi_serving": 10,
        "distance_serving": 200.0,
        "zone": "HANDOFF_ZONE",
        "rsrp_neighbors": [-95.0, -105.0, -110.0],
        "distance_neighbors": [250.0, 300.0, 350.0],
        "signal_quality": "FAIR",
        "velocity": 15.0,
        "num_neighbors": 3,
    }

    state = rl_module.build_state_vector(
        algo1_output=dummy,
        ttt_eff=160,
        hys_eff=3.0,
        time_since_last_ho=0.0,
        recent_rlf_count=0,
        recent_pp_count=0,
        recent_ho_count=0,
        rsrp_prev=None,
    )
    return int(np.asarray(state).shape[0])


def _collect_dataset_files(dataset_dir: str) -> list[str]:
    tick_files = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    noise_files = glob.glob(os.path.join(dataset_dir, "*_test_noise.csv"))
    return sorted(set(tick_files + noise_files))


def _parse_neighbors(row, max_neighbors: int = 6):
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


def _resolve_serving_measurements(
    row,
    serving_cell,
    neighbor_ids,
    neighbor_rsrp,
    neighbor_sinr,
    neighbor_distances,
):
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


def _simulate_trajectory(
    ue_df,
    algo,
    mode,
    static_ttt,
    static_hys,
    ppo_init_ttt=160,
    ppo_init_hys=3.0,
    rl_agent=None,
    rl_module=None,
    actor_policy: _NumpyActorPolicy | None = None,
    control_interval_steps: int = 5,
):
    if mode not in {"algo1", "algo2", "ppo"}:
        raise ValueError("mode must be one of {'algo1', 'algo2', 'ppo'}")

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
    recent_ho_events = deque(maxlen=10)
    rsrp_prev = None
    in_handover = False

    ttt_rule = int(ppo_init_ttt)
    hys_rule = float(ppo_init_hys)

    control_interval_steps = int(max(1, control_interval_steps))
    steps_until_policy_update = 0

    for _, row in ue_df.iterrows():
        now_s = float(row["time_s"])
        scenario_id = int(row.get("scenario_id", 1))

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

        if mode == "ppo":
            ttt_eff, hys_eff = int(ttt_rule), float(hys_rule)
        else:
            ttt_eff, hys_eff = int(static_ttt), float(static_hys)

        row_serving_cell = int(row["serving_cell"])
        if row_serving_cell == int(serving_cell):
            serving_cqi_raw = row.get("serving_cqi", None)
            if serving_cqi_raw is not None and not pd.isna(serving_cqi_raw):
                cqi_serving = int(serving_cqi_raw)
            else:
                cqi_serving = _approx_cqi_from_sinr(serving_sinr)
        else:
            cqi_serving = _approx_cqi_from_sinr(serving_sinr)

        decision = algo.step(
            rsrp_serving=serving_rsrp,
            sinr_serving=serving_sinr,
            cqi_serving=int(cqi_serving),
            distance_serving=serving_distance,
            rsrp_neighbors=neighbor_rsrp,
            neighbor_ids=neighbor_ids,
            distance_neighbors=neighbor_distances,
            velocity=velocity,
            now_s=now_s,
            TTT_eff=ttt_eff,
            HYS_eff=hys_eff,
        )
        decision["scenario_id"] = int(scenario_id)

        rsrp_current = float(decision.get("rsrp_serving_dbm", serving_rsrp))

        rlf_threshold = _scenario_rlf_threshold_dbm(scenario_id)
        if rsrp_current < rlf_threshold:
            rlf_counter += 1
        else:
            rlf_counter = 0

        rlf_event = rlf_counter >= 2
        if rlf_event:
            rlf_counter = 0
            total_rlfs += 1
        recent_rlf_events.append(1 if rlf_event else 0)

        target_cell = decision.get("target_cell_id", None)

        if "ho_occurred" in decision:
            ho_event = bool(decision.get("ho_occurred"))
        else:
            action_triggered = int(decision.get("action", 0))
            ho_event = action_triggered > 0 and target_cell is not None

        handover_occurred = False
        if ho_event and not in_handover:
            handover_occurred = True
            in_handover = True
        elif not ho_event:
            in_handover = False

        ping_pong_event = False
        if handover_occurred and serving_cell is not None and target_cell is not None:
            if (
                prev_serving_cell == target_cell
                and serving_cell != prev_serving_cell
                and (now_s - last_ho_time) < 1.0
            ):
                ping_pong_event = True
                total_ping_pongs += 1

        recent_pp_events.append(1 if ping_pong_event else 0)
        recent_ho_events.append(1 if handover_occurred else 0)

        if mode == "ppo":
            if actor_policy is None or rl_agent is None or rl_module is None:
                raise ValueError("rl_agent, rl_module, and actor_policy are required for mode='ppo'")

            if steps_until_policy_update <= 0:
                state = rl_module.build_state_vector(
                    algo1_output=decision,
                    ttt_eff=int(ttt_eff),
                    hys_eff=float(hys_eff),
                    time_since_last_ho=float(time_since_last_ho),
                    recent_rlf_count=int(sum(recent_rlf_events)),
                    recent_pp_count=int(sum(recent_pp_events)),
                    recent_ho_count=int(sum(recent_ho_events)),
                    rsrp_prev=rsrp_prev,
                )
                action = actor_policy.argmax_action(state)
                delta_ttt, delta_hys = rl_agent.decode_action(action)
                ttt_rule, hys_rule = rl_agent.compute_effective_parameters(
                    ttt_rule,
                    hys_rule,
                    delta_ttt,
                    delta_hys,
                )
                steps_until_policy_update = control_interval_steps

            steps_until_policy_update = max(0, steps_until_policy_update - 1)

        time_since_last_ho += 0.1
        rsrp_prev = float(rsrp_current)

        if handover_occurred and target_cell is not None:
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


def _safe_improvement(reference_value, candidate_value):
    if reference_value == 0:
        return 0.0
    return ((reference_value - candidate_value) / reference_value) * 100.0


def _build_improvement_triplet(reference_metrics, candidate_metrics):
    return {
        "HO_reduction_percent": _safe_improvement(reference_metrics["HOs"], candidate_metrics["HOs"]),
        "RLF_reduction_percent": _safe_improvement(reference_metrics["RLFs"], candidate_metrics["RLFs"]),
        "PingPong_reduction_percent": _safe_improvement(
            reference_metrics["PingPongs"], candidate_metrics["PingPongs"]
        ),
    }


def _plot_summary(results, output_png, algo1_ttt, algo1_hys, algo2_ttt, algo2_hys):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    metric_labels = ["Handovers", "RLF", "Ping-Pong"]

    algo1_vals = [results["Algo1"]["HOs"], results["Algo1"]["RLFs"], results["Algo1"]["PingPongs"]]
    algo2_vals = [results["Algo2"]["HOs"], results["Algo2"]["RLFs"], results["Algo2"]["PingPongs"]]
    ppo_vals = [results["PPO"]["HOs"], results["PPO"]["RLFs"], results["PPO"]["PingPongs"]]

    ppo_vs_algo1 = [
        _safe_improvement(results["Algo1"]["HOs"], results["PPO"]["HOs"]),
        _safe_improvement(results["Algo1"]["RLFs"], results["PPO"]["RLFs"]),
        _safe_improvement(results["Algo1"]["PingPongs"], results["PPO"]["PingPongs"]),
    ]
    ppo_vs_algo2 = [
        _safe_improvement(results["Algo2"]["HOs"], results["PPO"]["HOs"]),
        _safe_improvement(results["Algo2"]["RLFs"], results["PPO"]["RLFs"]),
        _safe_improvement(results["Algo2"]["PingPongs"], results["PPO"]["PingPongs"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase-3 Validation: Algo1 vs Algo2 vs PPO", fontsize=13, fontweight="bold")

    x = np.arange(len(metric_labels))
    width = 0.26

    ax = axes[0]
    bars_a1 = ax.bar(
        x - width,
        algo1_vals,
        width,
        label=f"Algo1 (TTT={int(algo1_ttt)}, HYS={algo1_hys:.1f})",
        color="#1f77b4",
    )
    bars_a2 = ax.bar(
        x,
        algo2_vals,
        width,
        label=f"Algo2 (TTT={int(algo2_ttt)}, HYS={algo2_hys:.1f})",
        color="#2ca02c",
    )
    bars_ppo = ax.bar(
        x + width,
        ppo_vals,
        width,
        label="PPO",
        color="#ff7f0e",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Total Count")
    ax.set_title("Absolute Metrics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bar in list(bars_a1) + list(bars_a2) + list(bars_ppo):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(1.0, h * 0.01),
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = axes[1]
    bars_vs_a1 = ax2.bar(x - width / 2.0, ppo_vs_algo1, width, label="PPO vs Algo1", color="#17becf")
    bars_vs_a2 = ax2.bar(x + width / 2.0, ppo_vs_algo2, width, label="PPO vs Algo2", color="#9467bd")

    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Relative Improvement")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    for bars in (bars_vs_a1, bars_vs_a2):
        for bar in bars:
            y = bar.get_height()
            offset = 0.8 if y >= 0 else -1.2
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                y + offset,
                f"{y:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparisons(
    dataset_dir,
    model_dir,
    output_dir,
    algo1_ttt=250,
    algo1_hys=4.5,
    algo2_ttt=160,
    algo2_hys=3.0,
    ppo_init_ttt=160,
    ppo_init_hys=3.0,
    control_interval_steps: int = 5,
):
    csv_files = _collect_dataset_files(dataset_dir)
    if not csv_files:
        raise RuntimeError(f"No supported dataset files found in: {dataset_dir}")

    actor_path = os.path.join(model_dir, "actor.h5")
    critic_path = os.path.join(model_dir, "critic.h5")
    if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
        raise FileNotFoundError(
            f"Trained model not found at {model_dir}. Expected actor.h5 and critic.h5."
        )

    print(f"Loaded {len(csv_files)} Phase-3 dataset files from: {dataset_dir}")
    print(f"Using trained model from: {model_dir}")

    rl_agent = PPOAgent()
    rl_agent.load(model_dir)
    rl_module = RLModule()

    expected_state_dim = _infer_state_dim(rl_module)

    actor_input_shape = getattr(rl_agent.actor, "input_shape", None)
    actor_output_shape = getattr(rl_agent.actor, "output_shape", None)
    if (
        isinstance(actor_input_shape, (list, tuple))
        and actor_input_shape
        and isinstance(actor_input_shape[0], (list, tuple))
    ):
        actor_input_shape = actor_input_shape[0]
    if (
        isinstance(actor_output_shape, (list, tuple))
        and actor_output_shape
        and isinstance(actor_output_shape[0], (list, tuple))
    ):
        actor_output_shape = actor_output_shape[0]

    if not actor_input_shape or actor_input_shape[-1] is None:
        raise RuntimeError("Could not determine actor input shape (state_dim) from loaded model.")
    if not actor_output_shape or actor_output_shape[-1] is None:
        raise RuntimeError("Could not determine actor output shape (action_dim) from loaded model.")

    actor_state_dim = int(actor_input_shape[-1])
    actor_action_dim = int(actor_output_shape[-1])

    if actor_action_dim != 15:
        raise RuntimeError(
            f"Loaded actor action_dim={actor_action_dim} does not match expected 15. "
            "Update Phase-2 PPOAgent.decode_action grid and retrain, or point to a compatible model."
        )
    if actor_state_dim != expected_state_dim:
        raise RuntimeError(
            f"State-dimension mismatch: actor expects {actor_state_dim} inputs, "
            f"but current RLModule.build_state_vector outputs {expected_state_dim}. "
            "Retrain Phase-2 model with the updated state, or use a matching --model-dir."
        )

    actor_policy = _NumpyActorPolicy.from_keras_model(rl_agent.actor)

    summary = {
        "Algo1": {"HOs": 0, "RLFs": 0, "PingPongs": 0},
        "Algo2": {"HOs": 0, "RLFs": 0, "PingPongs": 0},
        "PPO": {"HOs": 0, "RLFs": 0, "PingPongs": 0},
    }
    per_file_rows = []

    for idx, file_path in enumerate(csv_files, start=1):
        df = pd.read_csv(file_path)
        if "ue_id" not in df.columns:
            print(f"Skipping {os.path.basename(file_path)} (missing ue_id column)")
            continue

        algo1_metrics = {"HOs": 0, "RLFs": 0, "PingPongs": 0}
        algo2_metrics = {"HOs": 0, "RLFs": 0, "PingPongs": 0}
        ppo_metrics = {"HOs": 0, "RLFs": 0, "PingPongs": 0}

        for _, ue_df in df.groupby("ue_id"):
            ue_df = ue_df.sort_values("time_s").reset_index(drop=True)

            m_algo1 = _simulate_trajectory(
                ue_df,
                Phase1Algo(),
                mode="algo1",
                static_ttt=algo1_ttt,
                static_hys=algo1_hys,
                control_interval_steps=control_interval_steps,
            )
            m_algo2 = _simulate_trajectory(
                ue_df,
                Algo2Baseline(),
                mode="algo2",
                static_ttt=algo2_ttt,
                static_hys=algo2_hys,
                control_interval_steps=control_interval_steps,
            )
            m_ppo = _simulate_trajectory(
                ue_df,
                Algo2Baseline(),
                mode="ppo",
                static_ttt=algo2_ttt,
                static_hys=algo2_hys,
                ppo_init_ttt=ppo_init_ttt,
                ppo_init_hys=ppo_init_hys,
                rl_agent=rl_agent,
                rl_module=rl_module,
                actor_policy=actor_policy,
                control_interval_steps=control_interval_steps,
            )

            for k in algo1_metrics:
                algo1_metrics[k] += m_algo1[k]
                algo2_metrics[k] += m_algo2[k]
                ppo_metrics[k] += m_ppo[k]

        for k in summary["Algo1"]:
            summary["Algo1"][k] += algo1_metrics[k]
            summary["Algo2"][k] += algo2_metrics[k]
            summary["PPO"][k] += ppo_metrics[k]

        per_file_rows.append(
            {
                "file": os.path.basename(file_path),
                "algorithm": "Algo1",
                "HOs": algo1_metrics["HOs"],
                "RLFs": algo1_metrics["RLFs"],
                "PingPongs": algo1_metrics["PingPongs"],
            }
        )
        per_file_rows.append(
            {
                "file": os.path.basename(file_path),
                "algorithm": "Algo2",
                "HOs": algo2_metrics["HOs"],
                "RLFs": algo2_metrics["RLFs"],
                "PingPongs": algo2_metrics["PingPongs"],
            }
        )
        per_file_rows.append(
            {
                "file": os.path.basename(file_path),
                "algorithm": "PPO",
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
        "ppo_vs_algo1": _build_improvement_triplet(summary["Algo1"], summary["PPO"]),
        "ppo_vs_algo2": _build_improvement_triplet(summary["Algo2"], summary["PPO"]),
        "algo2_vs_algo1": _build_improvement_triplet(summary["Algo1"], summary["Algo2"]),
    }

    summary_payload = {
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "algo1": {
            "ttt_ms": int(algo1_ttt),
            "hys_db": float(algo1_hys),
            "metrics": summary["Algo1"],
        },
        "algo2": {
            "ttt_ms": int(algo2_ttt),
            "hys_db": float(algo2_hys),
            "metrics": summary["Algo2"],
        },
        "ppo": {
            "init_ttt_ms": int(ppo_init_ttt),
            "init_hys_db": float(ppo_init_hys),
            "control_interval_steps": int(control_interval_steps),
            "metrics": summary["PPO"],
        },
        "improvements": improvements,
        "files_evaluated": len(csv_files),
    }

    summary_json = os.path.join(output_dir, "comparison_summary_v3.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    plot_path = os.path.join(output_dir, "comparison_results_v3.png")
    _plot_summary(
        summary,
        plot_path,
        algo1_ttt=algo1_ttt,
        algo1_hys=algo1_hys,
        algo2_ttt=algo2_ttt,
        algo2_hys=algo2_hys,
    )

    print("\n--- PHASE-3 SUMMARY ---")
    print("Algo1:")
    print(f"  HOs:       {summary['Algo1']['HOs']}")
    print(f"  RLFs:      {summary['Algo1']['RLFs']}")
    print(f"  PingPongs: {summary['Algo1']['PingPongs']}")
    print("Algo2:")
    print(f"  HOs:       {summary['Algo2']['HOs']}")
    print(f"  RLFs:      {summary['Algo2']['RLFs']}")
    print(f"  PingPongs: {summary['Algo2']['PingPongs']}")
    print("PPO:")
    print(f"  HOs:       {summary['PPO']['HOs']}")
    print(f"  RLFs:      {summary['PPO']['RLFs']}")
    print(f"  PingPongs: {summary['PPO']['PingPongs']}")

    print("Improvements:")
    print(
        "  PPO vs Algo1: "
        f"HO {improvements['ppo_vs_algo1']['HO_reduction_percent']:+.2f}%, "
        f"RLF {improvements['ppo_vs_algo1']['RLF_reduction_percent']:+.2f}%, "
        f"PP {improvements['ppo_vs_algo1']['PingPong_reduction_percent']:+.2f}%"
    )
    print(
        "  PPO vs Algo2: "
        f"HO {improvements['ppo_vs_algo2']['HO_reduction_percent']:+.2f}%, "
        f"RLF {improvements['ppo_vs_algo2']['RLF_reduction_percent']:+.2f}%, "
        f"PP {improvements['ppo_vs_algo2']['PingPong_reduction_percent']:+.2f}%"
    )

    print(f"\nSaved per-file metrics: {per_file_csv}")
    print(f"Saved summary JSON:     {summary_json}")
    print(f"Saved plot:             {plot_path}")


def _build_arg_parser():
    default_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset")
    default_model_dir = os.path.join(REPO_ROOT, "phase_2", "models_rlf_fix", "final")
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

    parser = argparse.ArgumentParser(
        description="Phase-3 evaluator: compare algo_1, algo_2 baseline, and trained PPO"
    )
    parser.add_argument("--dataset-dir", default=default_dataset_dir, help="Directory containing Phase-3 CSV files")
    parser.add_argument("--model-dir", default=default_model_dir, help="Directory containing actor.h5 and critic.h5")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save Phase-3 outputs")

    parser.add_argument("--algo1-ttt", type=int, default=250, help="Algo1 static TTT (ms)")
    parser.add_argument("--algo1-hys", type=float, default=4.5, help="Algo1 static HYS (dB)")

    parser.add_argument("--algo2-ttt", type=int, default=160, help="Algo2 static TTT (ms)")
    parser.add_argument("--algo2-hys", type=float, default=3.0, help="Algo2 static HYS (dB)")

    parser.add_argument("--ppo-init-ttt", type=int, default=160, help="PPO initial TTT before policy updates (ms)")
    parser.add_argument("--ppo-init-hys", type=float, default=3.0, help="PPO initial HYS before policy updates (dB)")

    parser.add_argument(
        "--control-interval-steps",
        type=int,
        default=5,
        help="Hold PPO-selected TTT/HYS for N ticks before next policy update (default: 5).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_comparisons(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        algo1_ttt=args.algo1_ttt,
        algo1_hys=args.algo1_hys,
        algo2_ttt=args.algo2_ttt,
        algo2_hys=args.algo2_hys,
        ppo_init_ttt=args.ppo_init_ttt,
        ppo_init_hys=args.ppo_init_hys,
        control_interval_steps=args.control_interval_steps,
    )
