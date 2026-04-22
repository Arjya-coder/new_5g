from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_metrics_file(log_dir: str) -> str:
    candidates = glob.glob(os.path.join(log_dir, "metrics_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No training metrics file found in {log_dir} (expected metrics_*.json)")
    return max(candidates, key=os.path.getmtime)


def _moving_average(values: Sequence[float], window: int = 10) -> List[float]:
    window = max(1, int(window))
    out: List[float] = []
    acc = 0.0
    q: List[float] = []

    for v in values:
        q.append(float(v))
        acc += float(v)
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return (sum(vals) / len(vals)) if vals else 0.0


def _plot_training_curves(metrics_rows: List[dict], out_dir: str) -> str:
    episodes = [int(r.get("episode", i + 1)) for i, r in enumerate(metrics_rows)]
    rewards = [float(r.get("total_reward", 0.0)) for r in metrics_rows]
    reward_step = [float(r.get("avg_reward_per_step", 0.0)) for r in metrics_rows]

    rlf = [float(r.get("rlf_count", 0.0)) for r in metrics_rows]
    pp = [float(r.get("pp_count", 0.0)) for r in metrics_rows]
    ho = [float(r.get("ho_count", 0.0)) for r in metrics_rows]

    ttt = [float(r.get("ttt_final", 0.0)) for r in metrics_rows]
    hys = [float(r.get("hys_final", 0.0)) for r in metrics_rows]

    rewards_ma = _moving_average(rewards, window=10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    ax0 = axes[0]
    ax0.plot(episodes, rewards, label="Total reward", color="#1f77b4", linewidth=1.3, alpha=0.85)
    ax0.plot(episodes, rewards_ma, label="Reward MA(10)", color="#d62728", linewidth=2.0)
    ax0.plot(episodes, reward_step, label="Avg reward/step", color="#2ca02c", linewidth=1.3, alpha=0.85)
    ax0.set_title("Training Reward Curves")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Reward")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(episodes, rlf, label="RLF count", color="#8c564b", linewidth=1.5)
    ax1.plot(episodes, pp, label="Ping-pong count", color="#ff7f0e", linewidth=1.5)
    ax1.plot(episodes, ho, label="Handover count", color="#9467bd", linewidth=1.5)
    ax1.set_title("Training Event Counts Per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    ax2 = axes[2]
    ax2.plot(episodes, ttt, label="Final TTT (ms)", color="#17becf", linewidth=1.6)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("TTT (ms)")
    ax2.grid(True, linestyle="--", alpha=0.35)

    ax2b = ax2.twinx()
    ax2b.plot(episodes, hys, label="Final HYS (dB)", color="#bcbd22", linewidth=1.6)
    ax2b.set_ylabel("HYS (dB)")

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax2.set_title("Training Control Parameters Per Episode")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_eval_curves(val_rows: List[dict], test_rows: List[dict], out_dir: str) -> str:
    val_ep = [int(r.get("episode", i + 1)) for i, r in enumerate(val_rows)]
    test_ep = [int(r.get("episode", i + 1)) for i, r in enumerate(test_rows)]

    val_reward = [float(r.get("total_reward", 0.0)) for r in val_rows]
    test_reward = [float(r.get("total_reward", 0.0)) for r in test_rows]

    val_rlf = [float(r.get("rlf_count", 0.0)) for r in val_rows]
    test_rlf = [float(r.get("rlf_count", 0.0)) for r in test_rows]

    val_pp = [float(r.get("pp_count", 0.0)) for r in val_rows]
    test_pp = [float(r.get("pp_count", 0.0)) for r in test_rows]

    val_ho = [float(r.get("ho_count", 0.0)) for r in val_rows]
    test_ho = [float(r.get("ho_count", 0.0)) for r in test_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax0 = axes[0]
    ax0.plot(val_ep, val_reward, marker="o", label="Validation reward", color="#1f77b4")
    ax0.plot(test_ep, test_reward, marker="o", label="Test reward", color="#ff7f0e")
    ax0.set_title("Validation vs Test Reward")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Total reward")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend()

    ax1 = axes[1]
    metric_names = ["RLF", "PingPong", "HO"]
    val_means = [_safe_mean(val_rlf), _safe_mean(val_pp), _safe_mean(val_ho)]
    test_means = [_safe_mean(test_rlf), _safe_mean(test_pp), _safe_mean(test_ho)]

    x = list(range(len(metric_names)))
    width = 0.35
    ax1.bar([xi - width / 2 for xi in x], val_means, width=width, label="Validation", color="#2ca02c")
    ax1.bar([xi + width / 2 for xi in x], test_means, width=width, label="Test", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_title("Average Event Counts (Validation vs Test)")
    ax1.set_ylabel("Average count")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "validation_test_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_summary(val_rows: List[dict], test_rows: List[dict], out_dir: str) -> str:
    summary = {
        "validation": {
            "avg_total_reward": _safe_mean(float(r.get("total_reward", 0.0)) for r in val_rows),
            "avg_rlf_count": _safe_mean(float(r.get("rlf_count", 0.0)) for r in val_rows),
            "avg_pp_count": _safe_mean(float(r.get("pp_count", 0.0)) for r in val_rows),
            "avg_ho_count": _safe_mean(float(r.get("ho_count", 0.0)) for r in val_rows),
            "episodes": len(val_rows),
        },
        "test": {
            "avg_total_reward": _safe_mean(float(r.get("total_reward", 0.0)) for r in test_rows),
            "avg_rlf_count": _safe_mean(float(r.get("rlf_count", 0.0)) for r in test_rows),
            "avg_pp_count": _safe_mean(float(r.get("pp_count", 0.0)) for r in test_rows),
            "avg_ho_count": _safe_mean(float(r.get("ho_count", 0.0)) for r in test_rows),
            "episodes": len(test_rows),
        },
    }

    out_path = os.path.join(out_dir, "val_test_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training/validation/testing plots from PPO JSON logs")
    parser.add_argument("--log-dir", default=os.path.join(os.path.dirname(__file__), "logs_rlf_fix"))
    parser.add_argument("--metrics-file", default="", help="Optional explicit metrics_*.json path")
    parser.add_argument("--val-file", default="", help="Optional explicit validation JSON path")
    parser.add_argument("--test-file", default="", help="Optional explicit test JSON path")
    parser.add_argument("--out-dir", default="", help="Optional explicit output plot directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_dir = os.path.abspath(args.log_dir)
    metrics_file = os.path.abspath(args.metrics_file) if args.metrics_file else _latest_metrics_file(log_dir)
    val_file = os.path.abspath(args.val_file) if args.val_file else os.path.join(log_dir, "eval_val_latest.json")
    test_file = os.path.abspath(args.test_file) if args.test_file else os.path.join(log_dir, "eval_test_latest.json")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(log_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)

    metrics_rows = _read_json(metrics_file)
    val_rows = _read_json(val_file)
    test_rows = _read_json(test_file)

    if not isinstance(metrics_rows, list) or not metrics_rows:
        raise RuntimeError(f"Training metrics file is empty or invalid: {metrics_file}")
    if not isinstance(val_rows, list) or not val_rows:
        raise RuntimeError(f"Validation file is empty or invalid: {val_file}")
    if not isinstance(test_rows, list) or not test_rows:
        raise RuntimeError(f"Test file is empty or invalid: {test_file}")

    p1 = _plot_training_curves(metrics_rows, out_dir)
    p2 = _plot_eval_curves(val_rows, test_rows, out_dir)
    p3 = _write_summary(val_rows, test_rows, out_dir)

    print("Generated files:")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
