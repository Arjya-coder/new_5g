"""
train_dqn.py - Double DQN Training Loop for 5G Handover Control
================================================================
Folders (all relative to phase_2_dqn/):
  model/best/          <- best checkpoint by val J-score
  model/ep{N}/         <- periodic snapshots every 50 episodes
  model/final/         <- weights at end of training
  plots/               <- all diagnostic PNGs
  logs/                <- JSON metric logs

Dataset split (by seed in filename):
  Train : seeds 1-12
  Val   : seeds 13-15  (checkpoint selection)
  Test  : seeds 16-20  (final report only)

Usage
-----
  cd phase_2_dqn
  python train_dqn.py
  python train_dqn.py --episodes 500 --dataset_dir E:\\5g_handover\\dataset
"""

import os, sys, glob, json, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from env_dqn   import OfflineNs3Env, STATE_DIM, ACTION_DIM, decode_action
from dqn_agent import DoubleDQNAgent

# ── Fixed output paths (relative to this script) ────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(_HERE, "model")
PLOTS_DIR   = os.path.join(_HERE, "plots")
LOGS_DIR    = os.path.join(_HERE, "logs")
DATASET_DIR = r"E:\5g_handover\dataset"

# ── Seed split ───────────────────────────────────────────────────────────────
SEED_TRAIN = set(range(1,  13))
SEED_VAL   = set(range(13, 16))
SEED_TEST  = set(range(16, 21))


# ============================================================================
# DATASET SPLIT
# ============================================================================
def _parse_seed(path: str):
    m = re.search(r'seed(\d+)', os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def split_files(dataset_dir: str):
    files = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
    if not files:
        raise RuntimeError(
            f"No *_tick.csv files in '{dataset_dir}'.\n"
            "Generate the dataset first using dataset.cpp."
        )
    train, val, test, unsplit = [], [], [], []
    for f in files:
        s = _parse_seed(f)
        if   s in SEED_TRAIN: train.append(f)
        elif s in SEED_VAL:   val.append(f)
        elif s in SEED_TEST:  test.append(f)
        else:                  unsplit.append(f)
    train.extend(unsplit)   # files with no parseable seed go to train
    return train, val, test


# ============================================================================
# GREEDY EVALUATION
# ============================================================================
def evaluate_greedy(agent, env: OfflineNs3Env, n_episodes: int = 30) -> dict:
    """Greedy rollout on a pre-loaded env. Returns RLF/PP/HO totals and J-score."""
    rlf = pp = ho = steps = 0
    total_r = 0.0
    n = min(n_episodes, len(env._ue_traces))

    for _ in range(n):
        state = env.reset()
        done  = False
        while not done:
            action = agent.select_action(state, greedy=True)
            state, r, done, info = env.step(action)
            rlf += int(info['rlf'])
            pp  += int(info['pp'])
            ho  += int(info['ho'])
            total_r += r
            steps   += 1

    return {
        'rlf': rlf, 'pp': pp, 'ho': ho,
        'j_score':    1.0 * rlf + 0.2 * pp + 0.05 * ho,
        'mean_reward': total_r / max(steps, 1),
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train(args):
    # ── Directory setup ──────────────────────────────────────────────────────
    for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_dir = os.path.join(MODEL_DIR, "best")
    fin_dir  = os.path.join(MODEL_DIR, "final")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\nDataset dir : {args.dataset_dir}")
    train_files, val_files, test_files = split_files(args.dataset_dir)
    print(f"  Train : {len(train_files)} files")
    print(f"  Val   : {len(val_files)} files")
    print(f"  Test  : {len(test_files)} files\n")

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = DoubleDQNAgent(
        state_dim        = STATE_DIM,
        action_dim       = ACTION_DIM,
        lr               = args.lr,
        gamma            = 0.99,
        tau              = 0.005,
        buffer_size      = args.buffer_size,
        batch_size       = args.batch_size,
        n_step           = 5,
        eps_start        = 1.0,
        eps_end          = 0.05,
        eps_decay_steps  = int(args.episodes * args.rollout * 0.5),
    )
    # ── Environments (pre-loaded once) ───────────────────────────────────────
    print("\nCreating training environment...")
    env = OfflineNs3Env(train_files, max_steps=args.rollout)

    val_env = None
    if val_files:
        print("Creating validation environment...")
        val_env = OfflineNs3Env(val_files, max_steps=1024)

    test_env = None
    if test_files:
        print("Creating test environment...")
        test_env = OfflineNs3Env(test_files, max_steps=1024)

    print("=" * 70)
    print(f"DOUBLE DQN  |  {args.episodes} episodes  |  state={STATE_DIM}  actions={ACTION_DIM}")
    print(f"PER buffer={args.buffer_size:,}  batch={args.batch_size}  n-step=5  lr={args.lr}")
    print("=" * 70 + "\n")

    ep_metrics, val_metrics = [], []
    best_j = float('inf')

    # ── Episode loop ─────────────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done  = False
        ep_r = ep_rlf = ep_pp = ep_ho = ep_steps = 0
        ep_loss = []
        ttt_f, hys_f = 160, 3.0

        while not done:
            a = agent.select_action(state)
            ns, r, done, info = env.step(a)
            agent.store(state, a, r, ns, done)

            loss = agent.train_step()
            if loss is not None:
                ep_loss.append(loss)

            agent.decay_epsilon()
            ep_r   += r
            ep_rlf += int(info['rlf'])
            ep_pp  += int(info['pp'])
            ep_ho  += int(info['ho'])
            ep_steps += 1
            ttt_f, hys_f = decode_action(a)
            state = ns

        ml = float(np.mean(ep_loss)) if ep_loss else 0.0
        ep_metrics.append({
            'episode': ep, 'reward': round(ep_r, 3),
            'rlf': ep_rlf, 'pp': ep_pp, 'ho': ep_ho,
            'steps': ep_steps, 'loss': round(ml, 6),
            'epsilon': round(agent.eps, 4),
            'ttt_final': ttt_f, 'hys_final': hys_f,
            'buffer_size': len(agent.buffer),
        })

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | R:{ep_r:7.2f} | RLF:{ep_rlf:2d} PP:{ep_pp:3d} "
                  f"HO:{ep_ho:3d} | TTT:{ttt_f} HYS:{hys_f:.1f} | "
                  f"eps:{agent.eps:.3f} loss:{ml:.5f} buf:{len(agent.buffer):,}")

        # Validation
        if ep % 10 == 0 and val_env:
            vm = evaluate_greedy(agent, val_env, n_episodes=20)
            vm['episode'] = ep
            val_metrics.append(vm)
            print(f"       [VAL] J={vm['j_score']:.1f} "
                  f"RLF={vm['rlf']} PP={vm['pp']} HO={vm['ho']}")
            if vm['j_score'] < best_j:
                best_j = vm['j_score']
                agent.save(best_dir)
                print(f"       [BEST] J={best_j:.1f} saved -> model/best/")

        # Periodic snapshot
        if ep % 50 == 0:
            snap = os.path.join(MODEL_DIR, f"ep{ep}")
            agent.save(snap)
            print(f"       [SNAP] Saved -> model/ep{ep}/")

    # ── Save final ────────────────────────────────────────────────────────────
    agent.save(fin_dir)
    json.dump(ep_metrics,  open(os.path.join(LOGS_DIR, f"train_{ts}.json"),  'w'), indent=2)
    json.dump(val_metrics, open(os.path.join(LOGS_DIR, f"val_{ts}.json"),    'w'), indent=2)

    print(f"\n{'='*70}")
    print(f"DONE  |  best val J={best_j:.1f}  |  model/best/  model/final/")
    print(f"{'='*70}\n")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_m = None
    if test_env and os.path.exists(os.path.join(best_dir, "online.weights.h5")):
        print("TEST evaluation on held-out seeds 16-20...")
        agent.load(best_dir)
        test_m = evaluate_greedy(agent, test_env, n_episodes=50)
        print(f"TEST  J={test_m['j_score']:.1f}  "
              f"RLF={test_m['rlf']}  PP={test_m['pp']}  HO={test_m['ho']}")
        json.dump(test_m, open(os.path.join(LOGS_DIR, f"test_{ts}.json"), 'w'), indent=2)

    # ── Generate all plots ────────────────────────────────────────────────────
    plot_training_curves(ep_metrics, val_metrics, ts)
    plot_event_breakdown(ep_metrics, val_metrics, ts)
    plot_parameter_evolution(ep_metrics, ts)
    if test_m:
        plot_test_comparison(test_m, ts)

    return ep_metrics, val_metrics


# ============================================================================
# PLOT 1 — Training & validation curves (reward + loss + epsilon)
# ============================================================================
def plot_training_curves(ep_metrics, val_metrics, ts):
    BG, PANEL, TEXT = '#0d0f1a', '#13162a', '#d0d6f0'
    _apply_dark_theme(BG, PANEL, TEXT)

    eps    = [m['episode'] for m in ep_metrics]
    reward = [m['reward']  for m in ep_metrics]
    loss   = [m['loss']    for m in ep_metrics]
    epsilon= [m['epsilon'] for m in ep_metrics]
    buf    = [m['buffer_size'] for m in ep_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle('Plot 1 — Training Curves\nDouble DQN | 5G Handover',
                 fontsize=13, fontweight='bold', color='#e8eeff', y=0.98)

    _line(axes[0,0], eps, reward,  'Episode Reward',       'Total Reward',    '#4ecdc4')
    _line(axes[0,1], eps, loss,    'TD Loss (MSE)',         'Loss',            '#c3b1e1')
    _line(axes[1,0], eps, epsilon, 'Epsilon Decay',         'Epsilon',         '#87ceeb')
    _line(axes[1,1], eps, buf,     'Replay Buffer Growth',  'Transitions',     '#ffd166')

    # Overlay val J on reward plot
    if val_metrics:
        v_eps = [m['episode']  for m in val_metrics]
        v_j   = [m['j_score']  for m in val_metrics]
        ax2   = axes[0,0].twinx()
        ax2.plot(v_eps, v_j, 'o--', color='#ff6b9d', ms=5, lw=1.5)
        ax2.set_ylabel('Val J-score (lower=better)', color='#ff6b9d', fontsize=9)
        ax2.tick_params(axis='y', colors='#ff6b9d')

    plt.tight_layout()
    _save(fig, f"plot1_training_curves_{ts}.png", BG)


# ============================================================================
# PLOT 2 — Event counts (RLF / PP / HO per episode)
# ============================================================================
def plot_event_breakdown(ep_metrics, val_metrics, ts):
    BG, PANEL, TEXT = '#0d0f1a', '#13162a', '#d0d6f0'
    _apply_dark_theme(BG, PANEL, TEXT)

    eps = [m['episode'] for m in ep_metrics]
    rlf = [m['rlf'] for m in ep_metrics]
    pp  = [m['pp']  for m in ep_metrics]
    ho  = [m['ho']  for m in ep_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle('Plot 2 — Event Counts per Episode\n'
                 'What the agent is doing wrong / right',
                 fontsize=13, fontweight='bold', color='#e8eeff')

    _line(axes[0], eps, rlf, 'RLF Events / Episode',       'Count', '#ff4f6d')
    _line(axes[1], eps, pp,  'Ping-Pong Events / Episode', 'Count', '#ffb347')
    _line(axes[2], eps, ho,  'Handovers / Episode',        'Count', '#a8e6cf')

    # Annotate val milestones
    if val_metrics:
        for ax, key, c in zip(axes, ['rlf','pp','ho'],
                              ['#ff4f6d','#ffb347','#a8e6cf']):
            for vm in val_metrics:
                ax.axvline(vm['episode'], color=c, alpha=0.15, lw=1)

    plt.tight_layout()
    _save(fig, f"plot2_event_counts_{ts}.png", BG)


# ============================================================================
# PLOT 3 — TTT / HYS parameter evolution
# ============================================================================
def plot_parameter_evolution(ep_metrics, ts):
    BG, PANEL, TEXT = '#0d0f1a', '#13162a', '#d0d6f0'
    _apply_dark_theme(BG, PANEL, TEXT)

    eps = [m['episode']   for m in ep_metrics]
    ttt = [m['ttt_final'] for m in ep_metrics]
    hys = [m['hys_final'] for m in ep_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle('Plot 3 — Parameter Evolution (TTT / HYS)\n'
                 'Is the RL agent actively adjusting parameters?',
                 fontsize=13, fontweight='bold', color='#e8eeff')

    # TTT scatter + histogram
    axes[0].scatter(eps, ttt, color='#4ecdc4', alpha=0.3, s=6)
    axes[0].set_title('TTT Final Value per Episode', fontweight='bold')
    axes[0].set_ylabel('TTT (ms)'); axes[0].set_xlabel('Episode')
    axes[0].set_yticks([100, 160, 200, 240, 320])
    axes[0].grid(True)

    # HYS scatter
    axes[1].scatter(eps, hys, color='#ffd166', alpha=0.3, s=6)
    axes[1].set_title('HYS Final Value per Episode', fontweight='bold')
    axes[1].set_ylabel('HYS (dB)'); axes[1].set_xlabel('Episode')
    axes[1].grid(True)

    plt.tight_layout()
    _save(fig, f"plot3_parameter_evolution_{ts}.png", BG)


# ============================================================================
# PLOT 4 — Test comparison: DQN vs Baseline
# ============================================================================
def plot_test_comparison(test_m: dict, ts):
    # Baseline numbers from Phase-3 comparison_results_v3.png
    BASELINE = {'RLF': 1229, 'PP': 73, 'HO': 2597}
    DQN      = {'RLF': test_m['rlf'], 'PP': test_m['pp'], 'HO': test_m['ho']}

    BG, PANEL, TEXT = '#0d0f1a', '#13162a', '#d0d6f0'
    _apply_dark_theme(BG, PANEL, TEXT)

    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    metrics = ['RLF', 'Ping-Pong', 'Handovers']
    b_vals  = [BASELINE['RLF'], BASELINE['PP'], BASELINE['HO']]
    d_vals  = [DQN['RLF'],      DQN['PP'],      DQN['HO']]

    x = np.arange(len(metrics))
    w = 0.35
    bars_b = ax1.bar(x - w/2, b_vals, w, label='Baseline (static)',
                     color='#546e9b', alpha=0.85, edgecolor='#7090cc', linewidth=1.2)
    bars_d = ax1.bar(x + w/2, d_vals, w, label='Double DQN (ours)',
                     color='#4ecdc4', alpha=0.85, edgecolor='#6eeded', linewidth=1.2)

    for bar in bars_b:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 fontsize=9, color=TEXT)
    for bar in bars_d:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 fontsize=9, color='#4ecdc4', fontweight='bold')

    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=11)
    ax1.set_title('Plot 4 — DQN vs Baseline: Event Counts\n(Test Set: seeds 16-20)',
                  fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count'); ax1.legend(fontsize=10); ax1.grid(True, axis='y')

    # Improvement % panel
    improvements = []
    labels_imp   = []
    colors_imp   = []
    for k, bv, dv in zip(['RLF','PP','HO'], b_vals, d_vals):
        pct = (bv - dv) / max(bv, 1) * 100
        improvements.append(pct)
        labels_imp.append(k)
        colors_imp.append('#4ecdc4' if pct >= 0 else '#ff4f6d')

    bars_i = ax2.barh(labels_imp, improvements, color=colors_imp,
                      alpha=0.85, edgecolor='#ffffff22', linewidth=0.8)
    for bar, val in zip(bars_i, improvements):
        xpos = bar.get_width() + (1 if val >= 0 else -1)
        ax2.text(xpos, bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center', fontsize=10,
                 color='#4ecdc4' if val >= 0 else '#ff4f6d', fontweight='bold')

    ax2.axvline(0, color=TEXT, lw=1.2, alpha=0.5)
    ax2.set_title('Improvement over Baseline\n(+ = better)', fontweight='bold')
    ax2.set_xlabel('% Change (positive = improvement)')
    ax2.grid(True, axis='x')

    plt.tight_layout()
    _save(fig, f"plot4_test_comparison_{ts}.png", BG)
    print(f"Test comparison plot saved -> plots/plot4_test_comparison_{ts}.png")


# ============================================================================
# SHARED HELPERS
# ============================================================================
def _apply_dark_theme(bg, panel, text):
    plt.rcParams.update({
        'figure.facecolor': bg, 'axes.facecolor': panel,
        'axes.edgecolor': '#1f2340', 'axes.labelcolor': text,
        'xtick.color': text, 'ytick.color': text, 'text.color': text,
        'legend.facecolor': panel, 'legend.edgecolor': '#2a2f50',
        'grid.color': '#1f2340', 'grid.linestyle': '--', 'grid.alpha': 0.5,
    })


def _line(ax, x, y, title, ylabel, color, w=10):
    ax.plot(x, y, color=color, alpha=0.25, lw=0.8)
    if len(y) >= w:
        ma = np.convolve(y, np.ones(w)/w, mode='valid')
        ax.plot(x[w-1:], ma, color=color, lw=2.2, label=f'MA({w})')
        ax.legend(fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_xlabel('Episode')
    ax.grid(True)


def _save(fig, name, bg):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close(fig)
    print(f"Plot saved -> plots/{name}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Double DQN — 5G Handover')
    parser.add_argument('--dataset_dir', default=DATASET_DIR)
    parser.add_argument('--episodes',    type=int,   default=300)
    parser.add_argument('--rollout',     type=int,   default=1024)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--buffer_size', type=int,   default=100_000)
    args = parser.parse_args()

    os.chdir(_HERE)   # ensure relative imports work
    train(args)
