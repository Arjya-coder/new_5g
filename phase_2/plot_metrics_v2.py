import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open(r'E:\5g_handover\phase_2\logs\metrics_20260416_083217.json') as f:
    metrics = json.load(f)

episodes    = [m['episode']      for m in metrics]
rewards     = [m['total_reward'] for m in metrics]
rlf_counts  = [m['rlf_count']    for m in metrics]
pp_counts   = [m['pp_count']     for m in metrics]
ho_counts   = [m['ho_count']     for m in metrics]

def smooth(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#c8cee0',
    'xtick.color':      '#8890aa',
    'ytick.color':      '#8890aa',
    'grid.color':       '#2a2d40',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'text.color':       '#e0e4f0',
    'font.family':      'sans-serif',
})

ACCENT   = '#7c6dfa'   # purple-blue – reward
RED      = '#ff5c7c'   # RLF
AMBER    = '#ffb347'   # ping-pong
TEAL     = '#4ecdc4'   # handovers
SMOOTH   = '#ffffff'   # smooth overlay

fig = plt.figure(figsize=(16, 10), facecolor='#0f1117')
fig.suptitle(
    'PPO RL Agent — v2 Training Report  (19-dim state · 150 episodes)',
    fontsize=15, fontweight='bold', color='#e8ecff', y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

# ── 1. Episode Reward ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(episodes, rewards, color=ACCENT, alpha=0.35, width=1.0, label='Raw reward')
sw = 10
ax1.plot(episodes[sw-1:], smooth(rewards, sw), color=SMOOTH, lw=2, label=f'{sw}-ep MA')
ax1.set_title('Episode Reward', fontsize=12, fontweight='bold', pad=8)
ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward')
ax1.legend(fontsize=8, framealpha=0.2)
ax1.grid(True)

# ── 2. RLF Count ────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(episodes, rlf_counts, alpha=0.3, color=RED)
ax2.plot(episodes, rlf_counts, color=RED, lw=1.5, alpha=0.7, label='RLF/ep')
if len(episodes) >= sw:
    ax2.plot(episodes[sw-1:], smooth(rlf_counts, sw), color=SMOOTH, lw=2, label=f'{sw}-ep MA')
ax2.set_title('Radio Link Failures (RLF)', fontsize=12, fontweight='bold', pad=8)
ax2.set_xlabel('Episode'); ax2.set_ylabel('RLF Count')
ax2.legend(fontsize=8, framealpha=0.2)
ax2.grid(True)

# ── 3. Ping-Pong Count ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(episodes, pp_counts, alpha=0.3, color=AMBER)
ax3.plot(episodes, pp_counts, color=AMBER, lw=1.5, alpha=0.7, label='PP/ep')
if len(episodes) >= sw:
    ax3.plot(episodes[sw-1:], smooth(pp_counts, sw), color=SMOOTH, lw=2, label=f'{sw}-ep MA')
ax3.set_title('Ping-Pong Events', fontsize=12, fontweight='bold', pad=8)
ax3.set_xlabel('Episode'); ax3.set_ylabel('Ping-Pong Count')
ax3.legend(fontsize=8, framealpha=0.2)
ax3.grid(True)

# ── 4. Handover Count (proxy for exploration) ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(episodes, ho_counts, alpha=0.25, color=TEAL)
ax4.plot(episodes, ho_counts, color=TEAL, lw=1.5, alpha=0.7, label='HO/ep')
if len(episodes) >= sw:
    ax4.plot(episodes[sw-1:], smooth(ho_counts, sw), color=SMOOTH, lw=2, label=f'{sw}-ep MA')
ax4.set_title('Handover Count  (exploration proxy)', fontsize=12, fontweight='bold', pad=8)
ax4.set_xlabel('Episode'); ax4.set_ylabel('Handovers')
ax4.legend(fontsize=8, framealpha=0.2)
ax4.grid(True)

out = r'E:\5g_handover\phase_2\logs\learning_curves_v2.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
