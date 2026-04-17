"""
5 Diagnostic Plots for PPO v2 Training Analysis
================================================
Plot 1: Reward Curve       — does learning work at all?
Plot 2: Event Counts       — what is going wrong?
Plot 3: Parameter Evolution— is RL actively changing anything?
Plot 4: Zone Parameters    — did RL learn intelligently?
Plot 5: Policy Variance    — is the policy stable?
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Load data ────────────────────────────────────────────────────────────────
METRICS_FILE = r'E:\5g_handover\phase_2\logs\metrics_20260416_083217.json'
PARAMS_FILE  = r'E:\5g_handover\phase_2\logs\param_history_20260416_083217.json'
OUT_FILE     = r'E:\5g_handover\phase_2\logs\diagnostic_plots_v2.png'

with open(METRICS_FILE) as f:
    metrics = json.load(f)
with open(PARAMS_FILE) as f:
    param_history = json.load(f)

eps      = np.array([m['episode']        for m in metrics])
rewards  = np.array([m['total_reward']   for m in metrics])
rlfs     = np.array([m['rlf_count']      for m in metrics])
pps      = np.array([m['pp_count']       for m in metrics])
hos      = np.array([m['ho_count']       for m in metrics])
entropy  = np.array([m['entropy_coef']   for m in metrics])
ttt_fin  = np.array([m['ttt_final']      for m in metrics])
hys_fin  = np.array([m['hys_final']      for m in metrics])

# ── Smooth helper ────────────────────────────────────────────────────────────
def smooth(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

def seps(arr, w=10):
    return eps[w-1:]

# ── Global style ─────────────────────────────────────────────────────────────
BG    = '#0d0f1a'
PANEL = '#13162a'
GRID  = '#1f2340'
TEXT  = '#d0d6f0'
MUTE  = '#7a82a8'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': MUTE, 'ytick.color': MUTE,
    'grid.color': GRID, 'grid.linestyle': '--', 'grid.alpha': 0.6,
    'text.color': TEXT, 'font.family': 'sans-serif', 'font.size': 9,
})

C_REWARD = '#7c6dfa'
C_RLF    = '#ff4f6d'
C_PP     = '#ffb347'
C_HO     = '#4ecdc4'
C_TTT    = '#a8d8ea'
C_HYS    = '#f7a6c0'
C_SMOOTH = '#ffffff'
C_FILL   = '#6c63ff'

fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle(
    'PPO v2 — 5-Panel Diagnostic Report  (19-dim state | 150 episodes | seed 999 datasets)',
    fontsize=14, fontweight='bold', color='#e8eeff', y=0.985
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.30,
                       left=0.06, right=0.97, top=0.96, bottom=0.05)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Reward Curve
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
W = 10
ax1.fill_between(eps, rewards, alpha=0.18, color=C_FILL)
ax1.plot(eps, rewards, color=C_REWARD, lw=1, alpha=0.5, label='Raw reward')
ax1.plot(seps(rewards,W), smooth(rewards,W), color=C_SMOOTH, lw=2.2, label=f'{W}-ep moving avg')

# Annotate peak
peak_ep = eps[np.argmax(rewards)]
peak_r  = rewards.max()
ax1.annotate(f'Peak: {peak_r:.1f}\n(ep {peak_ep})',
             xy=(peak_ep, peak_r), xytext=(peak_ep+10, peak_r-8),
             arrowprops=dict(arrowstyle='->', color=C_SMOOTH, lw=1.2),
             fontsize=8, color=C_SMOOTH)

# Mean line
ax1.axhline(rewards.mean(), color=C_REWARD, lw=1.2, ls=':', alpha=0.7, label=f'Mean {rewards.mean():.1f}')
ax1.set_title('Plot 1 — Reward Curve  (Does learning work?)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward')
ax1.legend(fontsize=8, framealpha=0.15, loc='lower right')
ax1.grid(True)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Event Counts
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()

ax2.bar(eps, rlfs, color=C_RLF, alpha=0.7, width=1.0, label='RLF')
ax2.bar(eps, pps,  color=C_PP,  alpha=0.6, width=1.0, bottom=rlfs, label='Ping-Pong')
ax2_twin.plot(eps, hos, color=C_HO, lw=1.5, alpha=0.8, label='Handovers (R)')
ax2_twin.plot(seps(hos,W), smooth(hos,W), color=C_SMOOTH, lw=2, ls='--', label='HO MA (R)')

ax2.set_title('Plot 2 — Event Counts  (What is going wrong?)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
ax2.set_xlabel('Episode')
ax2.set_ylabel('RLF + PP Count', color=C_RLF)
ax2_twin.set_ylabel('Handover Count', color=C_HO)
ax2_twin.tick_params(axis='y', labelcolor=C_HO)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.15, loc='upper right')
ax2.grid(True, axis='y')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Parameter Evolution
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3_twin = ax3.twinx()

ax3.step(eps, ttt_fin, color=C_TTT, lw=1.5, where='mid', alpha=0.8, label='TTT_final (ms)')
ax3.plot(seps(ttt_fin,W), smooth(ttt_fin,W), color=C_SMOOTH, lw=2, label='TTT MA')
ax3_twin.step(eps, hys_fin, color=C_HYS, lw=1.5, where='mid', alpha=0.8, label='HYS_final (dB, R)')
ax3_twin.plot(seps(hys_fin,W), smooth(hys_fin,W), color=C_HYS, lw=2, ls='--', alpha=0.5, label='HYS MA (R)')

ax3.set_yticks([100, 160, 240, 320])
ax3_twin.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

ax3.set_title('Plot 3 — Parameter Evolution  (Is RL actively choosing?)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
ax3.set_xlabel('Episode')
ax3.set_ylabel('TTT (ms)', color=C_TTT)
ax3_twin.set_ylabel('HYS (dB)', color=C_HYS)
ax3_twin.tick_params(axis='y', labelcolor=C_HYS)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.15)
ax3.grid(True, axis='y')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Zone Parameters (from param_history)
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])

zone_ttt = defaultdict(list)
zone_hys = defaultdict(list)
for entry in param_history:
    z = entry['zone']
    zone_ttt[z].append(entry['ttt_eff'])
    zone_hys[z].append(entry['hys_eff'])

zone_order = sorted(zone_ttt.keys())
x = np.arange(len(zone_order))
w = 0.35

avg_ttt = [np.mean(zone_ttt[z]) for z in zone_order]
avg_hys = [np.mean(zone_hys[z]) for z in zone_order]
# Normalize HYS to TTT scale for dual-axis visibility
ax4_twin = ax4.twinx()

bars1 = ax4.bar(x - w/2, avg_ttt, width=w, color=C_TTT, alpha=0.8, label='Avg TTT (ms)')
bars2 = ax4_twin.bar(x + w/2, avg_hys, width=w, color=C_HYS, alpha=0.8, label='Avg HYS (dB, R)')

for bar, val in zip(bars1, avg_ttt):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{val:.0f}',
             ha='center', va='bottom', fontsize=8, color=C_TTT)
for bar, val in zip(bars2, avg_hys):
    ax4_twin.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val:.1f}',
                  ha='center', va='bottom', fontsize=8, color=C_HYS)

ax4.set_xticks(x); ax4.set_xticklabels([z.replace('_',' ') for z in zone_order], fontsize=8)
ax4.set_title('Plot 4 — Zone Parameters  (Did RL learn intelligently?)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
ax4.set_ylabel('Avg TTT (ms)', color=C_TTT)
ax4_twin.set_ylabel('Avg HYS (dB)', color=C_HYS)
ax4_twin.tick_params(axis='y', labelcolor=C_HYS)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.15)
ax4.grid(True, axis='y')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Policy Variance (rolling std of reward)
# ═══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, :])   # full-width bottom

W_var = 15
rolling_mean = smooth(rewards, W_var)
rolling_std  = np.array([rewards[max(0,i-W_var):i+1].std() for i in range(len(rewards))])
rolling_std_sm = smooth(rolling_std, W_var)
ep_ma = seps(rewards, W_var)

ax5.fill_between(ep_ma,
                 rolling_mean - rolling_std_sm[:len(rolling_mean)],
                 rolling_mean + rolling_std_sm[:len(rolling_mean)],
                 alpha=0.20, color=C_FILL, label='+/- 1 std band')
ax5.plot(ep_ma, rolling_mean,          color=C_SMOOTH, lw=2.2, label=f'{W_var}-ep mean')
ax5.plot(eps,   rolling_std,           color=C_REWARD, lw=1.2, alpha=0.5, label='Rolling std (raw)')
ax5.plot(ep_ma, rolling_std_sm[:len(ep_ma)], color=C_REWARD, lw=2, ls='--', label=f'{W_var}-ep std MA')

# Annotate stability zone
stable_threshold = rolling_std.mean()
ax5.axhline(stable_threshold, color=C_PP, lw=1, ls=':', alpha=0.7,
            label=f'Mean std = {stable_threshold:.1f}')

ax5.set_title('Plot 5 — Policy Variance  (Is the policy stable?)', fontsize=11, fontweight='bold', pad=8, color=TEXT)
ax5.set_xlabel('Episode'); ax5.set_ylabel('Reward')
ax5.legend(fontsize=9, framealpha=0.15, loc='upper right', ncol=3)
ax5.grid(True)

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {OUT_FILE}")
