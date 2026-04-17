"""
Auto-run after training: finds latest metrics + param_history, produces 5 plots.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Locate latest run files ──────────────────────────────────────────────────
LOG_DIR = r'E:\5g_handover\phase_2\logs'
OUT     = os.path.join(LOG_DIR, 'diagnostic_v3.png')

metrics_file  = max(glob.glob(os.path.join(LOG_DIR, 'metrics_*.json')),      key=os.path.getctime)
params_file   = max(glob.glob(os.path.join(LOG_DIR, 'param_history_*.json')),key=os.path.getctime)

print(f"Metrics : {os.path.basename(metrics_file)}")
print(f"Params  : {os.path.basename(params_file)}")

with open(metrics_file) as f:  metrics = json.load(f)
with open(params_file)  as f:  ph      = json.load(f)

eps     = np.array([m['episode']      for m in metrics])
rewards = np.array([m['total_reward'] for m in metrics])
rlfs    = np.array([m['rlf_count']    for m in metrics])
pps     = np.array([m['pp_count']     for m in metrics])
hos     = np.array([m['ho_count']     for m in metrics])
ttt_fin = np.array([m['ttt_final']    for m in metrics])
hys_fin = np.array([m['hys_final']    for m in metrics])

def smooth(a, w=10): return np.convolve(a, np.ones(w)/w, mode='valid')
def ep_sm(w=10):     return eps[w-1:]

# ── Global dark style ─────────────────────────────────────────────────────────
BG='#0d0f1a'; PANEL='#13162a'; GRID='#1f2340'; TEXT='#d0d6f0'; MUTE='#7a82a8'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':PANEL,
    'axes.edgecolor':GRID,'axes.labelcolor':TEXT,'xtick.color':MUTE,
    'ytick.color':MUTE,'grid.color':GRID,'grid.linestyle':'--','grid.alpha':.6,
    'text.color':TEXT,'font.family':'sans-serif','font.size':9})

CR='#7c6dfa'; CRLF='#ff4f6d'; CPP='#ffb347'; CHO='#4ecdc4'
CTTT='#a8d8ea'; CHYS='#f7a6c0'; CW='#ffffff'; CF='#6c63ff'

fig = plt.figure(figsize=(20,18), facecolor=BG)
fig.suptitle(
    'PPO v3 — 5-Panel Diagnostic  (20-dim state | 15-action | 150 episodes)',
    fontsize=14, fontweight='bold', color='#e8eeff', y=0.985)

gs = gridspec.GridSpec(3,2,figure=fig,hspace=.50,wspace=.30,
                       left=.06,right=.97,top=.96,bottom=.05)

W = 10

# ═══ 1. REWARD CURVE ══════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0,0])
ax1.fill_between(eps, rewards, alpha=.15, color=CF)
ax1.plot(eps, rewards, color=CR, lw=1, alpha=.45, label='Raw')
ax1.plot(ep_sm(W), smooth(rewards,W), color=CW, lw=2.2, label=f'{W}-ep MA')
pk = eps[np.argmax(rewards)]; pv = rewards.max()
ax1.annotate(f'Peak {pv:.1f}\n(ep {pk})', xy=(pk,pv),
             xytext=(pk+10,pv-10), arrowprops=dict(arrowstyle='->',color=CW,lw=1.2),
             fontsize=8, color=CW)
ax1.axhline(rewards.mean(), color=CR, lw=1.2, ls=':', alpha=.7,
            label=f'Mean {rewards.mean():.1f}')
ax1.set_title('Plot 1 — Reward Curve  (does learning work?)',
              fontsize=11, fontweight='bold', pad=8)
ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward')
ax1.legend(fontsize=8, framealpha=.15, loc='lower right')
ax1.grid(True)

# ═══ 2. EVENT COUNTS ══════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0,1])
ax2b = ax2.twinx()
ax2.bar(eps, rlfs, color=CRLF, alpha=.75, width=1.0, label='RLF')
ax2.bar(eps, pps,  color=CPP,  alpha=.65, width=1.0, bottom=rlfs, label='Ping-Pong')
ax2b.plot(eps, hos, color=CHO, lw=1.5, alpha=.8, label='HO (R)')
ax2b.plot(ep_sm(W), smooth(hos,W), color=CW, lw=1.8, ls='--', label='HO MA (R)')
ax2.set_title('Plot 2 — Event Counts  (what is going wrong?)',
              fontsize=11, fontweight='bold', pad=8)
ax2.set_xlabel('Episode')
ax2.set_ylabel('RLF + PP', color=CRLF)
ax2b.set_ylabel('Handovers', color=CHO); ax2b.tick_params(axis='y', labelcolor=CHO)
l1,lb1 = ax2.get_legend_handles_labels(); l2,lb2 = ax2b.get_legend_handles_labels()
ax2.legend(l1+l2, lb1+lb2, fontsize=8, framealpha=.15, loc='upper right')
ax2.grid(True, axis='y')

# ═══ 3. PARAMETER EVOLUTION ═══════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1,0])
ax3b = ax3.twinx()
ax3.step(eps, ttt_fin, color=CTTT, lw=1.5, where='mid', alpha=.8, label='TTT (ms)')
ax3.plot(ep_sm(W), smooth(ttt_fin,W), color=CW, lw=2, label='TTT MA')
ax3b.step(eps, hys_fin, color=CHYS, lw=1.5, where='mid', alpha=.8, label='HYS dB (R)')
ax3b.plot(ep_sm(W), smooth(hys_fin,W), color=CHYS, lw=2, ls='--', alpha=.5, label='HYS MA (R)')
ax3.set_yticks([100,160,200,240,320])
ax3b.set_yticks([1.0,2.0,3.0,4.0,5.0,6.0])
ax3.set_title('Plot 3 — Parameter Evolution  (is RL actively choosing?)',
              fontsize=11, fontweight='bold', pad=8)
ax3.set_xlabel('Episode')
ax3.set_ylabel('TTT (ms)', color=CTTT)
ax3b.set_ylabel('HYS (dB)', color=CHYS); ax3b.tick_params(axis='y', labelcolor=CHYS)
l1,lb1=ax3.get_legend_handles_labels(); l2,lb2=ax3b.get_legend_handles_labels()
ax3.legend(l1+l2, lb1+lb2, fontsize=8, framealpha=.15)
ax3.grid(True, axis='y')

# ═══ 4. ZONE PARAMETERS ═══════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1,1])
zone_ttt = defaultdict(list); zone_hys = defaultdict(list)
for e in ph:
    zone_ttt[e['zone']].append(e['ttt_eff'])
    zone_hys[e['zone']].append(e['hys_eff'])

zones = sorted(zone_ttt.keys())
x = np.arange(len(zones)); w = .35
avg_ttt = [np.mean(zone_ttt[z]) for z in zones]
avg_hys = [np.mean(zone_hys[z]) for z in zones]
ax4b = ax4.twinx()
b1 = ax4.bar(x-w/2, avg_ttt, w, color=CTTT, alpha=.85, label='Avg TTT (ms)')
b2 = ax4b.bar(x+w/2, avg_hys, w, color=CHYS, alpha=.85, label='Avg HYS (R)')
for bar,v in zip(b1,avg_ttt):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=CTTT)
for bar,v in zip(b2,avg_hys):
    ax4b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05,
              f'{v:.1f}', ha='center', va='bottom', fontsize=8, color=CHYS)
ax4.set_xticks(x); ax4.set_xticklabels([z.replace('_','\n') for z in zones], fontsize=8)
ax4.set_title('Plot 4 — Zone Parameters  (did RL learn intelligently?)',
              fontsize=11, fontweight='bold', pad=8)
ax4.set_ylabel('Avg TTT (ms)', color=CTTT)
ax4b.set_ylabel('Avg HYS (dB)', color=CHYS); ax4b.tick_params(axis='y',labelcolor=CHYS)
l1,lb1=ax4.get_legend_handles_labels(); l2,lb2=ax4b.get_legend_handles_labels()
ax4.legend(l1+l2, lb1+lb2, fontsize=8, framealpha=.15)
ax4.grid(True, axis='y')

# ═══ 5. POLICY VARIANCE (full-width) ══════════════════════════════════════════
ax5 = fig.add_subplot(gs[2,:])
WV = 15
roll_mean = smooth(rewards, WV)
roll_std  = np.array([rewards[max(0,i-WV):i+1].std() for i in range(len(rewards))])
roll_std_sm = smooth(roll_std, WV)
ep_ma = ep_sm(WV)

ax5.fill_between(ep_ma,
                 roll_mean - roll_std_sm[:len(roll_mean)],
                 roll_mean + roll_std_sm[:len(roll_mean)],
                 alpha=.18, color=CF, label='+/- 1 std band')
ax5.plot(ep_ma, roll_mean, color=CW,   lw=2.2, label=f'{WV}-ep mean')
ax5.plot(eps,   roll_std,  color=CR,   lw=1.1, alpha=.45, label='Rolling std (raw)')
ax5.plot(ep_ma, roll_std_sm[:len(ep_ma)], color=CR, lw=2, ls='--', label=f'{WV}-ep std MA')
ax5.axhline(roll_std.mean(), color=CPP, lw=1, ls=':', alpha=.7,
            label=f'Mean std = {roll_std.mean():.1f}')
ax5.set_title('Plot 5 — Policy Variance  (is the policy stable?)',
              fontsize=11, fontweight='bold', pad=8)
ax5.set_xlabel('Episode'); ax5.set_ylabel('Reward')
ax5.legend(fontsize=9, framealpha=.15, loc='upper right', ncol=3)
ax5.grid(True)

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
