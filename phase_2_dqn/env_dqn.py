"""
env_dqn.py — Offline ns-3 Environment for Double-DQN Handover Agent
====================================================================
Fixes vs original train_rl.py (PPO):
  Fix 1 - SINR: NO max(-20) clamp → RLF events are learnable
  Fix 2 - distance_serving: reads serving_d_m column (new dataset.cpp)
  Fix 3 - cqi_serving: reads serving_cqi column (new dataset.cpp)
  Fix 4 - State dim 20-21: current TTT/HYS in state (fixes POMDP)
  Fix 5 - Reward: no +0.1 baseline, dwell-scaled HO cost, no clipping
  Fix 6 - Action: absolute 5×11=55 grid, not delta steps
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from collections import deque

# ── Import Algorithm1 from phase_2 ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase_2'))
from algo_2 import Algorithm1

# ============================================================================
# ACTION SPACE  — absolute 5 TTT × 11 HYS = 55 actions
# ============================================================================
TTT_LEVELS = [100, 160, 200, 240, 320]          # ms
HYS_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0,
              3.5, 4.0, 4.5, 5.0, 5.5, 6.0]    # dB
ACTION_DIM = len(TTT_LEVELS) * len(HYS_LEVELS)  # 55
STATE_DIM  = 22                                  # 20 base + ttt_norm + hys_norm


def decode_action(idx: int):
    """Flat index → (ttt_ms, hys_db)."""
    return TTT_LEVELS[idx // len(HYS_LEVELS)], HYS_LEVELS[idx % len(HYS_LEVELS)]


def encode_action(ttt_ms: int, hys_db: float) -> int:
    """(ttt_ms, hys_db) → flat index (nearest HYS)."""
    ti = TTT_LEVELS.index(ttt_ms)
    hi = min(range(len(HYS_LEVELS)), key=lambda i: abs(HYS_LEVELS[i] - hys_db))
    return ti * len(HYS_LEVELS) + hi


# ============================================================================
# STATE BUILDER  — 22 dimensions
# ============================================================================
class StateBuilder:
    """
    Dimensions
    ----------
    0  rsrp_norm            (rsrp+130)/60        clipped [0,1]
    1  sinr_norm            (sinr+20)/30         clipped [0,1]
    2  cqi_norm             cqi/15               clipped [0,1]  ← real value
    3  dist_norm            dist/500             clipped [0,1]  ← real distance
    4-7  zone_onehot        CENTER/HANDOFF/CRITICAL/EDGE
    8-10 top3_margin_norm   (nbr_rsrp-rsrp)/30   clipped [-1,1]
    11-13 signal_quality    POOR/FAIR/GOOD
    14 time_since_ho_norm   t/30                 clipped [0,1]  ← 30s window
    15 velocity_norm        v/50                 clipped [0,1]
    16 neighbor_count_norm  n/10                 clipped [0,1]
    17 recent_rlf_rate      count/10             clipped [0,1]
    18 recent_pp_rate       count/10             clipped [0,1]
    19 rsrp_trend_norm      Δrsrp/5              clipped [-1,1]
    20 ttt_norm             ttt/320              clipped [0,1]  ← POMDP fix
    21 hys_norm             hys/6.0              clipped [0,1]  ← POMDP fix
    """

    def build(self, algo1_out: dict, time_since_ho: float,
              recent_rlf: int, recent_pp: int,
              rsrp_prev: float | None,
              ttt_eff: int, hys_eff: float) -> np.ndarray:

        state = []
        rsrp = float(algo1_out.get('rsrp_serving_dbm', -120.0))
        sinr = float(algo1_out.get('sinr_serving_db', -20.0))

        state.append(float(np.clip((rsrp + 130.0) / 60.0,   0.0, 1.0)))   # 0
        state.append(float(np.clip((sinr +  20.0) / 30.0,   0.0, 1.0)))   # 1
        state.append(float(np.clip(algo1_out.get('cqi_serving', 7) / 15.0, 0.0, 1.0)))  # 2
        state.append(float(np.clip(algo1_out.get('distance_serving', 200.0) / 500.0, 0.0, 1.0)))  # 3

        zone_map = {
            'CELL_CENTER':  [1, 0, 0, 0],
            'HANDOFF_ZONE': [0, 1, 0, 0],
            'CRITICAL_ZONE':[0, 0, 1, 0],
            'EDGE_ZONE':    [0, 0, 0, 1],
        }
        state.extend(zone_map.get(algo1_out.get('zone', 'HANDOFF_ZONE'), [0, 1, 0, 0]))  # 4-7

        nbr_rsrp = algo1_out.get('rsrp_neighbors', [])
        nbr_dist = algo1_out.get('distance_neighbors', [])
        top3 = sorted(zip(nbr_rsrp, nbr_dist), key=lambda x: -x[0])
        for i in range(3):                                                                 # 8-10
            m = (top3[i][0] - rsrp) if i < len(top3) else -20.0
            state.append(float(np.clip(m / 30.0, -1.0, 1.0)))

        sq_map = {'POOR': [1, 0, 0], 'FAIR': [0, 1, 0], 'GOOD': [0, 0, 1]}
        state.extend(sq_map.get(algo1_out.get('signal_quality', 'FAIR'), [0, 1, 0]))      # 11-13

        state.append(float(np.clip(time_since_ho / 30.0, 0.0, 1.0)))                     # 14
        state.append(float(np.clip(algo1_out.get('velocity', 0.0) / 50.0, 0.0, 1.0)))    # 15
        state.append(float(np.clip(len(nbr_rsrp) / 10.0, 0.0, 1.0)))                     # 16
        state.append(float(np.clip(recent_rlf / 10.0, 0.0, 1.0)))                        # 17
        state.append(float(np.clip(recent_pp  / 10.0, 0.0, 1.0)))                        # 18

        trend = np.clip((rsrp - rsrp_prev) / 5.0, -1.0, 1.0) if rsrp_prev is not None else 0.0
        state.append(float(trend))                                                         # 19
        state.append(float(np.clip(ttt_eff / 320.0, 0.0, 1.0)))                          # 20 ← POMDP fix
        state.append(float(np.clip(hys_eff / 6.0,   0.0, 1.0)))                          # 21 ← POMDP fix

        arr = np.array(state, dtype=np.float32)
        assert len(arr) == STATE_DIM, f"State dim mismatch: {len(arr)} != {STATE_DIM}"
        return arr


# ============================================================================
# REWARD FUNCTION  — redesigned, no clipping
# ============================================================================
class RewardFunction:
    """
    Components
    ----------
    RLF event          : -2.0  (highest priority)
    Ping-pong event    : -1.5
    Rapid HO (<2s)     : up to -0.6 (dwell-scaled: longer wait → smaller penalty)
    SINR quality       : +0.05 (small, capped)
    SINR improvement   : +0.02 per dB (capped ±2dB)
    Stability (center) : +0.04 per tick if no events and in CELL_CENTER
    Stability (handoff): +0.01 per tick
    Min-TTT penalty    : -0.02  (discourages parking at TTT=100ms forever)
    No clipping        : critic sees full gradient signal
    """

    def compute(self, rlf: bool, pp: bool, ho: bool,
                sinr_cur: float, sinr_prev: float | None,
                time_since_ho: float, zone: str, ttt_eff: int) -> float:
        r = 0.0

        if rlf:
            r -= 2.0
        if pp:
            r -= 1.5

        if ho:
            # dwell_factor: 0 = HO right after last HO, 1 = waited ≥2s
            dwell = min(time_since_ho / 2.0, 1.0)
            r -= 0.6 * (1.0 - dwell)

        sinr_norm = np.clip((sinr_cur + 20.0) / 30.0, 0.0, 1.0)
        r += 0.05 * sinr_norm

        if sinr_prev is not None:
            r += 0.02 * float(np.clip(sinr_cur - sinr_prev, -2.0, 2.0))

        if not ho and not rlf:
            if zone == 'CELL_CENTER':
                r += 0.04
            elif zone == 'HANDOFF_ZONE':
                r += 0.01

        if ttt_eff <= 100:
            r -= 0.02  # discourage permanently aggressive stance

        return float(r)


# ============================================================================
# OFFLINE NS-3 ENVIRONMENT
# ============================================================================
class OfflineNs3Env:
    """
    Wraps pre-generated CSV tick files.  One episode = one UE trace
    from a single file, up to max_steps ticks.

    State tracking inside this class:
      - time_since_last_ho : seconds since last handover
      - recent_rlf_events  : deque(maxlen=10) of 0/1 indicators
      - recent_pp_events   : deque(maxlen=10) of 0/1 indicators
      - in_handover gate   : prevents double-counting repeated ho_event flags
      - prev_serving, last_ho_to : for ping-pong detection (A→B→A within 1s)
    """

    def __init__(self, csv_files: list, max_steps: int = 1024):
        if not csv_files:
            raise RuntimeError("OfflineNs3Env: no CSV files provided")
        self.max_steps = max_steps

        # Pre-load all CSVs once into per-UE DataFrames — O(1) on reset()
        print(f"  Pre-loading {len(csv_files)} CSV files into memory...", flush=True)
        self._ue_traces: list = []
        for path in csv_files:
            try:
                df_full = pd.read_csv(path)
                for uid in df_full['ue_id'].unique():
                    trace = df_full[df_full['ue_id'] == uid].reset_index(drop=True)
                    if len(trace) >= 10:   # skip trivially short traces
                        self._ue_traces.append(trace)
            except Exception as e:
                print(f"  WARN: skipping {path}: {e}")
        if not self._ue_traces:
            raise RuntimeError("No valid UE traces found in provided CSV files")
        print(f"  Loaded {len(self._ue_traces):,} UE traces total.", flush=True)

        self.df               = None
        self.row_idx          = 0
        self.algo1            = None

        self.time_since_ho    = 0.0
        self.sinr_prev        = None
        self.rsrp_prev        = None
        self.in_handover      = False
        self.prev_serving     = None
        self.last_ho_to       = None
        self.last_ho_time     = -999.0

        self.recent_rlf       = deque(maxlen=10)
        self.recent_pp        = deque(maxlen=10)
        self.recent_ho        = deque(maxlen=10)
        self.rlf_counter      = 0
        self._state_builder   = StateBuilder()

        self.ttt_eff          = 160
        self.hys_eff          = 3.0

    # ── Reset ───────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """Pick a random pre-loaded UE trace. O(1) — no disk I/O."""
        self.df      = random.choice(self._ue_traces)
        self.row_idx = 0

        # Fresh algorithm instance per episode
        self.algo1 = Algorithm1()
        self.algo1.serving_cell_id = int(self.df.iloc[0]['serving_cell'])

        # Reset all tracking state
        self.time_since_ho  = 0.0
        self.sinr_prev      = None
        self.rsrp_prev      = None
        self.in_handover    = False
        self.prev_serving   = None
        self.last_ho_to     = None
        self.last_ho_time   = -999.0
        self.rlf_counter    = 0
        self.recent_rlf.clear()
        self.recent_pp.clear()
        self.recent_ho.clear()

        self.ttt_eff = 160
        self.hys_eff = 3.0

        decision = self._process_row(self.ttt_eff, self.hys_eff)
        return self._state_builder.build(
            decision, self.time_since_ho,
            sum(self.recent_rlf), sum(self.recent_pp),
            self.rsrp_prev, self.ttt_eff, self.hys_eff
        )

    # ── Step ────────────────────────────────────────────────────────────────
    def step(self, action_idx: int):
        """
        Apply action, advance one tick.

        Returns
        -------
        next_state : np.ndarray  (STATE_DIM,)
        reward     : float
        done       : bool
        info       : dict  {rlf, pp, ho, sinr, ttt, hys, zone}
        """
        self.ttt_eff, self.hys_eff = decode_action(action_idx)

        # Advance row index
        self.row_idx += 1
        done = (self.row_idx >= len(self.df) - 1) or (self.row_idx >= self.max_steps)

        decision   = self._process_row(self.ttt_eff, self.hys_eff)
        sinr_cur   = float(decision.get('sinr_serving_db', -20.0))
        rsrp_cur   = float(decision.get('rsrp_serving_dbm', -120.0))
        zone       = decision.get('zone', 'HANDOFF_ZONE')
        ho_flag    = bool(decision.get('ho_occurred', False))
        now_s      = float(decision.get('now_s', 0.0))
        target_c   = decision.get('target_cell_id', -1)
        serving_c  = decision.get('serving_cell_id', -1)

        # ── RLF detection (T310: 2 consecutive ticks below -20 dB SINR) ────
        if sinr_cur < -20.0:
            self.rlf_counter += 1
        else:
            self.rlf_counter = 0
        rlf_event = (self.rlf_counter >= 2)
        self.recent_rlf.append(1 if rlf_event else 0)

        # ── Handover deduplication ────────────────────────────────────────
        ho_occurred = False
        if ho_flag and not self.in_handover:
            ho_occurred    = True
            self.in_handover = True
        elif not ho_flag:
            self.in_handover = False

        # ── Ping-pong: A→B→A within 1 second ─────────────────────────────
        pp_event = False
        if ho_occurred and self.prev_serving is not None:
            if (self.last_ho_to == serving_c and
                    target_c == self.prev_serving and
                    now_s - self.last_ho_time < 1.0):
                pp_event = True
        self.recent_pp.append(1 if pp_event else 0)
        self.recent_ho.append(1 if ho_occurred else 0)

        # ── Reward ────────────────────────────────────────────────────────
        reward = RewardFunction().compute(
            rlf=rlf_event, pp=pp_event, ho=ho_occurred,
            sinr_cur=sinr_cur, sinr_prev=self.sinr_prev,
            time_since_ho=self.time_since_ho,
            zone=zone, ttt_eff=self.ttt_eff
        )

        # ── Next state ────────────────────────────────────────────────────
        next_state = self._state_builder.build(
            decision,
            self.time_since_ho,
            sum(self.recent_rlf), sum(self.recent_pp),
            self.rsrp_prev,
            self.ttt_eff, self.hys_eff
        )

        # ── Update tracking ───────────────────────────────────────────────
        self.sinr_prev     = sinr_cur
        self.rsrp_prev     = rsrp_cur
        self.time_since_ho = (0.0 if ho_occurred else self.time_since_ho + 0.1)
        if ho_occurred:
            self.prev_serving = serving_c
            self.last_ho_to   = target_c
            self.last_ho_time = now_s

        info = {
            'rlf': rlf_event, 'pp': pp_event, 'ho': ho_occurred,
            'sinr': sinr_cur,  'ttt': self.ttt_eff, 'hys': self.hys_eff,
            'zone': zone
        }
        return next_state, reward, done, info

    # ── Internal row reader (all bugs fixed) ─────────────────────────────
    def _process_row(self, ttt_eff: int, hys_eff: float) -> dict:
        row = self.df.iloc[self.row_idx]

        # Build neighbor lists (n1..n6 in new dataset exclude serving cell)
        nbr_ids, nbr_rsrp, nbr_dist = [], [], []
        for i in range(1, 7):
            nid = row.get(f'n{i}_id', -1)
            if pd.isna(nid) or int(nid) == -1:
                continue
            nbr_ids.append(int(nid))
            nbr_rsrp.append(float(row[f'n{i}_rsrp_dbm']))
            nbr_dist.append(float(row[f'n{i}_d_m']))

        # Serving metrics — look up from neighbor cols if cell differs
        csv_serving = int(row['serving_cell'])
        if csv_serving == self.algo1.serving_cell_id:
            s_rsrp = float(row['serving_rsrp_dbm'])
            s_sinr = float(row['serving_sinr_db'])
            # Fix 2: real distance
            s_dist = float(row['serving_d_m']) if 'serving_d_m' in row and not pd.isna(row.get('serving_d_m')) else 200.0
        else:
            s_rsrp, s_sinr, s_dist = -140.0, -30.0, 400.0  # deep fade, not clamped
            for i, nid in enumerate(nbr_ids):
                if nid == self.algo1.serving_cell_id:
                    s_rsrp = nbr_rsrp[i]
                    s_sinr = float(row[f'n{i+1}_sinr_db'])  # Fix 1: no clamp
                    s_dist = nbr_dist[i]                     # Fix 2: real dist
                    break

        # Fix 3: real CQI
        if 'serving_cqi' in row and not pd.isna(row.get('serving_cqi')):
            cqi = int(row['serving_cqi'])
        else:
            cqi = 10  # fallback for legacy CSVs

        decision = self.algo1.step(
            rsrp_serving    = s_rsrp,
            sinr_serving    = s_sinr,
            cqi_serving     = cqi,
            distance_serving= s_dist,
            rsrp_neighbors  = nbr_rsrp,
            neighbor_ids    = nbr_ids,
            distance_neighbors = nbr_dist,
            velocity        = float(row.get('speed_mps', 1.5)),
            now_s           = float(row['time_s']),
            TTT_eff         = ttt_eff,
            HYS_eff         = hys_eff
        )
        return decision
