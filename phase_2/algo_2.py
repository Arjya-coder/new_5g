"""
PHASE 2 IMPROVED VERSION - RLF CRITICAL FIX ONLY
MINIMAL CHANGE: Only modify RLF penalty and velocity-aware scaling
NO changes to handover or ping-pong logic
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import json
from datetime import datetime
import os

# ============================================================================
# ALGORITHM 1 — NO CHANGES (Keep existing)
# ============================================================================

class Algorithm1:
    """Baseline 5G Handover Algorithm - UNCHANGED"""
    
    def __init__(self):
        self.rsrp_s_f = None
        self.sinr_s_f = None
        self.rsrp_n_f = {}
        self.a3_hold_ms = {}
        self.previous_strongest_neighbor = None
        self.recent_handoffs = []
        self.last_handover_time = -float('inf')
        self.rsrp_history_serving = []
        self.rsrp_history_neighbors = {}
        self.serving_cell_id = 0

    def _update_history(self, now_s, val, history_list):
        history_list.append((now_s, val))
        while len(history_list) > 0 and now_s - history_list[0][0] > 5.0:
            history_list.pop(0)

    def step(self, 
             rsrp_serving: float, 
             sinr_serving: float, 
             cqi_serving: int, 
             distance_serving: float,
             rsrp_neighbors: list, 
             neighbor_ids: list, 
             distance_neighbors: list,
             velocity: float, 
             now_s: float, 
             TTT_eff: int, 
             HYS_eff: float):
        """Execute one 100ms tick"""
        dt_s = 0.1
        K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
        rsrp_neighbors = rsrp_neighbors[:K]
        neighbor_ids = neighbor_ids[:K]
        distance_neighbors = distance_neighbors[:K]

        for nid in list(self.rsrp_n_f.keys()):
            if nid not in neighbor_ids:
                del self.rsrp_n_f[nid]
                self.a3_hold_ms.pop(nid, None)
                self.rsrp_history_neighbors.pop(nid, None)

        alpha = 0.2
        
        if self.rsrp_s_f is None:
            self.rsrp_s_f = rsrp_serving
            self.sinr_s_f = sinr_serving
        else:
            self.rsrp_s_f = alpha * rsrp_serving + (1 - alpha) * self.rsrp_s_f
            self.sinr_s_f = alpha * sinr_serving + (1 - alpha) * self.sinr_s_f
            
        for i, nid in enumerate(neighbor_ids):
            if nid not in self.rsrp_n_f:
                self.rsrp_n_f[nid] = rsrp_neighbors[i]
                self.a3_hold_ms[nid] = 0
                self.rsrp_history_neighbors[nid] = []
            else:
                self.rsrp_n_f[nid] = alpha * rsrp_neighbors[i] + (1 - alpha) * self.rsrp_n_f[nid]
                
        self._update_history(now_s, self.rsrp_s_f, self.rsrp_history_serving)
        for nid in neighbor_ids:
            self._update_history(now_s, self.rsrp_n_f[nid], self.rsrp_history_neighbors[nid])

        if self.rsrp_s_f < -120 or self.sinr_s_f < -10 or cqi_serving < 5:
            signal_quality = "POOR"
        elif self.rsrp_s_f >= -100 and self.sinr_s_f >= 0 and cqi_serving >= 10:
            signal_quality = "GOOD"
        else:
            signal_quality = "FAIR"

        if distance_serving < 350:
            zone = "CELL_CENTER"
        elif distance_serving < 480:
            zone = "HANDOFF_ZONE"
        elif distance_serving <= 500:
            zone = "CRITICAL_ZONE"
        else:
            zone = "EDGE_ZONE"

        rlf_risk = 0.0
        if self.rsrp_s_f < -120: rlf_risk += 0.40
        elif self.rsrp_s_f < -110: rlf_risk += 0.20
        if self.sinr_s_f < -10: rlf_risk += 0.40
        elif self.sinr_s_f < -5: rlf_risk += 0.20
        if distance_serving >= 480: rlf_risk += 0.05
        rlf_risk = min(rlf_risk, 1.0)

        in_dwell = (now_s - self.last_handover_time) < 0.5
        MUST_HO = (signal_quality == "POOR") or (zone == "EDGE_ZONE") or (rlf_risk >= 0.60)

        eligible_candidates = []
        K_distance_max = 1500
        
        if len(neighbor_ids) > 0:
            current_strongest = max(neighbor_ids, key=lambda nid: self.rsrp_n_f[nid])
        else:
            current_strongest = None
        
        for i, nid in enumerate(neighbor_ids):
            if distance_neighbors[i] >= K_distance_max:
                continue
                
            A3_condition = (self.rsrp_n_f[nid] - self.rsrp_s_f) >= HYS_eff
            
            if A3_condition:
                self.a3_hold_ms[nid] += int(dt_s * 1000)
            else:
                self.a3_hold_ms[nid] = 0
            
            if self.a3_hold_ms[nid] >= TTT_eff:
                if current_strongest == nid and self.previous_strongest_neighbor == nid:
                    eligible_candidates.append(i)

        margin_pred = {}
        for i in eligible_candidates:
            nid = neighbor_ids[i]
            margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
                
        eligible_candidates.sort(key=lambda i: (
            margin_pred[i], 
            self.rsrp_n_f[neighbor_ids[i]], 
            -distance_neighbors[i]
        ), reverse=True)

        action = 0
        target_cell_id = None
        ho_occurred = False
        
        if MUST_HO:
            if eligible_candidates:
                target_idx = eligible_candidates[0]
                action = 1
                target_cell_id = neighbor_ids[target_idx]
                self.recent_handoffs.append(now_s)
                self.last_handover_time = now_s
                self.serving_cell_id = target_cell_id
                ho_occurred = True
            else:
                if len(neighbor_ids) > 0:
                    candidates_valid = [(i, self.rsrp_n_f[neighbor_ids[i]]) 
                                       for i in range(len(neighbor_ids)) 
                                       if self.rsrp_n_f[neighbor_ids[i]] > -115]
                    
                    if candidates_valid:
                        best_idx = max(candidates_valid, key=lambda x: x[1])[0]
                        action = 1
                        target_cell_id = neighbor_ids[best_idx]
                        self.recent_handoffs.append(now_s)
                        self.last_handover_time = now_s
                        self.serving_cell_id = target_cell_id
                        ho_occurred = True
        
        elif rlf_risk >= 0.35 and eligible_candidates and not in_dwell:
            target_idx = eligible_candidates[0]
            action = 1
            target_cell_id = neighbor_ids[target_idx]
            self.recent_handoffs.append(now_s)
            self.last_handover_time = now_s
            self.serving_cell_id = target_cell_id
            ho_occurred = True
        
        self.previous_strongest_neighbor = current_strongest
        
        rsrp_neighbors_ordered = [float(self.rsrp_n_f[nid]) for nid in neighbor_ids]
        
        return {
            'action': action,
            'target_cell_id': target_cell_id,
            'rsrp_serving_dbm': float(self.rsrp_s_f),
            'sinr_serving_db': float(self.sinr_s_f),
            'cqi_serving': int(cqi_serving),
            'distance_serving': float(distance_serving),
            'velocity': float(velocity),
            'neighbor_ids': neighbor_ids,
            'rsrp_neighbors': rsrp_neighbors_ordered,
            'distance_neighbors': [float(d) for d in distance_neighbors],
            'zone': zone,
            'signal_quality': signal_quality,
            'now_s': now_s,
            'serving_cell_id': self.serving_cell_id,
            'num_neighbors': len(neighbor_ids),
            'ho_occurred': ho_occurred
        }


# ============================================================================
# RL MODULE - CRITICAL FIX #5: RLF PENALTY ONLY
# ============================================================================
# MINIMAL CHANGE: Only modify RLF penalty calculation
# PRESERVES: All HO and PP logic (100% same)
# ============================================================================

class RLModule:
    """Production RL module - RLF FIX ONLY"""
    
    def __init__(self):
        self.entropy_coef_start = 0.05
        self.entropy_coef_end = 0.005
        self.rlf_counter = 0
        self.recent_rlf_history = deque(maxlen=10)
        self.recent_pp_history = deque(maxlen=10)
        self.recent_ho_history = deque(maxlen=10)

    def build_state_vector(self, algo1_output, ttt_eff: int, hys_eff: float,
                           time_since_last_ho=0.0, recent_rlf_count=0, recent_pp_count=0, 
                           recent_ho_count=0, rsrp_prev=None, velocity=0.0):
        """
        Build 23-dim state (UNCHANGED except velocity used for RLF scaling)
        
        Note: velocity parameter added for context (used in reward, not state)
        """
        state = []
        
        rsrp = algo1_output['rsrp_serving_dbm']
        rsrp_norm = np.clip((rsrp - (-130)) / (-70 - (-130)), 0, 1)
        state.append(rsrp_norm)
        
        sinr = algo1_output['sinr_serving_db']
        sinr_norm = np.clip((sinr - (-20)) / (10 - (-20)), 0, 1)
        state.append(sinr_norm)
        
        cqi = algo1_output['cqi_serving']
        cqi_norm = np.clip(cqi / 15.0, 0, 1)
        state.append(cqi_norm)
        
        distance = algo1_output['distance_serving']
        distance_norm = np.clip(distance / 500.0, 0, 1)
        state.append(distance_norm)
        
        zone = algo1_output['zone']
        zone_map = {
            'CELL_CENTER': [1, 0, 0, 0],
            'HANDOFF_ZONE': [0, 1, 0, 0],
            'CRITICAL_ZONE': [0, 0, 1, 0],
            'EDGE_ZONE': [0, 0, 0, 1]
        }
        zone_onehot = zone_map.get(zone, [0, 1, 0, 0])
        state.extend(zone_onehot)
        
        rsrp_neighbors = algo1_output['rsrp_neighbors']
        distance_neighbors = algo1_output['distance_neighbors']
        
        neighbors_data = list(zip(rsrp_neighbors, distance_neighbors))
        neighbors_data.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(3):
            if i < len(neighbors_data):
                margin = neighbors_data[i][0] - rsrp
            else:
                margin = -20.0
            
            margin_norm = np.clip(margin / 30.0, -1, 1)
            state.append(margin_norm)
        
        signal_quality = algo1_output['signal_quality']
        sq_map = {
            'POOR': [1, 0, 0],
            'FAIR': [0, 1, 0],
            'GOOD': [0, 0, 1]
        }
        sq_onehot = sq_map.get(signal_quality, [0, 1, 0])
        state.extend(sq_onehot)
        
        time_norm = np.clip(time_since_last_ho / 10.0, 0, 1)
        state.append(time_norm)

        ttt_norm = np.clip((float(ttt_eff) - 100.0) / (320.0 - 100.0), 0, 1)
        state.append(ttt_norm)

        hys_norm = np.clip((float(hys_eff) - 2.0) / (5.0 - 2.0), 0, 1)
        state.append(hys_norm)
        
        velocity = algo1_output['velocity']
        velocity_norm = np.clip(velocity / 50.0, 0, 1)
        state.append(velocity_norm)
        
        neighbor_count = algo1_output['num_neighbors']
        neighbor_norm = np.clip(neighbor_count / 10.0, 0, 1)
        state.append(neighbor_norm)
        
        rlf_rate = np.clip(recent_rlf_count / 10.0, 0, 1)
        state.append(rlf_rate)
        
        pp_rate = np.clip(recent_pp_count / 10.0, 0, 1)
        state.append(pp_rate)
        
        ho_rate = np.clip(recent_ho_count / 10.0, 0, 1)
        state.append(ho_rate)
        
        if rsrp_prev is not None:
            rsrp_trend = (rsrp - rsrp_prev) / 0.1
            rsrp_trend_norm = np.clip(rsrp_trend / 5.0, -1, 1)
        else:
            rsrp_trend_norm = 0.0
        state.append(rsrp_trend_norm)
        
        state = np.array(state, dtype=np.float32)
        assert len(state) == 23, f"State dim should be 23, got {len(state)}"
        assert np.all(state >= -1.0) and np.all(state <= 1.0), \
            f"State out of bounds: min={state.min()}, max={state.max()}"
        
        return state
    
    def compute_reward(self, rlf_event, ping_pong_event, sinr_delta, sinr_current,
                      handover_occurred, delta_ttt_step, delta_hys_db, 
                      recent_ho_count, velocity=0.0, no_op=False):
        """
        ✅ CRITICAL FIX #5: VELOCITY-AWARE RLF PENALTY
        
        Problem: RLF penalty -1.5 insufficient in high-speed scenarios
        - At low speed (urban): RLF penalty -1.5 is OK
        - At high speed (rural): RLF penalty should be -3.0+ (cascading effect)
        
        Solution: Scale RLF penalty by velocity
        - Captures the fact: higher speed = more dangerous RLF
        - Formula: base_penalty × (1 + velocity/speed_scale)
        
        UNCHANGED: HO penalty (-0.5), PP penalty (-1.0)
        """
        r = 0.0
        
        # ========================================================================
        # ✅ CRITICAL FIX #5: Velocity-Aware RLF Penalty
        # ========================================================================
        # Base RLF penalty increased from -1.5 to -2.0
        # Then scaled by velocity for high-speed scenarios
        
        if rlf_event:
            # Base RLF penalty (increased from -1.5)
            base_rlf_penalty = -2.0
            
            # Velocity scaling: higher speed = higher penalty
            # At 0 m/s (pedestrian): scale = 1.0 → penalty = -2.0
            # At 50 m/s (train): scale = 2.0 → penalty = -4.0
            # At 70 m/s (high-speed): scale = 2.4 → penalty = -4.8
            velocity_scale = 1.0 + (velocity / 50.0)  # Linear scale by velocity
            
            r_rlf = base_rlf_penalty * velocity_scale
            r_rlf = np.clip(r_rlf, -5.0, -1.5)  # Clip between -5.0 and -1.5
        else:
            r_rlf = 0.0
        
        # ========================================================================
        # UNCHANGED: Ping-Pong Penalty (keep at -1.0)
        # ========================================================================
        if ping_pong_event:
            r_pp = -1.0  # UNCHANGED
        else:
            r_pp = 0.0
        
        # ========================================================================
        # UNCHANGED: SINR Improvement (keep same)
        # ========================================================================
        r_sinr_delta = np.clip(sinr_delta * 0.1, -0.2, 0.2)
        r_sinr_abs = np.clip((sinr_current + 20) / 30 * 0.1, 0, 0.1)
        r_sinr = r_sinr_delta + r_sinr_abs
        
        # ========================================================================
        # UNCHANGED: Handover Penalty (keep at -0.5)
        # ========================================================================
        if handover_occurred:
            r_ho = -0.5  # UNCHANGED
        else:
            r_ho = 0.0
        
        # ========================================================================
        # UNCHANGED: Smoothing Penalty (keep same)
        # ========================================================================
        r_smooth = -0.15 * (abs(delta_ttt_step) + abs(delta_hys_db))
        
        # ========================================================================
        # UNCHANGED: No-op Penalty (keep same)
        # ========================================================================
        r_noop = 0.0
        if no_op:
            r_noop = -0.01
        
        # ========================================================================
        # UNCHANGED: HO Frequency Penalty (keep at -0.1)
        # ========================================================================
        r_ho_freq = -0.1 * recent_ho_count  # UNCHANGED
        
        # ========================================================================
        # Total Reward (with new RLF penalty)
        # ========================================================================
        r = r_rlf + r_pp + r_sinr + r_ho + r_smooth + r_noop + r_ho_freq
        r = np.clip(r, -5.0, +1.0)  # Updated range for higher RLF penalty
        
        return float(r)
    
    def get_entropy_coef(self, episode, total_episodes):
        """Entropy decay (UNCHANGED)"""
        progress = min(episode / total_episodes, 1.0)
        entropy_coef = self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)
        return entropy_coef


# ============================================================================
# PPO AGENT - NO CHANGES
# ============================================================================

class PPOAgent:
    """PPO Agent (UNCHANGED - only RLF penalty modified in reward)"""
    
    def __init__(self, state_dim=23, action_dim=15, learning_rate_actor=1e-3, 
                 learning_rate_critic=3e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_actor)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate_critic)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.max_grad_norm = 0.5
        
        self.batch_size = 64
        self.epochs_per_update = 10
        self.target_kl = 0.015
        
    def _build_actor(self):
        """Actor network"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='linear')
        ])
        return model
    
    def _build_critic(self):
        """Critic network"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        return model
    
    def select_action(self, state):
        """Select action from policy"""
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        logits = self.actor(state, training=False)[0].numpy()
        value = self.critic(state, training=False)[0].numpy()[0]
        
        probs = tf.nn.softmax(logits).numpy()
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = float(np.log(probs[action] + 1e-8))
        
        return action, log_prob, value
    
    def decode_action(self, action):
        """Decode 15-dim action to (ΔTTT_step, ΔHYS_db) (UNCHANGED)"""
        grid = [
            (-2, -0.5), (-2,  0.0), (-2, +0.5),
            (-1, -0.5), (-1,  0.0), (-1, +0.5),
            (0, -0.5), (0,  0.0), (0, +0.5),
            (+1, -0.5), (+1,  0.0), (+1, +0.5),
            (+2, -0.5), (+2,  0.0), (+2, +0.5)
        ]
        return grid[action]
    
    def compute_effective_parameters(self, ttt_rule, hys_rule, delta_ttt_step, delta_hys_db):
        """Compute effective parameters (UNCHANGED)"""
        ttt_steps = [100, 160, 200, 240, 320]
        
        if ttt_rule not in ttt_steps:
            ttt_rule = 160
        
        idx = ttt_steps.index(ttt_rule)
        idx = max(0, min(4, idx + delta_ttt_step))
        ttt_eff = ttt_steps[idx]
        
        hys_eff = np.clip(hys_rule + delta_hys_db, 2.0, 5.0)
        
        return ttt_eff, hys_eff
    
    def train_step(self, trajectories, entropy_coef):
        """PPO training step (UNCHANGED)"""
        states = np.array([t['s'] for t in trajectories], dtype=np.float32)
        actions = np.array([t['a'] for t in trajectories], dtype=np.int32)
        advantages = np.array([t['A_norm'] for t in trajectories], dtype=np.float32)
        returns = np.array([t['G'] for t in trajectories], dtype=np.float32)
        old_log_probs = np.array([t['log_prob'] for t in trajectories], dtype=np.float32)
        
        for epoch in range(self.epochs_per_update):
            indices = np.arange(len(trajectories))
            np.random.shuffle(indices)
            
            for batch_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[batch_idx:batch_idx + self.batch_size]
                
                s_batch = states[batch_indices]
                a_batch = actions[batch_indices]
                adv_batch = advantages[batch_indices]
                ret_batch = returns[batch_indices]
                old_lp_batch = old_log_probs[batch_indices]
                
                with tf.GradientTape() as tape:
                    logits = self.actor(s_batch, training=True)
                    probs = tf.nn.softmax(logits)
                    new_log_probs = tf.math.log(
                        tf.reduce_sum(tf.one_hot(a_batch, self.action_dim) * probs, axis=1) + 1e-8
                    )
                    
                    ratio = tf.exp(new_log_probs - old_lp_batch)
                    surr1 = ratio * adv_batch
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                    loss_clip = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                    loss_entropy = -entropy_coef * tf.reduce_mean(entropy)
                    
                    loss_actor = loss_clip + loss_entropy
                
                grads = tape.gradient(loss_actor, self.actor.trainable_variables)
                grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in grads]
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                
                with tf.GradientTape() as tape:
                    values = tf.squeeze(self.critic(s_batch, training=True), axis=1)
                    v_clipped = tf.clip_by_value(
                        values, 
                        ret_batch - self.clip_eps, 
                        ret_batch + self.clip_eps
                    )
                    loss_value = 0.5 * tf.reduce_mean(
                        tf.minimum(
                            (values - ret_batch) ** 2,
                            (v_clipped - ret_batch) ** 2
                        )
                    )
                
                grads = tape.gradient(loss_value, self.critic.trainable_variables)
                grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in grads]
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
    
    def save(self, path):
        """Save networks"""
        self.actor.save(f"{path}/actor.h5")
        self.critic.save(f"{path}/critic.h5")
    
    def load(self, path):
        """Load networks"""
        self.actor = keras.models.load_model(f"{path}/actor.h5")
        self.critic = keras.models.load_model(f"{path}/critic.h5")


# ============================================================================
# TRAINING ENV - MODIFIED TO PASS VELOCITY TO REWARD
# ============================================================================

class TrainingEnv:
    """
    Production environment wrapper.
    
    ✅ Modified: Pass velocity to reward computation (for RLF scaling)
    ✅ UNCHANGED: All HO and PP logic
    """
    
    def __init__(self, ns3_env, algo1, rl_module):
        self.ns3_env = ns3_env
        self.algo1 = algo1
        self.rl_module = rl_module
        
        self.time_since_last_ho = 0.0
        self.sinr_prev = None
        self.rsrp_prev = None
        
        self.prev_serving_cell = None
        self.serving_cell = None
        self.target_cell = None
        self.last_ho_time = -float('inf')
        
        self.rlf_counter = 0
        self.recent_rlf_events = deque(maxlen=10)
        self.recent_pp_events = deque(maxlen=10)
        self.recent_ho_events = deque(maxlen=10)
        
        self.param_history = []
        self.in_handover = False

    @staticmethod
    def _scenario_rlf_threshold_dbm(scenario_id: int) -> float:
        return {
            1: -122.0, 2: -121.0, 3: -124.0, 4: -122.0,
            5: -125.0, 6: -123.0, 7: -118.0,
        }.get(int(scenario_id), -122.0)

    def step(self, ttt_eff, hys_eff):
        """Execute one step"""
        algo1_output = self.ns3_env.step(ttt_eff, hys_eff, self.algo1)
        done = self.ns3_env.is_done()
        
        rsrp_current = algo1_output['rsrp_serving_dbm']
        sinr_current = algo1_output['sinr_serving_db']
        algo1_action = algo1_output['action']
        target_cell = algo1_output['target_cell_id']
        now_s = algo1_output['now_s']
        serving_cell = algo1_output['serving_cell_id']
        zone = algo1_output['zone']
        velocity = algo1_output['velocity']
        distance = algo1_output['distance_serving']
        ho_event = algo1_output['ho_occurred']
        
        scenario_id = int(algo1_output.get('scenario_id', 1))
        rlf_threshold = self._scenario_rlf_threshold_dbm(scenario_id)

        if rsrp_current < rlf_threshold:
            self.rlf_counter += 1
        else:
            self.rlf_counter = 0

        rlf_event = (self.rlf_counter >= 2)
        if rlf_event:
            self.rlf_counter = 0
        self.recent_rlf_events.append(1 if rlf_event else 0)
        
        handover_occurred = False
        ping_pong_event = False
        
        if ho_event and not self.in_handover:
            handover_occurred = True
            self.in_handover = True
        elif not ho_event:
            self.in_handover = False
        
        if handover_occurred and self.serving_cell is not None:
            if (self.prev_serving_cell == target_cell and
                self.serving_cell != self.prev_serving_cell and
                now_s - self.last_ho_time < 1.0):
                ping_pong_event = True
        
        self.recent_pp_events.append(1 if ping_pong_event else 0)
        self.recent_ho_events.append(1 if handover_occurred else 0)
        
        if self.sinr_prev is not None:
            sinr_delta = sinr_current - self.sinr_prev
        else:
            sinr_delta = 0.0
        
        recent_rlf_count = sum(self.recent_rlf_events)
        recent_pp_count = sum(self.recent_pp_events)
        recent_ho_count = sum(self.recent_ho_events)
        
        state = self.rl_module.build_state_vector(
            algo1_output,
            ttt_eff,
            hys_eff,
            self.time_since_last_ho,
            recent_rlf_count,
            recent_pp_count,
            recent_ho_count,
            self.rsrp_prev,
        )
        
        self.sinr_prev = sinr_current
        self.rsrp_prev = rsrp_current
        self.time_since_last_ho += 0.1
        
        if handover_occurred:
            self.prev_serving_cell = self.serving_cell
            self.serving_cell = target_cell
            self.target_cell = target_cell
            self.last_ho_time = now_s
            self.time_since_last_ho = 0.0
        
        self.param_history.append({
            'timestamp': now_s,
            'ttt_eff': ttt_eff,
            'hys_eff': hys_eff,
            'zone': zone,
            'velocity': velocity,
            'distance': distance,
            'rsrp': rsrp_current,
            'sinr': sinr_current
        })
        
        return state, rlf_event, ping_pong_event, sinr_delta, sinr_current, handover_occurred, recent_ho_count, done, velocity

    def step_n(self, ttt_eff, hys_eff, delta_ttt_step, delta_hys_db, no_op, n_steps: int,
               gamma: float = 0.99):
        """Advance environment by n_steps"""
        n_steps = int(max(1, n_steps))
        reward_accum = 0.0
        discount = 1.0
        rlf_count = 0
        pp_count = 0
        ho_count = 0
        steps_executed = 0
        state_next = None
        done = False
        velocity_avg = 0.0

        for micro in range(n_steps):
            state_next, rlf_event, ping_pong_event, sinr_delta, sinr_current, ho_occurred, recent_ho_count, done, velocity = \
                self.step(ttt_eff, hys_eff)

            steps_executed += 1
            velocity_avg = velocity  # Track velocity for reward

            if micro == 0:
                dt_step = delta_ttt_step
                dh_db = delta_hys_db
                no_op_flag = bool(no_op)
            else:
                dt_step = 0
                dh_db = 0.0
                no_op_flag = False

            # ✅ CRITICAL FIX #5: Pass velocity to compute_reward
            r_tick = self.rl_module.compute_reward(
                rlf_event,
                ping_pong_event,
                sinr_delta,
                sinr_current,
                ho_occurred,
                dt_step,
                dh_db,
                recent_ho_count,
                velocity=velocity_avg,  # ✅ NEW: Pass velocity
                no_op=no_op_flag,
            )

            reward_accum += discount * float(r_tick)
            discount *= float(gamma)

            rlf_count += 1 if rlf_event else 0
            pp_count += 1 if ping_pong_event else 0
            ho_count += 1 if ho_occurred else 0

            if done:
                break

        return (
            state_next,
            float(reward_accum),
            int(rlf_count),
            int(pp_count),
            int(ho_count),
            int(steps_executed),
            bool(done),
        )
    
    def reset(self):
        """Reset environment (UNCHANGED)"""
        self.algo1 = Algorithm1()
        self.time_since_last_ho = 0.0
        self.sinr_prev = None
        self.rsrp_prev = None
        self.prev_serving_cell = None
        self.serving_cell = None
        self.target_cell = None
        self.last_ho_time = -float('inf')
        self.rlf_counter = 0
        self.recent_rlf_events.clear()
        self.recent_pp_events.clear()
        self.recent_ho_events.clear()
        self.param_history = []
        self.in_handover = False
        
        algo1_output = self.ns3_env.reset(self.algo1)
        ttt_eff = 160
        hys_eff = 3.0
        state = self.rl_module.build_state_vector(algo1_output, ttt_eff, hys_eff, 0.0, 0, 0, 0, None)
        return state


# ============================================================================
# TRAINING LOOP (350 EPISODES - RLF FIX ONLY)
# ============================================================================

def train_ppo(agent, rl_module, training_env, num_episodes=350, rollout_horizon=2048,
              save_dir="models", log_dir="logs", control_interval_steps: int = 5):
    """
    Production training loop - 350 episodes
    
    ✅ CRITICAL FIX #5: Velocity-aware RLF penalty
    ✅ UNCHANGED: HO and PP logic (100% preserved)
    """
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    metrics_history = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print(f"PPO TRAINING START - RLF FIX VERSION - {num_episodes} EPISODES")
    print("=" * 80)
    print(f"✅ State dimension: 23 (unchanged)")
    print(f"✅ Action dimension: 15 (unchanged)")
    print(f"✅ CRITICAL FIX #5 APPLIED:")
    print(f"   └─ RLF penalty: -1.5 → -2.0 + velocity scaling")
    print(f"   └─ At 0 m/s: -2.0 (urban, normal)")
    print(f"   └─ At 50 m/s: -4.0 (train, aggressive)")
    print(f"   └─ At 70 m/s: -4.8 (high-speed, very aggressive)")
    print(f"\n✅ PRESERVED (0 changes):")
    print(f"   └─ HO penalty: -0.5 (unchanged)")
    print(f"   └─ PP penalty: -1.0 (unchanged)")
    print(f"   └─ HO freq penalty: -0.1 (unchanged)")
    print(f"   └─ All state logic (unchanged)")
    print("=" * 80)
    
    for episode in range(num_episodes):
        entropy_coef = rl_module.get_entropy_coef(episode, num_episodes)
        
        trajectories = []
        state = training_env.reset()
        
        episode_reward = 0.0
        episode_rlf = 0
        episode_pp = 0
        episode_ho = 0
        episode_length = 0
        
        ttt_rule = 160
        hys_rule = 3.0
        
        tick = 0
        control_interval_steps = int(max(1, control_interval_steps))

        while tick < rollout_horizon:
            action, log_prob, value = agent.select_action(state)

            delta_ttt, delta_hys = agent.decode_action(action)
            no_op = (delta_ttt == 0 and delta_hys == 0)

            ttt_eff, hys_eff = agent.compute_effective_parameters(
                ttt_rule, hys_rule, delta_ttt, delta_hys
            )

            steps_to_run = min(control_interval_steps, rollout_horizon - tick)

            state_next, reward, rlf_c, pp_c, ho_c, steps_executed, done = training_env.step_n(
                ttt_eff,
                hys_eff,
                delta_ttt,
                delta_hys,
                no_op,
                steps_to_run,
                gamma=agent.gamma,
            )

            trajectories.append({
                's': state,
                'a': action,
                'r': reward,
                'V': value,
                'log_prob': log_prob,
                'n_steps': steps_executed,
            })

            episode_reward += reward
            episode_rlf += int(rlf_c)
            episode_pp += int(pp_c)
            episode_ho += int(ho_c)
            episode_length += int(steps_executed)

            ttt_rule = ttt_eff
            hys_rule = hys_eff

            state = state_next
            tick += int(steps_executed)

            if done:
                break
        
        if done:
            bootstrap_value = 0.0
        else:
            _, _, bootstrap_value = agent.select_action(state)
        
        advantages = []
        returns = []
        gae = 0.0
        
        for i in range(len(trajectories) - 1, -1, -1):
            if i == len(trajectories) - 1:
                next_value = bootstrap_value
            else:
                next_value = trajectories[i + 1]['V']

            n = int(trajectories[i].get('n_steps', 1))
            gamma_n = agent.gamma ** n
            gae_discount = (agent.gamma * agent.gae_lambda) ** n

            delta = trajectories[i]['r'] + gamma_n * next_value - trajectories[i]['V']
            gae = delta + gae_discount * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + trajectories[i]['V'])
        
        advantages = np.array(advantages)
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages_normalized = (advantages - adv_mean) / adv_std
        
        for i, traj in enumerate(trajectories):
            traj['A_norm'] = advantages_normalized[i]
            traj['G'] = returns[i]
        
        agent.train_step(trajectories, entropy_coef)
        
        metrics = {
            'episode': episode + 1,
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'rlf_count': episode_rlf,
            'pp_count': episode_pp,
            'ho_count': episode_ho,
            'avg_reward_per_step': float(episode_reward / episode_length) if episode_length > 0 else 0.0,
            'entropy_coef': float(entropy_coef),
            'ttt_final': int(ttt_rule),
            'hys_final': float(hys_rule)
        }
        
        metrics_history.append(metrics)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | Reward: {metrics['total_reward']:8.2f} | "
                  f"RLF: {metrics['rlf_count']:2d} | PP: {metrics['pp_count']:2d} | "
                  f"HO: {metrics['ho_count']:3d} | TTT: {ttt_rule} | HYS: {hys_rule:.2f}")
            
            agent.save(f"{save_dir}/checkpoint_ep{episode + 1}")
    
    agent.save(f"{save_dir}/final")
    
    with open(f"{log_dir}/metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    with open(f"{log_dir}/param_history_{timestamp}.json", 'w') as f:
        json.dump(training_env.param_history, f, indent=2)
    
    print("=" * 80)
    print(f"TRAINING COMPLETE - {num_episodes} EPISODES WITH RLF FIX")
    print(f"Final model: {save_dir}/final")
    print(f"Logs: {log_dir}/metrics_{timestamp}.json")
    print("=" * 80)
    
    return metrics_history


def evaluate_agent(agent, rl_module, training_env, num_episodes=10, greedy=True, control_interval_steps: int = 5):
    """Evaluate trained agent (UNCHANGED)"""
    all_metrics = []
    
    print("\n" + "=" * 80)
    print("EVALUATION PHASE (RLF-FIXED AGENT)")
    print("=" * 80)
    
    for ep in range(num_episodes):
        state = training_env.reset()
        
        total_reward = 0.0
        rlf_count = 0
        pp_count = 0
        ho_count = 0
        
        ttt_rule = 160
        hys_rule = 3.0
        
        tick = 0
        control_interval_steps = int(max(1, control_interval_steps))

        while True:
            state_input = np.array(state, dtype=np.float32).reshape(1, -1)
            logits = agent.actor(state_input, training=False)[0].numpy()
            
            if greedy:
                action = np.argmax(logits)
            else:
                probs = tf.nn.softmax(logits).numpy()
                action = np.random.choice(agent.action_dim, p=probs)
            
            delta_ttt, delta_hys = agent.decode_action(action)
            no_op = (delta_ttt == 0 and delta_hys == 0)
            ttt_eff, hys_eff = agent.compute_effective_parameters(ttt_rule, hys_rule, delta_ttt, delta_hys)

            state_next, reward, rlf_c, pp_c, ho_c, steps_executed, done = training_env.step_n(
                ttt_eff,
                hys_eff,
                delta_ttt,
                delta_hys,
                no_op,
                control_interval_steps,
                gamma=agent.gamma,
            )

            total_reward += reward
            rlf_count += int(rlf_c)
            pp_count += int(pp_c)
            ho_count += int(ho_c)

            tick += int(steps_executed)
            ttt_rule = ttt_eff
            hys_rule = hys_eff
            state = state_next
            
            if done:
                break
        
        metrics = {
            'episode': ep + 1,
            'total_reward': total_reward,
            'rlf_count': rlf_count,
            'pp_count': pp_count,
            'ho_count': ho_count
        }
        all_metrics.append(metrics)
        
        print(f"Eval {ep + 1:2d} | Reward: {total_reward:8.2f} | "
              f"RLF: {rlf_count:2d} | PP: {pp_count:2d} | HO: {ho_count:3d}")
    
    print("=" * 80 + "\n")
    
    return all_metrics


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PPO PHASE 2 - RLF CRITICAL FIX ONLY (v3)")
    print("=" * 80)
    print("\n✅ CRITICAL FIX #5 APPLIED:")
    print("   Velocity-Aware RLF Penalty")
    print("")
    print("   Formula: r_rlf = -2.0 × (1 + velocity/50)")
    print("")
    print("   Examples:")
    print("   ├─ Pedestrian (1 m/s):  penalty = -2.04")
    print("   ├─ Car (15 m/s):        penalty = -2.6")
    print("   ├─ Bus (20 m/s):        penalty = -2.8")
    print("   ├─ Train (50 m/s):      penalty = -4.0")
    print("   └─ High-speed (70 m/s): penalty = -4.8")
    print("")
    print("✅ PRESERVED (NO CHANGES):")
    print("   ├─ HO penalty: -0.5")
    print("   ├─ PP penalty: -1.0")
    print("   ├─ State logic: 100% same")
    print("   ├─ Action space: 100% same")
    print("   └─ HO/PP control: 100% same")
    print("\n" + "=" * 80 + "\n")