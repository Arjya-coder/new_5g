import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

class Algorithm1Final:
    """
    Algorithm 1: Publication-Quality 5G Handover Engine
    
    Implements all improvements + final hardening + ultimate refinements.
    """
    
    def __init__(self):
        # ========== PERSISTENT STATE ==========
        self.rsrp_s_f = None
        self.sinr_s_f = None
        self.rsrp_n_f: Dict[int, float] = {}
        self.sinr_n_f: Dict[int, float] = {}
        
        self.a3_hold_ms: Dict[int, int] = {}
        self.a5_hold_ms: Dict[int, int] = {}
        
        self.previous_strongest_neighbor: Optional[int] = None
        self.strongest_neighbor_count: int = 0
        
        self.recent_handoffs: List[float] = []
        self.last_handover_time: float = -float('inf')
        self.serving_cell_entry_time: float = -float('inf')
        
        self.rsrp_history_serving: List[Tuple[float, float]] = []
        self.rsrp_history_neighbors: Dict[int, List[Tuple[float, float]]] = {}
        
        self.serving_slope_history: deque = deque(maxlen=7)
        self.serving_slope_confident = False
        
        self.handover_sequence: deque = deque(maxlen=10)
        
        self.cell_load: Dict[int, float] = {}
        self.cell_ho_activity: Dict[int, deque] = {}
        self.load_update_counter = 0
        
        self.last_ho_attempt_success = True
        self.failed_handovers_count = 0
        
        # ULTIMATE REFINEMENT 5: Measurement noise
        self.rng = np.random.RandomState(None)
        self.measurement_noise_sigma = 0.5  # dB
        
        # ========== CONFIGURATION CONSTANTS ==========
        self.EMA_ALPHA = 0.2
        
        self.DWELL_TIME_MS = 400
        self.SERVING_MIN_TIME_MS = 250
        
        self.TTT_MIN_MS = 200
        self.TTT_MAX_MS = 300
        self.HYS_MIN_DB = 3.0
        self.HYS_MAX_DB = 5.0
        
        self.T_PRED_S = 0.3
        self.W_HISTORY = 10
        self.DT_TICK_S = 0.1
        self.K_DISTANCE_MAX = 1500
        
        self.BEST_NEIGHBOR_PERSISTENCE_TICKS = 2
        
        self.RLF_THRESHOLD_DBM = -119.0
        self.POOR_SIGNAL_THRESHOLD_DBM = -118.0
        self.POOR_SINR_THRESHOLD_DB = -10.0
        self.POOR_CQI_THRESHOLD = 5
        
        self.GOOD_RSRP_THRESHOLD_DBM = -100.0
        self.GOOD_SINR_THRESHOLD_DB = 0.0
        self.GOOD_CQI_THRESHOLD = 10
        
        self.CELL_CENTER_THRESHOLD_M = 350
        self.HANDOFF_ZONE_THRESHOLD_M = 480
        self.CELL_RADIUS_M = 500
        
        self.PING_PONG_HARD_BLOCK_THRESHOLD = 0.7
        self.PING_PONG_SOFT_PENALTY_THRESHOLD = 0.4
        self.PING_PONG_DETECTION_WINDOW_S = 60.0
        
        self.RLF_RISK_THRESHOLD = 0.50
        self.MODERATE_RLF_RISK = 0.35
        self.MILD_RLF_RISK = 0.25
        self.A5_SERVING_THRESHOLD_DBM = -110
        self.A5_NEIGHBOR_THRESHOLD_DBM = -108
        
        self.HIGH_MOBILITY_THRESHOLD_MPS = 10.0
        
        self.CRITICAL_SLOPE_THRESHOLD_DBS = -5.0
        self.MODERATE_SLOPE_THRESHOLD_DBS = -2.0
        self.SLOPE_RISK_CRITICAL = 0.25
        self.SLOPE_RISK_MODERATE = 0.10
        self.URGENCY_SLOPE_THRESHOLD_DBS = -3.0
        
        self.HO_GAIN_MARGIN_BONUS_DB = 1.5
        
        self.DISTANCE_PENALTY_MAX_DB = 5.0
        
        self.LOAD_PENALTY_MAX_DB = 1.0
        self.LOAD_CHANGE_RATE = 0.05
        self.LOAD_PENALTY_ENABLED = True
        
        self.PING_PONG_SEQUENCE_WINDOW_S = 2.0
        self.PING_PONG_OSCILLATION_THRESHOLD = 3
        
        # ULTIMATE REFINEMENT 1: HO failure thresholds
        self.HO_FAILURE_SINR_THRESHOLD_DB = -12.0
        self.HO_FAILURE_RSRP_THRESHOLD_DBM = -125.0
        self.HO_FAILURE_BASE_PROBABILITY = 0.05
        self.HO_FAILURE_PROBABILITY_THRESHOLD = 0.3
        
        self.LOAD_ACTIVITY_WINDOW_S = 30.0
        self.LOAD_ACTIVITY_WEIGHT = 0.3
        
        # ULTIMATE REFINEMENT 3: Continuous SINR penalty
        self.SINR_PENALTY_SCALE = 1.0 / 10.0
        
        # ULTIMATE REFINEMENT 4: Zone-based bias
        self.ZONE_BIAS_CELL_CENTER = -0.5  # Discourage HO in cell center
        self.ZONE_BIAS_HANDOFF_ZONE = 0.0  # Neutral
        self.ZONE_BIAS_CRITICAL_ZONE = 0.5  # Encourage HO in critical zone
        self.ZONE_BIAS_EDGE_ZONE = 1.0  # Strong encourage HO in edge

    def _update_history(self, now_s: float, val: float, 
                       history_list: List[Tuple[float, float]]) -> None:
        history_list.append((now_s, val))
        while len(history_list) > 0 and now_s - history_list[0][0] > 5.0:
            history_list.pop(0)

    def _clamp_parameter(self, value: float, min_val: float, 
                        max_val: float, param_name: str) -> float:
        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value

    def _compute_serving_slope_improved(self) -> Tuple[float, bool]:
        slope_dbs = 0.0
        slope_confident = False
        
        min_samples = 5
        
        if len(self.rsrp_history_serving) >= min_samples:
            recent_rsrps = [val for _, val in self.rsrp_history_serving[-min_samples:]]
            recent_times = [t for t, _ in self.rsrp_history_serving[-min_samples:]]
            
            if len(recent_rsrps) >= min_samples:
                n = len(recent_rsrps)
                t_mean = sum(recent_times) / n
                rsrp_mean = sum(recent_rsrps) / n
                
                numerator = sum((recent_times[i] - t_mean) * (recent_rsrps[i] - rsrp_mean)
                              for i in range(n))
                denominator = sum((recent_times[i] - t_mean) ** 2 for i in range(n))
                
                if denominator > 1e-6:
                    slope_dbs = numerator / denominator
                    slope_confident = True
                
                self.serving_slope_history.append(slope_dbs)

        return slope_dbs, slope_confident

    def _detect_ping_pong_sequence(self, now_s: float) -> Tuple[float, int]:
        recent_hos = [ho for ho in self.recent_handoffs 
                     if now_s - ho <= self.PING_PONG_SEQUENCE_WINDOW_S]
        
        if len(recent_hos) < 3:
            return 0.0, 0
        
        oscillation_count = 0
        
        if len(self.handover_sequence) >= 3:
            recent_seq = list(self.handover_sequence)[-5:]
            
            for i in range(len(recent_seq) - 2):
                time_1, from_1, to_1 = recent_seq[i]
                time_2, from_2, to_2 = recent_seq[i + 1]
                time_3, from_3, to_3 = recent_seq[i + 2]
                
                if from_1 == to_3 and to_1 == from_2:
                    time_gap = time_3 - time_1
                    if time_gap <= self.PING_PONG_SEQUENCE_WINDOW_S:
                        oscillation_count += 1
        
        if oscillation_count >= self.PING_PONG_OSCILLATION_THRESHOLD:
            sequence_risk = 1.0
        elif oscillation_count >= 2:
            sequence_risk = 0.8
        elif oscillation_count >= 1:
            sequence_risk = 0.6
        else:
            sequence_risk = 0.0
        
        return sequence_risk, oscillation_count

    def _compute_ho_failure_probability(self, target_rsrp_dbm: float, 
                                       target_sinr_db: float) -> float:
        failure_prob = self.HO_FAILURE_BASE_PROBABILITY
        
        if target_rsrp_dbm < self.HO_FAILURE_RSRP_THRESHOLD_DBM:
            failure_prob += 0.20
        elif target_rsrp_dbm < -120:
            failure_prob += 0.10
        
        if target_sinr_db < self.HO_FAILURE_SINR_THRESHOLD_DB:
            failure_prob += 0.15
        elif target_sinr_db < -10:
            failure_prob += 0.08
        
        failure_prob = min(failure_prob, 1.0)
        return failure_prob

    def _update_activity_based_load(self, neighbor_ids: List[int], now_s: float) -> None:
        if not self.LOAD_PENALTY_ENABLED:
            return
        
        for nid in neighbor_ids:
            if nid not in self.cell_ho_activity:
                self.cell_ho_activity[nid] = deque(maxlen=20)
            if nid not in self.cell_load:
                self.cell_load[nid] = 0.05
        
        for nid in list(self.cell_ho_activity.keys()):
            valid_hos = [t for t in self.cell_ho_activity[nid] 
                        if now_s - t <= self.LOAD_ACTIVITY_WINDOW_S]
            self.cell_ho_activity[nid] = deque(valid_hos, maxlen=20)
        
        for nid in neighbor_ids:
            recent_ho_count = len(self.cell_ho_activity[nid])
            activity_load = min(recent_ho_count * 0.05, 0.3)
            base_load = 0.05
            self.cell_load[nid] = (self.LOAD_ACTIVITY_WEIGHT * activity_load +
                                  (1 - self.LOAD_ACTIVITY_WEIGHT) * base_load)
        
        for nid in list(self.cell_load.keys()):
            if nid not in neighbor_ids:
                del self.cell_load[nid]
                self.cell_ho_activity.pop(nid, None)

    def _compute_relative_load_penalty(self, candidate_idx: int, 
                                      neighbor_ids: List[int],
                                      eligible_candidates: List[int]) -> float:
        """
        ULTIMATE REFINEMENT 2: Relative load comparison
        Compare candidate's load to average load of neighbors.
        """
        if not self.LOAD_PENALTY_ENABLED or len(eligible_candidates) == 0:
            return 0.0
        
        nid = neighbor_ids[candidate_idx]
        candidate_load = self.cell_load.get(nid, 0.05)
        
        # Compute average load of all eligible candidates
        eligible_loads = [self.cell_load.get(neighbor_ids[i], 0.05) 
                         for i in eligible_candidates]
        avg_load = sum(eligible_loads) / len(eligible_loads) if eligible_loads else 0.05
        
        # Relative penalty: higher if candidate load > average
        load_diff = candidate_load - avg_load
        if load_diff > 0:
            relative_penalty = min(load_diff * 10.0, self.LOAD_PENALTY_MAX_DB)
        else:
            relative_penalty = 0.0
        
        return relative_penalty

    def _compute_continuous_sinr_penalty(self, target_sinr_db: float) -> float:
        """
        ULTIMATE REFINEMENT 3: Continuous SINR-based penalty
        Instead of threshold-based steps, use continuous model.
        penalty = max(0, -sinr / 10)
        """
        if target_sinr_db >= 0:
            return 0.0
        sinr_penalty = max(0.0, -target_sinr_db * self.SINR_PENALTY_SCALE)
        return min(sinr_penalty, 3.0)  # Cap at 3 dB

    def _get_zone_bias(self, zone: str) -> float:
        """
        ULTIMATE REFINEMENT 4: Zone-based bias
        Use zone classification to guide decisions.
        """
        if zone == "CELL_CENTER":
            return self.ZONE_BIAS_CELL_CENTER
        elif zone == "HANDOFF_ZONE":
            return self.ZONE_BIAS_HANDOFF_ZONE
        elif zone == "CRITICAL_ZONE":
            return self.ZONE_BIAS_CRITICAL_ZONE
        elif zone == "EDGE_ZONE":
            return self.ZONE_BIAS_EDGE_ZONE
        return 0.0

    def _rank_candidates_with_all_penalties(self, 
                                           eligible_candidates: List[int],
                                           neighbor_ids: List[int],
                                           distance_neighbors: List[float],
                                           sinr_neighbors: List[float],
                                           margin_pred: Dict[int, float],
                                           zone: str) -> None:
        """
        Apply all ranking penalties including:
        - SINR penalty (continuous model)
        - Distance penalty
        - Relative load penalty
        - Zone-based bias
        - Measurement noise
        """
        zone_bias = self._get_zone_bias(zone)
        
        for i in eligible_candidates:
            nid = neighbor_ids[i]
            
            # ULTIMATE REFINEMENT 3: Continuous SINR penalty
            if i < len(sinr_neighbors):
                target_sinr = sinr_neighbors[i]
                sinr_penalty = self._compute_continuous_sinr_penalty(target_sinr)
            else:
                sinr_penalty = 0.0
            
            margin_pred[i] -= sinr_penalty
            
            # Distance penalty
            distance_normalized = min(distance_neighbors[i] / 1500.0, 1.0)
            distance_penalty = distance_normalized * self.DISTANCE_PENALTY_MAX_DB
            margin_pred[i] -= distance_penalty
            
            # ULTIMATE REFINEMENT 2: Relative load penalty
            relative_load_penalty = self._compute_relative_load_penalty(
                i, neighbor_ids, eligible_candidates)
            margin_pred[i] -= relative_load_penalty
            
            # ULTIMATE REFINEMENT 4: Zone-based bias
            margin_pred[i] += zone_bias
            
            # ULTIMATE REFINEMENT 5: Measurement noise
            noise = self.rng.normal(0, self.measurement_noise_sigma)
            margin_pred[i] += noise

    def step(self,
             rsrp_serving: float,
             sinr_serving: float,
             cqi_serving: int,
             distance_serving: float,
             rsrp_neighbors: List[float],
             neighbor_ids: List[int],
             distance_neighbors: List[float],
             sinr_neighbors: List[float],
             velocity: float,
             now_s: float,
             TTT_eff: int,
             HYS_eff: float,
             los_probability: float = 1.0) -> Dict:
        
        TTT_eff = int(self._clamp_parameter(TTT_eff, self.TTT_MIN_MS, 
                                           self.TTT_MAX_MS, "TTT_eff"))
        HYS_eff = self._clamp_parameter(HYS_eff, self.HYS_MIN_DB, 
                                       self.HYS_MAX_DB, "HYS_eff")
        
        K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors), 
               len(sinr_neighbors))
        rsrp_neighbors = rsrp_neighbors[:K]
        neighbor_ids = neighbor_ids[:K]
        distance_neighbors = distance_neighbors[:K]
        sinr_neighbors = sinr_neighbors[:K]
        
        for nid in list(self.rsrp_n_f.keys()):
            if nid not in neighbor_ids:
                del self.rsrp_n_f[nid]
                self.sinr_n_f.pop(nid, None)
                self.a3_hold_ms.pop(nid, None)
                self.a5_hold_ms.pop(nid, None)
                self.rsrp_history_neighbors.pop(nid, None)
                self.cell_load.pop(nid, None)
                self.cell_ho_activity.pop(nid, None)

        if self.rsrp_s_f is None:
            self.rsrp_s_f = rsrp_serving
            self.sinr_s_f = sinr_serving
            self.serving_cell_entry_time = now_s
        else:
            self.rsrp_s_f = self.EMA_ALPHA * rsrp_serving + (1 - self.EMA_ALPHA) * self.rsrp_s_f
            self.sinr_s_f = self.EMA_ALPHA * sinr_serving + (1 - self.EMA_ALPHA) * self.sinr_s_f
        
        for i, nid in enumerate(neighbor_ids):
            if nid not in self.rsrp_n_f:
                self.rsrp_n_f[nid] = rsrp_neighbors[i]
                self.sinr_n_f[nid] = sinr_neighbors[i]
                self.a3_hold_ms[nid] = 0
                self.a5_hold_ms[nid] = 0
                self.rsrp_history_neighbors[nid] = []
            else:
                self.rsrp_n_f[nid] = (self.EMA_ALPHA * rsrp_neighbors[i] + 
                                     (1 - self.EMA_ALPHA) * self.rsrp_n_f[nid])
                self.sinr_n_f[nid] = (self.EMA_ALPHA * sinr_neighbors[i] + 
                                     (1 - self.EMA_ALPHA) * self.sinr_n_f[nid])
        
        self._update_history(now_s, self.rsrp_s_f, self.rsrp_history_serving)
        for nid in neighbor_ids:
            self._update_history(now_s, self.rsrp_n_f[nid], 
                               self.rsrp_history_neighbors[nid])

        if (self.rsrp_s_f < self.POOR_SIGNAL_THRESHOLD_DBM or
            self.sinr_s_f < self.POOR_SINR_THRESHOLD_DB or
            cqi_serving < self.POOR_CQI_THRESHOLD):
            signal_quality = "POOR"
        elif (self.rsrp_s_f >= self.GOOD_RSRP_THRESHOLD_DBM and
              self.sinr_s_f >= self.GOOD_SINR_THRESHOLD_DB and
              cqi_serving >= self.GOOD_CQI_THRESHOLD):
            signal_quality = "GOOD"
        else:
            signal_quality = "FAIR"

        if distance_serving < self.CELL_CENTER_THRESHOLD_M:
            zone = "CELL_CENTER"
        elif distance_serving < self.HANDOFF_ZONE_THRESHOLD_M:
            zone = "HANDOFF_ZONE"
        elif distance_serving <= self.CELL_RADIUS_M:
            zone = "CRITICAL_ZONE"
        else:
            zone = "EDGE_ZONE"

        serving_slope_dbs, slope_confident = self._compute_serving_slope_improved()

        rlf_risk = 0.0
        
        if self.rsrp_s_f < self.RLF_THRESHOLD_DBM:
            rlf_risk += 0.30
        elif self.rsrp_s_f < -110:
            rlf_risk += 0.15
        
        if self.sinr_s_f < self.POOR_SINR_THRESHOLD_DB:
            rlf_risk += 0.30
        elif self.sinr_s_f < -5:
            rlf_risk += 0.15
        
        if distance_serving >= self.HANDOFF_ZONE_THRESHOLD_M:
            rlf_risk += 0.15
        
        slope_risk_increment = 0.0
        if slope_confident:
            if serving_slope_dbs < self.CRITICAL_SLOPE_THRESHOLD_DBS:
                slope_risk_increment = self.SLOPE_RISK_CRITICAL
            elif serving_slope_dbs < self.MODERATE_SLOPE_THRESHOLD_DBS:
                slope_risk_increment = self.SLOPE_RISK_MODERATE
        
        rlf_risk += slope_risk_increment
        rlf_risk = min(rlf_risk, 1.0)

        self.recent_handoffs = [t for t in self.recent_handoffs 
                               if now_s - t <= self.PING_PONG_DETECTION_WINDOW_S]
        handoff_count_60s = len(self.recent_handoffs)
        
        if handoff_count_60s == 0:
            freq_ping_pong_risk = 0.0
        elif handoff_count_60s <= 2:
            freq_ping_pong_risk = 0.10
        elif handoff_count_60s <= 5:
            freq_ping_pong_risk = 0.40
        elif handoff_count_60s <= 10:
            freq_ping_pong_risk = 0.70
        else:
            freq_ping_pong_risk = 1.0
        
        seq_ping_pong_risk, oscillation_count = self._detect_ping_pong_sequence(now_s)
        
        ping_pong_risk = max(freq_ping_pong_risk, seq_ping_pong_risk)

        in_dwell = (now_s - self.last_handover_time) * 1000 < self.DWELL_TIME_MS
        in_serving_grace = (now_s - self.serving_cell_entry_time) * 1000 < self.SERVING_MIN_TIME_MS
        non_emergency_allowed = (ping_pong_risk < self.PING_PONG_SOFT_PENALTY_THRESHOLD) and not in_dwell

        MUST_HO = (signal_quality == "POOR" or
                   zone == "EDGE_ZONE" or
                   rlf_risk >= self.RLF_RISK_THRESHOLD)

        self._update_activity_based_load(neighbor_ids, now_s)

        eligible_candidates = []
        
        if len(neighbor_ids) > 0:
            current_strongest = max(neighbor_ids, 
                                   key=lambda nid: self.rsrp_n_f.get(nid, -200))
        else:
            current_strongest = None
        
        if current_strongest == self.previous_strongest_neighbor:
            self.strongest_neighbor_count += 1
        else:
            self.strongest_neighbor_count = 1
            self.previous_strongest_neighbor = current_strongest
        
        for i, nid in enumerate(neighbor_ids):
            if distance_neighbors[i] >= self.K_DISTANCE_MAX:
                continue
            
            A3_condition = (self.rsrp_n_f[nid] - self.rsrp_s_f) >= HYS_eff
            A5_condition = (self.rsrp_s_f <= self.A5_SERVING_THRESHOLD_DBM and
                           self.rsrp_n_f[nid] >= self.A5_NEIGHBOR_THRESHOLD_DBM)
            
            if A3_condition:
                self.a3_hold_ms[nid] += int(self.DT_TICK_S * 1000)
            else:
                self.a3_hold_ms[nid] = 0
            
            if A5_condition:
                self.a5_hold_ms[nid] += int(self.DT_TICK_S * 1000)
            else:
                self.a5_hold_ms[nid] = 0
            
            ttt_elapsed = (self.a3_hold_ms[nid] >= TTT_eff or
                          self.a5_hold_ms[nid] >= TTT_eff)
            
            is_persistent_best = (nid == current_strongest and
                                 self.strongest_neighbor_count >= 
                                 self.BEST_NEIGHBOR_PERSISTENCE_TICKS)
            
            if ttt_elapsed and is_persistent_best:
                eligible_candidates.append(i)

        margin_pred = {}
        
        if len(self.rsrp_history_serving) >= self.W_HISTORY:
            time_oldest, rsrp_s_oldest = self.rsrp_history_serving[-self.W_HISTORY]
            time_current = self.rsrp_history_serving[-1][0]
            time_span = max(time_current - time_oldest, 0.001)
            
            slope_s = (self.rsrp_s_f - rsrp_s_oldest) / time_span
            
            for i in eligible_candidates:
                nid = neighbor_ids[i]
                history_n = self.rsrp_history_neighbors[nid]
                
                if len(history_n) >= self.W_HISTORY:
                    time_n_old, rsrp_n_old = history_n[-self.W_HISTORY]
                    time_n_span = max(time_current - time_n_old, 0.001)
                    
                    slope_n = (self.rsrp_n_f[nid] - rsrp_n_old) / time_n_span
                    margin_delta = (slope_n - slope_s) * self.T_PRED_S
                    margin_pred[i] = (self.rsrp_n_f[nid] - self.rsrp_s_f) + margin_delta
                else:
                    margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
        else:
            for i in eligible_candidates:
                nid = neighbor_ids[i]
                margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
        
        self._rank_candidates_with_all_penalties(eligible_candidates, neighbor_ids,
                                                distance_neighbors, sinr_neighbors, 
                                                margin_pred, zone)
        
        eligible_candidates.sort(key=lambda i: (
            -margin_pred.get(i, -999),
            -self.rsrp_n_f[neighbor_ids[i]],
            distance_neighbors[i]
        ))

        action = 0
        target_cell_id = None
        confidence = max(0.4, min(1.0, 1.0 - (rlf_risk + ping_pong_risk) / 2.0))
        reason = "No HO trigger"
        ho_failure_probability = 0.0
        
        def finalize_ho(trg_idx: int, conf: float, rsn: str) -> None:
            nonlocal action, target_cell_id, confidence, reason, ho_failure_probability
            action = trg_idx + 1
            target_cell_id = neighbor_ids[trg_idx]
            confidence = conf
            reason = rsn
            
            target_rsrp = self.rsrp_n_f[target_cell_id]
            target_sinr = sinr_neighbors[trg_idx] if trg_idx < len(sinr_neighbors) else 0.0
            ho_failure_probability = self._compute_ho_failure_probability(target_rsrp, target_sinr)
            
            if target_cell_id not in self.cell_ho_activity:
                self.cell_ho_activity[target_cell_id] = deque(maxlen=20)
            self.cell_ho_activity[target_cell_id].append(now_s)
            
            self.recent_handoffs.append(now_s)
            
            if len(self.handover_sequence) > 0:
                _, _, last_to_cell = self.handover_sequence[-1]
                from_cell = last_to_cell
            else:
                from_cell = 0
            
            self.handover_sequence.append((now_s, from_cell, target_cell_id))
            
            self.last_handover_time = now_s
            self.serving_cell_entry_time = now_s
            self.strongest_neighbor_count = 0
            self.previous_strongest_neighbor = None

        if (ping_pong_risk >= self.PING_PONG_HARD_BLOCK_THRESHOLD and 
            not MUST_HO):
            reason = (f"BLOCKED: Extreme ping-pong activity "
                     f"({handoff_count_60s} HOs/60s). System must stabilize.")
            return self._build_return(0, None, 0.1, reason, 
                                    rlf_risk, ping_pong_risk, signal_quality, zone,
                                    eligible_candidates, self.rsrp_s_f, self.sinr_s_f,
                                    in_dwell, in_serving_grace, serving_slope_dbs,
                                    TTT_eff, HYS_eff, slope_confident, 0.0)

        if MUST_HO:
            if eligible_candidates:
                target_idx = eligible_candidates[0]
                target_rsrp = self.rsrp_n_f[neighbor_ids[target_idx]]
                target_sinr = sinr_neighbors[target_idx] if target_idx < len(sinr_neighbors) else 0.0
                
                # ULTIMATE REFINEMENT 1: Check HO failure probability
                ho_fail_prob = self._compute_ho_failure_probability(target_rsrp, target_sinr)
                
                base_confidence = 0.90
                
                if ho_fail_prob > self.HO_FAILURE_PROBABILITY_THRESHOLD:
                    conf = base_confidence * 0.70
                    rsn = f"MUST-HO: RLF imminent (HO failure prob {ho_fail_prob:.2f}, caution advised)"
                elif ping_pong_risk >= self.PING_PONG_HARD_BLOCK_THRESHOLD:
                    conf = base_confidence * 0.75
                    rsn = "MUST-HO: Critical RLF imminent (extreme ping-pong caution)"
                elif ping_pong_risk >= self.PING_PONG_SOFT_PENALTY_THRESHOLD:
                    conf = base_confidence * 0.85
                    rsn = "MUST-HO: Critical RLF imminent (moderate ping-pong caution)"
                else:
                    conf = base_confidence
                    rsn = "MUST-HO: Critical signal loss or RLF imminent"
                
                finalize_ho(target_idx, conf, rsn)
            else:
                if len(neighbor_ids) > 0:
                    best_fallback_idx = max(range(len(neighbor_ids)),
                                          key=lambda i: self.rsrp_n_f[neighbor_ids[i]])
                    base_confidence = 0.70
                    rsn = "MUST-HO: Emergency fallback (ignoring TTT)"
                    finalize_ho(best_fallback_idx, base_confidence, rsn)
                else:
                    reason = "MUST-HO triggered but NO candidates available"
                    confidence = 0.2
        
        elif (non_emergency_allowed and
              not in_serving_grace and
              rlf_risk >= self.MODERATE_RLF_RISK and
              eligible_candidates):
            
            target_idx = eligible_candidates[0]
            predicted_margin = margin_pred.get(target_idx, 0)
            
            urgency_boost = 0.0
            if serving_slope_dbs < self.URGENCY_SLOPE_THRESHOLD_DBS and slope_confident:
                urgency_boost = 0.1
            
            min_gain_db = HYS_eff + self.HO_GAIN_MARGIN_BONUS_DB
            
            if predicted_margin < min_gain_db:
                reason = (f"Skip HO: Margin {predicted_margin:.1f} dB < "
                         f"threshold {min_gain_db:.1f} dB (cost not justified)")
            else:
                target_rsrp = self.rsrp_n_f[neighbor_ids[target_idx]]
                target_sinr = sinr_neighbors[target_idx] if target_idx < len(sinr_neighbors) else 0.0
                ho_fail_prob = self._compute_ho_failure_probability(target_rsrp, target_sinr)
                
                # ULTIMATE REFINEMENT 1: Consider HO failure probability
                if ho_fail_prob > self.HO_FAILURE_PROBABILITY_THRESHOLD:
                    confidence = confidence * (1.0 - ho_fail_prob)
                    reason = f"Proactive HO at risk: failure prob {ho_fail_prob:.2f}, reduced confidence"
                else:
                    base_confidence = 0.82 + urgency_boost
                    
                    if ping_pong_risk >= self.PING_PONG_SOFT_PENALTY_THRESHOLD:
                        confidence = base_confidence * 0.80
                        rsn = f"Proactive HO: RLF risk elevated (moderate ping-pong caution)"
                    else:
                        confidence = base_confidence
                        rsn = (f"Proactive HO: RLF risk elevated "
                              f"(margin {predicted_margin:.1f} dB)")
                    reason = rsn
                    finalize_ho(target_idx, confidence, rsn)
                    return self._build_return(action, target_cell_id, confidence, reason,
                                            rlf_risk, ping_pong_risk, signal_quality, zone,
                                            eligible_candidates, self.rsrp_s_f, self.sinr_s_f,
                                            in_dwell, in_serving_grace, serving_slope_dbs,
                                            TTT_eff, HYS_eff, slope_confident, ho_failure_probability)
        
        elif (non_emergency_allowed and
              not in_serving_grace and
              velocity >= self.HIGH_MOBILITY_THRESHOLD_MPS and
              zone == "CRITICAL_ZONE" and
              rlf_risk >= self.MILD_RLF_RISK and
              eligible_candidates):
            
            target_idx = eligible_candidates[0]
            predicted_margin = margin_pred.get(target_idx, 0)
            
            urgency_boost = 0.0
            if serving_slope_dbs < self.URGENCY_SLOPE_THRESHOLD_DBS and slope_confident:
                urgency_boost = 0.1
            
            min_gain_db = HYS_eff + self.HO_GAIN_MARGIN_BONUS_DB
            
            if predicted_margin >= min_gain_db:
                target_rsrp = self.rsrp_n_f[neighbor_ids[target_idx]]
                target_sinr = sinr_neighbors[target_idx] if target_idx < len(sinr_neighbors) else 0.0
                ho_fail_prob = self._compute_ho_failure_probability(target_rsrp, target_sinr)
                
                if ho_fail_prob > self.HO_FAILURE_PROBABILITY_THRESHOLD:
                    base_confidence = 0.78 + urgency_boost
                    conf = base_confidence * (1.0 - ho_fail_prob)
                    rsn = ("Proactive HO: High mobility in critical zone "
                          f"(failure risk {ho_fail_prob:.2f})")
                else:
                    base_confidence = 0.78 + urgency_boost
                    rsn = ("Proactive HO: High mobility in critical zone "
                          "with signal degradation")
                    conf = base_confidence
                
                finalize_ho(target_idx, conf, rsn)
            else:
                reason = (f"Skip HO: Margin too small for high-mobility case "
                         f"({predicted_margin:.1f} < {min_gain_db:.1f} dB)")
        
        elif (non_emergency_allowed and
              not in_serving_grace and
              signal_quality == "FAIR" and
              self.MILD_RLF_RISK <= rlf_risk < self.MODERATE_RLF_RISK and
              eligible_candidates):
            
            target_idx = eligible_candidates[0]
            predicted_margin = margin_pred.get(target_idx, 0)
            
            urgency_boost = 0.0
            if serving_slope_dbs < self.URGENCY_SLOPE_THRESHOLD_DBS and slope_confident:
                urgency_boost = 0.1
            
            min_gain_db = HYS_eff + self.HO_GAIN_MARGIN_BONUS_DB
            
            if predicted_margin >= min_gain_db:
                target_rsrp = self.rsrp_n_f[neighbor_ids[target_idx]]
                target_sinr = sinr_neighbors[target_idx] if target_idx < len(sinr_neighbors) else 0.0
                ho_fail_prob = self._compute_ho_failure_probability(target_rsrp, target_sinr)
                
                if ho_fail_prob > self.HO_FAILURE_PROBABILITY_THRESHOLD:
                    base_confidence = 0.75 + urgency_boost
                    conf = base_confidence * (1.0 - ho_fail_prob)
                    rsn = f"Proactive HO: Fair signal + RLF risk (failure risk {ho_fail_prob:.2f})"
                else:
                    base_confidence = 0.75 + urgency_boost
                    rsn = "Proactive HO: Fair signal with moderate RLF risk"
                    conf = base_confidence
                
                finalize_ho(target_idx, conf, rsn)
            else:
                reason = (f"Skip HO: Margin insufficient for fair-signal case "
                         f"({predicted_margin:.1f} < {min_gain_db:.1f} dB)")
        
        self.previous_strongest_neighbor = current_strongest
        
        return self._build_return(action, target_cell_id, confidence, reason,
                                rlf_risk, ping_pong_risk, signal_quality, zone,
                                eligible_candidates, self.rsrp_s_f, self.sinr_s_f,
                                in_dwell, in_serving_grace, serving_slope_dbs,
                                TTT_eff, HYS_eff, slope_confident, ho_failure_probability)

    def compute_reward(self, decision_dict: Dict, 
                      rlf_occurred: bool = False, 
                      ping_pong_occurred: bool = False) -> float:
        """
        ULTIMATE REFINEMENT 6: Reward function for RL Phase
        
        Penalize:
        - Unnecessary handovers
        - Ping-pong events
        - RLF events
        
        Reward:
        - Stable connections
        - Good signal quality
        """
        reward = 0.0
        
        # Reward for good signal quality
        if decision_dict['signal_quality'] == "GOOD":
            reward += 1.0
        elif decision_dict['signal_quality'] == "FAIR":
            reward += 0.5
        else:
            reward -= 1.0
        
        # Reward for stable connection (no HO)
        if decision_dict['action'] == 0:
            reward += 0.5
        else:
            # Penalty for handover (has cost)
            reward -= 0.3
        
        # Penalty for RLF
        if rlf_occurred:
            reward -= 5.0
        
        # Penalty for ping-pong
        if ping_pong_occurred:
            reward -= 2.0
        
        # Penalty for high RLF risk without action
        if decision_dict['rlf_risk'] > 0.5 and decision_dict['action'] == 0:
            reward -= 1.0
        
        # Penalty for HO failure probability
        if decision_dict['action'] > 0:
            ho_fail_prob = decision_dict['ho_failure_probability']
            reward -= ho_fail_prob * 2.0
        
        return reward

    def _build_return(self,
                     action: int,
                     target_cell_id: Optional[int],
                     confidence: float,
                     reason: str,
                     rlf_risk: float,
                     ping_pong_risk: float,
                     signal_quality: str,
                     zone: str,
                     eligible_candidates: List[int],
                     rsrp_serving: float,
                     sinr_serving: float,
                     in_dwell: bool,
                     in_serving_grace: bool,
                     serving_slope_dbs: float,
                     ttt_ms: int,
                     hys_db: float,
                     slope_confident: bool,
                     ho_failure_probability: float) -> Dict:
        return {
            "action": action,
            "target_cell_id": target_cell_id,
            "confidence": confidence,
            "reason": reason,
            
            "rlf_risk": rlf_risk,
            "ping_pong_risk": ping_pong_risk,
            
            "signal_quality": signal_quality,
            "zone": zone,
            "rsrp_serving_dbm": rsrp_serving,
            "sinr_serving_db": sinr_serving,
            
            "serving_slope_dbs": serving_slope_dbs,
            "slope_confident": slope_confident,
            
            "in_dwell": in_dwell,
            "in_serving_grace": in_serving_grace,
            
            "ttt_ms_effective": ttt_ms,
            "hys_db_effective": hys_db,
            
            "eligible_candidates": eligible_candidates,
            "ho_failure_probability": ho_failure_probability,
        }