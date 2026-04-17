import numpy as np

class Algorithm1:
    """
    Algorithm 1: Clean, Minimal 5G Handover Baseline
    
    Designed as a realistic but slightly suboptimal baseline for RL comparison.
    
    Key characteristics:
    - EMA filtering (α = 0.2)
    - A3/A5 event detection with TTT
    - Dwell time constraint (0.375s)
    - Simple ping-pong frequency tracking
    - Slope-based margin prediction
    - Fixed decision logic (no optimization)
    
    Input requirements:
    - HYS_eff must be in [4.0, 5.0] dB
    - TTT_eff must be in [200, 300] ms
    """
    
    def __init__(self):
        # Persistent state
        self.rsrp_s_f = None
        self.sinr_s_f = None
        self.rsrp_n_f = {}
        
        self.a3_hold_ms = {}
        self.a5_hold_ms = {}
        self.previous_strongest_neighbor = None
        
        self.recent_handoffs = []
        self.last_handover_time = -float('inf')
        
        # History buffers for slope prediction
        self.rsrp_history_serving = []
        self.rsrp_history_neighbors = {}

    def _update_history(self, now_s, val, history_list):
        """Update history buffer, keep last 5 seconds."""
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
        """
        Executes one 100ms tick of Algorithm 1 Baseline.
        
        Parameters:
        - TTT_eff: Time-to-trigger in ms, must be in [200, 300]
        - HYS_eff: Hysteresis in dB, must be in [4.0, 5.0]
        
        Returns:
        Dictionary with action, target_cell_id, confidence, reason, and diagnostics.
        """
        dt_s = 0.1
        
        # Step 0: Sanitize Neighbor Lists
        K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
        rsrp_neighbors = rsrp_neighbors[:K]
        neighbor_ids = neighbor_ids[:K]
        distance_neighbors = distance_neighbors[:K]

        # Cleanup disappeared neighbors
        for nid in list(self.rsrp_n_f.keys()):
            if nid not in neighbor_ids:
                del self.rsrp_n_f[nid]
                self.a3_hold_ms.pop(nid, None)
                self.a5_hold_ms.pop(nid, None)
                self.rsrp_history_neighbors.pop(nid, None)

        # Step 1: EMA Filtering
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
                self.a5_hold_ms[nid] = 0
                self.rsrp_history_neighbors[nid] = []
            else:
                self.rsrp_n_f[nid] = alpha * rsrp_neighbors[i] + (1 - alpha) * self.rsrp_n_f[nid]
                
        # Update history
        self._update_history(now_s, self.rsrp_s_f, self.rsrp_history_serving)
        for nid in neighbor_ids:
            self._update_history(now_s, self.rsrp_n_f[nid], self.rsrp_history_neighbors[nid])

        # Step 2: Signal Quality Classification
        if self.rsrp_s_f < -120 or self.sinr_s_f < -10 or cqi_serving < 5:
            signal_quality = "POOR"
        elif self.rsrp_s_f >= -100 and self.sinr_s_f >= 0 and cqi_serving >= 10:
            signal_quality = "GOOD"
        else:
            signal_quality = "FAIR"

        # Step 3: Coverage Zone Classification
        if distance_serving < 350:
            zone = "CELL_CENTER"
        elif distance_serving < 480:
            zone = "HANDOFF_ZONE"
        elif distance_serving <= 500:
            zone = "CRITICAL_ZONE"
        else:
            zone = "EDGE_ZONE"

        # Step 4: RLF Risk Estimation (simplified)
        rlf_risk = 0.0
        
        # RSRP contribution
        if self.rsrp_s_f < -120:
            rlf_risk += 0.40
        elif self.rsrp_s_f < -110:
            rlf_risk += 0.20
        
        # SINR contribution
        if self.sinr_s_f < -10:
            rlf_risk += 0.40
        elif self.sinr_s_f < -5:
            rlf_risk += 0.20
        
        # Distance contribution (reduced from 0.15 to 0.05)
        if distance_serving >= 480:
            rlf_risk += 0.05
            
        rlf_risk = min(rlf_risk, 1.0)

        # Step 5: Ping-Pong Risk Estimation
        self.recent_handoffs = [t for t in self.recent_handoffs if now_s - t <= 60.0]
        handoff_count_60s = len(self.recent_handoffs)
        
        if handoff_count_60s == 0:
            ping_pong_risk = 0.0
        elif handoff_count_60s <= 2:
            ping_pong_risk = 0.10
        elif handoff_count_60s <= 5:
            ping_pong_risk = 0.40
        elif handoff_count_60s <= 10:
            ping_pong_risk = 0.70
        else:
            ping_pong_risk = 1.0
        
        # Dwell time (consistent at 0.375s)
        in_dwell = (now_s - self.last_handover_time) < 0.375
        non_emergency_allowed = (ping_pong_risk < 0.40) and not in_dwell

        # Step 6: MUST-HO Flag
        MUST_HO = (signal_quality == "POOR") or (zone == "EDGE_ZONE") or (rlf_risk >= 0.60)

        # Step 8: A3/A5 Holding & Eligibility Check
        eligible_candidates = []
        K_distance_max = 1500
        
        if len(neighbor_ids) > 0:
            current_strongest = max(neighbor_ids, key=lambda nid: self.rsrp_n_f[nid])
        else:
            current_strongest = None
        
        for i, nid in enumerate(neighbor_ids):
            if distance_neighbors[i] >= K_distance_max:
                continue
                
            # A3 condition (using input HYS_eff directly, assumes it's in [4.0, 5.0])
            A3_condition = (self.rsrp_n_f[nid] - self.rsrp_s_f) >= HYS_eff
            
            # A5 condition (emergency)
            A5_condition = (self.rsrp_s_f <= -110) and (self.rsrp_n_f[nid] >= -108)
            
            if A3_condition:
                self.a3_hold_ms[nid] += int(dt_s * 1000)
            else:
                self.a3_hold_ms[nid] = 0
            
            if A5_condition:
                self.a5_hold_ms[nid] += int(dt_s * 1000)
            else:
                self.a5_hold_ms[nid] = 0
            
            # Check TTT expiry (using input TTT_eff directly, assumes it's in [200, 300])
            if (self.a3_hold_ms[nid] >= TTT_eff) or (self.a5_hold_ms[nid] >= TTT_eff):
                if current_strongest == nid and self.previous_strongest_neighbor == nid:
                    eligible_candidates.append(i)

        # Step 9: Rank Eligible Candidates by Predicted Signal Margin
        T_PRED = 0.3
        W = 10
        
        margin_pred = {}
        
        if len(self.rsrp_history_serving) >= W:
            past_s = self.rsrp_history_serving[-W]
            rsrp_s_past = past_s[1]
            time_diff_s = max(now_s - past_s[0], 0.1)
            
            slope_s = (self.rsrp_s_f - rsrp_s_past) / time_diff_s
            
            for i in eligible_candidates:
                nid = neighbor_ids[i]
                history_n = self.rsrp_history_neighbors[nid]
                if len(history_n) >= W:
                    past_n = history_n[-W]
                    rsrp_n_past = past_n[1]
                    time_diff_n = max(now_s - past_n[0], 0.1)
                    
                    slope_n = (self.rsrp_n_f[nid] - rsrp_n_past) / time_diff_n
                    margin_delta = (slope_n - slope_s) * T_PRED
                    margin_pred[i] = (self.rsrp_n_f[nid] - self.rsrp_s_f) + margin_delta
                else:
                    margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
        else:
            for i in eligible_candidates:
                nid = neighbor_ids[i]
                margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
                
        # Sort candidates
        eligible_candidates.sort(key=lambda i: (
            margin_pred[i], 
            self.rsrp_n_f[neighbor_ids[i]], 
            -distance_neighbors[i]
        ), reverse=True)

        # Step 10: Handover Decision Engine
        action = 0
        target_cell_id = None
        confidence = max(0.4, min(1.0, 1.0 - (rlf_risk + ping_pong_risk) / 2.0))
        reason = "No HO trigger"
        
        def finalize_ho(trg_idx, conf, rsn):
            nonlocal action, target_cell_id, confidence, reason
            action = trg_idx + 1
            target_cell_id = neighbor_ids[trg_idx]
            confidence = conf
            reason = rsn
            
            self.recent_handoffs.append(now_s)
            self.last_handover_time = now_s

        # DECISION: MUST-HO
        if MUST_HO:
            if eligible_candidates:
                target_idx = eligible_candidates[0]
                base_confidence = 0.90
                
                if ping_pong_risk >= 0.70:
                    conf = base_confidence * 0.75
                    rsn = "MUST-HO: Critical signal loss (extreme ping-pong caution)"
                elif ping_pong_risk >= 0.40:
                    conf = base_confidence * 0.85
                    rsn = "MUST-HO: Critical signal loss (moderate ping-pong caution)"
                else:
                    conf = base_confidence
                    rsn = "MUST-HO: Critical signal loss, RLF imminent"
                    
                finalize_ho(target_idx, conf, rsn)
                return self._build_return(action, target_cell_id, confidence, reason, 
                                        rlf_risk, ping_pong_risk, signal_quality, zone, 
                                        eligible_candidates)
            else:
                reason = "MUST-HO triggered but no eligible candidates"
        
        # DECISION: PROACTIVE HO - RLF risk elevated
        elif rlf_risk >= 0.35 and eligible_candidates and non_emergency_allowed:
            target_idx = eligible_candidates[0]
            base_confidence = 0.82
            
            if in_dwell:
                conf = base_confidence * 0.60
                rsn = "Proactive HO: RLF risk 0.35+ (dwell constraint active)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.65
                rsn = "Proactive HO: RLF risk 0.35+ (high ping-pong caution)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.80
                rsn = "Proactive HO: RLF risk 0.35+ (moderate ping-pong caution)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: RLF risk elevated (0.35+)"
                
            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, 
                                    rlf_risk, ping_pong_risk, signal_quality, zone, 
                                    eligible_candidates)
        
        # DECISION: PROACTIVE HO - High mobility in critical zone
        elif (velocity >= 10 and zone == "CRITICAL_ZONE" and 
              rlf_risk >= 0.25 and eligible_candidates and non_emergency_allowed):
            target_idx = eligible_candidates[0]
            base_confidence = 0.78
            
            if in_dwell:
                conf = base_confidence * 0.55
                rsn = "Proactive HO: High mobility in critical zone (dwell active)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.60
                rsn = "Proactive HO: High mobility in critical zone (extreme ping-pong)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.75
                rsn = "Proactive HO: High mobility in critical zone (moderate ping-pong)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: High mobility in critical zone with signal degradation"

            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, 
                                    rlf_risk, ping_pong_risk, signal_quality, zone, 
                                    eligible_candidates)
        
        # DECISION: PROACTIVE HO - Fair signal with moderate RLF risk
        elif (signal_quality == "FAIR" and 0.25 <= rlf_risk < 0.50 and 
              eligible_candidates and non_emergency_allowed):
            target_idx = eligible_candidates[0]
            base_confidence = 0.75
            
            if in_dwell:
                conf = base_confidence * 0.50
                rsn = "Proactive HO: Fair signal + RLF risk (dwell active)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.55
                rsn = "Proactive HO: Fair signal + RLF risk (extreme ping-pong)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.70
                rsn = "Proactive HO: Fair signal + RLF risk (moderate ping-pong)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: Fair signal with moderate RLF risk"
                
            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, 
                                    rlf_risk, ping_pong_risk, signal_quality, zone, 
                                    eligible_candidates)
        
        # DECISION: STAY (default)
        self.previous_strongest_neighbor = current_strongest
        return self._build_return(action, target_cell_id, confidence, reason, 
                                rlf_risk, ping_pong_risk, signal_quality, zone, 
                                eligible_candidates)

    def _build_return(self, action, target_cell_id, confidence, reason, 
                     rlf_risk, ping_pong_risk, signal_quality, zone, 
                     eligible_candidates):
        """Build return dictionary with all diagnostics."""
        return {
            "action": action,
            "target_cell_id": target_cell_id,
            "confidence": confidence,
            "reason": reason,
            "rlf_risk": rlf_risk,
            "ping_pong_risk": ping_pong_risk,
            "signal_quality": signal_quality,
            "zone": zone,
            "eligible_candidates": eligible_candidates
        }