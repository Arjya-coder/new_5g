import numpy as np

class Algorithm1:
    def __init__(self):
        # Persistent state
        self.rsrp_s_f = None
        self.sinr_s_f = None
        self.rsrp_n_f = {}  # dict of id -> filtered rsrp
        
        self.a3_hold_ms = {}  # dict of id -> held time in ms
        self.a5_hold_ms = {}  # dict of id -> held time in ms
        self.previous_strongest_neighbor = None
        
        self.recent_handoffs = []  # list of timestamps (seconds)
        self.last_handover_time = -float('inf')
        
        # History buffers for slope prediction (Step 9)
        self.rsrp_history_serving = []  # list of (time, val)
        self.rsrp_history_neighbors = {}  # id -> list of (time, val)

    def _update_history(self, now_s, val, history_list):
        history_list.append((now_s, val))
        # keep last 5 seconds to be safe, we need at least W=10 (1 second at 100ms ticks)
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
             HYS_eff: float, 
             los_probability: float = 1.0):
        """
        Executes one 100ms tick of Algorithm 1.
        Returns a dict containing action, target_cell_id, confidence, reason, and diagnostics.
        """
        dt_s = 0.1  # Assuming 100ms ticks as specified in hold timer logic
        
        # Step 0: Sanitize Neighbor Lists
        K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
        rsrp_neighbors = rsrp_neighbors[:K]
        neighbor_ids = neighbor_ids[:K]
        distance_neighbors = distance_neighbors[:K]

        # Cleanup disappeared neighbors from internal state
        for nid in list(self.rsrp_n_f.keys()):
            if nid not in neighbor_ids:
                del self.rsrp_n_f[nid]
                if nid in self.a3_hold_ms: del self.a3_hold_ms[nid]
                if nid in self.a5_hold_ms: del self.a5_hold_ms[nid]
                if nid in self.rsrp_history_neighbors: del self.rsrp_history_neighbors[nid]

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
                
        # Update historians
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

        # Step 4: RLF Risk Estimation (0.0 to 1.0)
        rlf_risk = 0.0
        if self.rsrp_s_f < -120: rlf_risk += 0.30
        elif self.rsrp_s_f < -110: rlf_risk += 0.15
        
        if self.sinr_s_f < -10: rlf_risk += 0.30
        elif self.sinr_s_f < -5: rlf_risk += 0.15
        
        if distance_serving >= 480:
            rlf_risk += 0.15
            
        rlf_risk = min(rlf_risk, 1.0)

        # Step 5: Ping-Pong Risk Estimation & Dwell Guard
        # clean out old handoffs
        self.recent_handoffs = [t for t in self.recent_handoffs if now_s - t <= 60.0]
        handoff_count_60s = len(self.recent_handoffs)
        
        if handoff_count_60s == 0: ping_pong_risk = 0.0
        elif handoff_count_60s <= 2: ping_pong_risk = 0.10
        elif handoff_count_60s <= 5: ping_pong_risk = 0.40
        elif handoff_count_60s <= 10: ping_pong_risk = 0.70
        else: ping_pong_risk = 1.0
        
        in_dwell = (now_s - self.last_handover_time) < 0.375
        non_emergency_allowed = (ping_pong_risk < 0.40) and not in_dwell

        # Step 6: MUST-HO Flag
        MUST_HO = (signal_quality == "POOR") or (zone == "EDGE_ZONE") or (rlf_risk >= 0.50)

        # Step 7: Parameters (Passed as input arguments TTT_eff, HYS_eff)
        
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
                
            A3_condition = (self.rsrp_n_f[nid] - self.rsrp_s_f) >= HYS_eff
            A5_condition = (self.rsrp_s_f <= -110) and (self.rsrp_n_f[nid] >= -108)
            
            if A3_condition: self.a3_hold_ms[nid] += int(dt_s * 1000)
            else: self.a3_hold_ms[nid] = 0
            
            if A5_condition: self.a5_hold_ms[nid] += int(dt_s * 1000)
            else: self.a5_hold_ms[nid] = 0
            
            if (self.a3_hold_ms[nid] >= TTT_eff) or (self.a5_hold_ms[nid] >= TTT_eff):
                if current_strongest == nid and self.previous_strongest_neighbor == nid:
                    eligible_candidates.append(i)

        # Step 9: Rank Eligible Candidates by Predicted Signal Margin
        T_PRED = 0.3 # 300ms
        W = 10       # 1-second history window at 100ms intervals
        
        margin_pred = {}
        slope_available = False
        
        # To compute slope, we need at least W items backwards
        if len(self.rsrp_history_serving) >= W:
            # past index is W steps back (i.e., -W)
            past_s = self.rsrp_history_serving[-W]
            rsrp_s_past = past_s[1]
            time_diff_s = max(now_s - past_s[0], 0.1) # avoid div0 just in case
            
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
                    margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f # fallback
            slope_available = True
        else:
            for i in eligible_candidates:
                nid = neighbor_ids[i]
                margin_pred[i] = self.rsrp_n_f[nid] - self.rsrp_s_f
                
        # Sort candidates
        eligible_candidates.sort(key=lambda i: (
            margin_pred[i], 
            self.rsrp_n_f[neighbor_ids[i]], 
            -distance_neighbors[i] # ascending distance == descending negative distance
        ), reverse=True)


        # Step 10: Handover Decision Engine
        action = 0
        target_cell_id = None
        confidence = max(0.4, min(1.0, 1.0 - (rlf_risk + ping_pong_risk)/2.0))
        reason = "No HO trigger: Signal adequate or no eligible candidates"
        
        def finalize_ho(trg_idx, conf, rsn):
            nonlocal action, target_cell_id, confidence, reason
            action = trg_idx + 1
            target_cell_id = neighbor_ids[trg_idx]
            confidence = conf
            reason = rsn
            
            # Post handover registry (simulate HO execution success here for context tracking,
            # though an NS3 env would confirm HO actually happened. We just track intent here.)
            self.recent_handoffs.append(now_s)
            self.last_handover_time = now_s

        if MUST_HO:
            if eligible_candidates:
                target_idx = eligible_candidates[0]
                base_confidence = 0.90
                
                if ping_pong_risk >= 0.70:
                    conf = base_confidence * 0.75
                    rsn = "MUST-HO (extreme ping-pong risk, caution advised)"
                elif ping_pong_risk >= 0.40:
                    conf = base_confidence * 0.85
                    rsn = "MUST-HO (moderate ping-pong risk)"
                else:
                    conf = base_confidence
                    rsn = "MUST-HO: Critical signal loss, RLF imminent"
                    
                finalize_ho(target_idx, conf, rsn)
                return self._build_return(action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates)
            else:
                # Emergency fallback... no candidates available
                reason = "MUST-HO triggered but NO eligible candidates"
        
        elif rlf_risk >= 0.35 and eligible_candidates:
            target_idx = eligible_candidates[0]
            base_confidence = 0.82
            
            if in_dwell:
                conf = base_confidence * 0.60
                rsn = "Proactive HO: RLF risk 0.35+, CAUTION (within 1s of last HO)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.65
                rsn = "Proactive HO: RLF risk 0.35+, CAUTION (high ping-pong activity)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.80
                rsn = "Proactive HO: RLF risk 0.35+, CAUTION (moderate ping-pong risk)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: RLF risk elevated (0.35+), eligible neighbor available"
                
            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates)
            
        elif velocity >= 10 and zone == "CRITICAL_ZONE" and rlf_risk >= 0.25 and eligible_candidates:
            target_idx = eligible_candidates[0]
            base_confidence = 0.78
            
            if in_dwell:
                conf = base_confidence * 0.55
                rsn = "Proactive HO: High mobility + signal dip, CAUTION (recent HO)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.60
                rsn = "Proactive HO: High mobility + signal dip, CAUTION (extreme ping-pong)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.75
                rsn = "Proactive HO: High mobility + signal dip, CAUTION (moderate ping-pong)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: High mobility in critical zone with signal degradation"

            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates)
            
        elif signal_quality == "FAIR" and 0.25 <= rlf_risk < 0.50 and eligible_candidates:
            target_idx = eligible_candidates[0]
            base_confidence = 0.75
            
            if in_dwell:
                conf = base_confidence * 0.50
                rsn = "Proactive HO: Fair signal + RLF risk, CAUTION (within dwell)"
            elif ping_pong_risk >= 0.70:
                conf = base_confidence * 0.55
                rsn = "Proactive HO: Fair signal + RLF risk, CAUTION (extreme ping-pong)"
            elif ping_pong_risk >= 0.40:
                conf = base_confidence * 0.70
                rsn = "Proactive HO: Fair signal + RLF risk, CAUTION (moderate ping-pong)"
            else:
                conf = base_confidence
                rsn = "Proactive HO: Fair signal with moderate RLF risk, eligible neighbor ready"
                
            finalize_ho(target_idx, conf, rsn)
            return self._build_return(action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates)
        self.previous_strongest_neighbor = current_strongest
        return self._build_return(action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates)

    def _build_return(self, action, target_cell_id, confidence, reason, rlf_risk, ping_pong_risk, signal_quality, zone, eligible_candidates):
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
