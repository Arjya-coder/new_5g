import numpy as np


class Algorithm1:
    """
    Algorithm 1: Clean, Minimal 5G Handover Baseline
    Upgraded for PPO compatibility (dynamic TTT/HYS).

    PPO-compatible runtime ranges:
    - HYS_eff in [2.0, 5.0] dB
    - TTT_eff in [100, 320] ms

    Notes:
    - We preserve original handover logic structure.
    - We only make the baseline robust to wider control ranges.
    """

    def __init__(self):
        self.rsrp_s_f = None
        self.sinr_s_f = None
        self.rsrp_n_f = {}

        self.a3_hold_ms = {}
        self.a5_hold_ms = {}
        self.previous_strongest_neighbor = None

        self.recent_handoffs = []
        self.last_handover_time = -float("inf")

        self.rsrp_history_serving = []
        self.rsrp_history_neighbors = {}

        # Keep current serving cell id for offline wrappers that depend on it
        self.serving_cell_id = 0

    @staticmethod
    def _sanitize_ttt(ttt_eff: int) -> int:
        # PPO-compatible clamp
        return int(np.clip(int(ttt_eff), 100, 320))

    @staticmethod
    def _sanitize_hys(hys_eff: float) -> float:
        # PPO-compatible clamp
        return float(np.clip(float(hys_eff), 2.0, 5.0))

    def _update_history(self, now_s, val, history_list):
        history_list.append((now_s, val))
        while len(history_list) > 0 and now_s - history_list[0][0] > 5.0:
            history_list.pop(0)

    def step(
        self,
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
    ):
        """
        Executes one 100ms tick of Algorithm 1 baseline with PPO-compatible controls.
        """
        dt_s = 0.1

        # --- NEW: sanitize external control inputs ---
        TTT_eff = self._sanitize_ttt(TTT_eff)
        HYS_eff = self._sanitize_hys(HYS_eff)

        # Step 0: sanitize neighbor arrays
        K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
        rsrp_neighbors = rsrp_neighbors[:K]
        neighbor_ids = neighbor_ids[:K]
        distance_neighbors = distance_neighbors[:K]

        for nid in list(self.rsrp_n_f.keys()):
            if nid not in neighbor_ids:
                del self.rsrp_n_f[nid]
                self.a3_hold_ms.pop(nid, None)
                self.a5_hold_ms.pop(nid, None)
                self.rsrp_history_neighbors.pop(nid, None)

        # Step 1: EMA filtering
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

        self._update_history(now_s, self.rsrp_s_f, self.rsrp_history_serving)
        for nid in neighbor_ids:
            self._update_history(now_s, self.rsrp_n_f[nid], self.rsrp_history_neighbors[nid])

        # Step 2: signal quality
        if self.rsrp_s_f < -120 or self.sinr_s_f < -10 or cqi_serving < 5:
            signal_quality = "POOR"
        elif self.rsrp_s_f >= -100 and self.sinr_s_f >= 0 and cqi_serving >= 10:
            signal_quality = "GOOD"
        else:
            signal_quality = "FAIR"

        # Step 3: zone classification
        if distance_serving < 350:
            zone = "CELL_CENTER"
        elif distance_serving < 480:
            zone = "HANDOFF_ZONE"
        elif distance_serving <= 500:
            zone = "CRITICAL_ZONE"
        else:
            zone = "EDGE_ZONE"

        # Step 4: RLF risk
        rlf_risk = 0.0
        if self.rsrp_s_f < -120:
            rlf_risk += 0.40
        elif self.rsrp_s_f < -110:
            rlf_risk += 0.20

        if self.sinr_s_f < -10:
            rlf_risk += 0.40
        elif self.sinr_s_f < -5:
            rlf_risk += 0.20

        if distance_serving >= 480:
            rlf_risk += 0.05

        rlf_risk = min(rlf_risk, 1.0)

        # Step 5: ping-pong risk
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

        # Adaptive dwell (small improvement, still conservative)
        # shorter dwell allowed at low HYS/TTT, longer at higher stability settings
        dwell_s = float(np.clip(0.25 + 0.001 * (TTT_eff - 100) + 0.05 * (HYS_eff - 2.0), 0.25, 0.6))
        in_dwell = (now_s - self.last_handover_time) < dwell_s
        non_emergency_allowed = (ping_pong_risk < 0.40) and (not in_dwell)

        # Step 6: MUST-HO
        MUST_HO = (signal_quality == "POOR") or (zone == "EDGE_ZONE") or (rlf_risk >= 0.60)

        # Step 8: A3/A5 hold and eligibility
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

            if A3_condition:
                self.a3_hold_ms[nid] += int(dt_s * 1000)
            else:
                self.a3_hold_ms[nid] = 0

            if A5_condition:
                self.a5_hold_ms[nid] += int(dt_s * 1000)
            else:
                self.a5_hold_ms[nid] = 0

            if (self.a3_hold_ms[nid] >= TTT_eff) or (self.a5_hold_ms[nid] >= TTT_eff):
                if current_strongest == nid and self.previous_strongest_neighbor == nid:
                    eligible_candidates.append(i)

        # Step 9: rank by predicted margin
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

        eligible_candidates.sort(
            key=lambda i: (
                margin_pred[i],
                self.rsrp_n_f[neighbor_ids[i]],
                -distance_neighbors[i],
            ),
            reverse=True,
        )

        # Step 10: decision engine
        action = 0
        target_cell_id = None
        confidence = max(0.4, min(1.0, 1.0 - (rlf_risk + ping_pong_risk) / 2.0))
        reason = "No HO trigger"
        ho_occurred = False

        def finalize_ho(trg_idx, conf, rsn):
            nonlocal action, target_cell_id, confidence, reason, ho_occurred
            action = trg_idx + 1
            target_cell_id = neighbor_ids[trg_idx]
            confidence = conf
            reason = rsn
            ho_occurred = True

            self.recent_handoffs.append(now_s)
            self.last_handover_time = now_s
            self.serving_cell_id = int(target_cell_id)

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
            else:
                reason = "MUST-HO triggered but no eligible candidates"

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

        self.previous_strongest_neighbor = current_strongest

        return self._build_return(
            action=action,
            target_cell_id=target_cell_id,
            confidence=confidence,
            reason=reason,
            rlf_risk=rlf_risk,
            ping_pong_risk=ping_pong_risk,
            signal_quality=signal_quality,
            zone=zone,
            eligible_candidates=eligible_candidates,
            ho_occurred=ho_occurred,
            serving_cell_id=self.serving_cell_id,
            rsrp_serving_dbm=float(self.rsrp_s_f),
            sinr_serving_db=float(self.sinr_s_f),
            cqi_serving=int(cqi_serving),
            distance_serving=float(distance_serving),
            velocity=float(velocity),
            neighbor_ids=neighbor_ids,
            rsrp_neighbors=[float(self.rsrp_n_f[nid]) for nid in neighbor_ids],
            distance_neighbors=[float(d) for d in distance_neighbors],
            num_neighbors=len(neighbor_ids),
            now_s=float(now_s),
        )

    def _build_return(
        self,
        action,
        target_cell_id,
        confidence,
        reason,
        rlf_risk,
        ping_pong_risk,
        signal_quality,
        zone,
        eligible_candidates,
        ho_occurred,
        serving_cell_id,
        rsrp_serving_dbm,
        sinr_serving_db,
        cqi_serving,
        distance_serving,
        velocity,
        neighbor_ids,
        rsrp_neighbors,
        distance_neighbors,
        num_neighbors,
        now_s,
    ):
        return {
            "action": action,
            "target_cell_id": target_cell_id,
            "confidence": confidence,
            "reason": reason,
            "rlf_risk": rlf_risk,
            "ping_pong_risk": ping_pong_risk,
            "signal_quality": signal_quality,
            "zone": zone,
            "eligible_candidates": eligible_candidates,

            # PPO/Offline compatibility fields
            "ho_occurred": bool(ho_occurred),
            "serving_cell_id": int(serving_cell_id),
            "rsrp_serving_dbm": float(rsrp_serving_dbm),
            "sinr_serving_db": float(sinr_serving_db),
            "cqi_serving": int(cqi_serving),
            "distance_serving": float(distance_serving),
            "velocity": float(velocity),
            "neighbor_ids": [int(x) for x in neighbor_ids],
            "rsrp_neighbors": [float(x) for x in rsrp_neighbors],
            "distance_neighbors": [float(x) for x in distance_neighbors],
            "num_neighbors": int(num_neighbors),
            "now_s": float(now_s),
        }