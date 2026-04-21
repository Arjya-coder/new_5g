INPUT:
1. rsrp_serving (dBm)
2. sinr_serving (dB)
3. cqi_serving (0-15)
4. distance_serving (meters)
5. rsrp_neighbors (list of dBm values)
6. neighbor_ids (list)
7. distance_neighbors (list of meters)
8. velocity (m/s)
9. now_s (current time in seconds)
10. TTT_eff (Time-To-Trigger in ms) in [200, 300]
11. HYS_eff (Hysteresis in dB) in [4, 5]

PERSISTENT STATE (kept across ticks):
1. rsrp_s_f (EMA-filtered serving RSRP)
2. sinr_s_f (EMA-filtered serving SINR)
3. rsrp_n_f[nid] (EMA-filtered neighbor RSRP by neighbor ID)
4. a3_hold_ms[nid] (A3 hold timer per neighbor)
5. a5_hold_ms[nid] (A5 hold timer per neighbor)
6. previous_strongest_neighbor (neighbor ID from previous tick)
7. recent_handoffs (handover timestamps, last 60s window)
8. last_handover_time (seconds)
9. rsrp_history_serving (time series of (time, serving_rsrp), 5s window)
10. rsrp_history_neighbors[nid] (time series of (time, neighbor_rsrp), 5s window)

OUTPUT:
1. action (0 = STAY, i+1 = HO to neighbor at index i)
2. target_cell_id (or None)
3. confidence (0.0-1.0)
4. reason (short text)
5. rlf_risk (0.0-1.0) [diagnostic]
6. ping_pong_risk (0.0-1.0) [diagnostic]
7. signal_quality ("POOR" / "FAIR" / "GOOD") [diagnostic]
8. zone ("CELL_CENTER" / "HANDOFF_ZONE" / "CRITICAL_ZONE" / "EDGE_ZONE") [diagnostic]
9. eligible_candidates (ranked list of indices) [diagnostic]

================================================================================

PROCESS:

Step 0: Sanitize Neighbor Lists
  K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
  Truncate all neighbor lists to length K

  Also clean disappeared neighbors from persistent maps:
  - rsrp_n_f
  - a3_hold_ms
  - a5_hold_ms
  - rsrp_history_neighbors

Step 1: EMA Filtering (Reduce measurement noise)
  alpha = 0.2 (smoothing factor)
  dt = 100 ms per tick

  IF first tick (rsrp_s_f is None) THEN
      rsrp_s_f = rsrp_serving
      sinr_s_f = sinr_serving
  ELSE
      rsrp_s_f = alpha*rsrp_serving + (1-alpha)*rsrp_s_f
      sinr_s_f = alpha*sinr_serving + (1-alpha)*sinr_s_f
  END IF

  FOR each neighbor ID nid in neighbor_ids:
      IF nid is new THEN
          rsrp_n_f[nid] = current measured neighbor RSRP
          a3_hold_ms[nid] = 0
          a5_hold_ms[nid] = 0
          rsrp_history_neighbors[nid] = []
      ELSE
          rsrp_n_f[nid] = alpha*current_neighbor_rsrp + (1-alpha)*rsrp_n_f[nid]
      END IF
  END FOR

  Update history buffers:
  - append (now_s, rsrp_s_f) to rsrp_history_serving
  - append (now_s, rsrp_n_f[nid]) for each neighbor
  - keep only last 5 seconds in each list

Step 2: Signal Quality Classification
  Classify current serving signal as POOR / FAIR / GOOD

  IF (rsrp_s_f < -120) OR (sinr_s_f < -10) OR (cqi_serving < 5) THEN
      signal_quality = "POOR"
  ELSE IF (rsrp_s_f >= -100) AND (sinr_s_f >= 0) AND (cqi_serving >= 10) THEN
      signal_quality = "GOOD"
  ELSE
      signal_quality = "FAIR"
  END IF

Step 3: Coverage Zone Classification
  Determine UE position with nominal cell radius R = 500 m

  IF distance_serving < 350 THEN
      zone = "CELL_CENTER"
  ELSE IF distance_serving < 480 THEN
      zone = "HANDOFF_ZONE"
  ELSE IF distance_serving <= 500 THEN
      zone = "CRITICAL_ZONE"
  ELSE
      zone = "EDGE_ZONE"
  END IF

Step 4: RLF (Radio Link Failure) Risk Estimation
  Quantify connection-loss likelihood if no HO occurs (0.0-1.0)

  rlf_risk = 0.0

  # RSRP component
  IF rsrp_s_f < -120 THEN
      rlf_risk += 0.40
  ELSE IF rsrp_s_f < -110 THEN
      rlf_risk += 0.20
  END IF

  # SINR component
  IF sinr_s_f < -10 THEN
      rlf_risk += 0.40
  ELSE IF sinr_s_f < -5 THEN
      rlf_risk += 0.20
  END IF

  # Distance component (light edge penalty)
  IF distance_serving >= 480 THEN
      rlf_risk += 0.05
  END IF

  rlf_risk = MIN(rlf_risk, 1.0)

Step 5: Ping-Pong Risk Estimation and Dwell Guard
  Keep only handoffs in last 60 seconds:
  recent_handoffs = [t in recent_handoffs where now_s - t <= 60.0]

  handoff_count_60s = len(recent_handoffs)

  IF handoff_count_60s == 0 THEN
      ping_pong_risk = 0.0
  ELSE IF handoff_count_60s <= 2 THEN
      ping_pong_risk = 0.10
  ELSE IF handoff_count_60s <= 5 THEN
      ping_pong_risk = 0.40
  ELSE IF handoff_count_60s <= 10 THEN
      ping_pong_risk = 0.70
  ELSE
      ping_pong_risk = 1.0
  END IF

  # Dwell timer in current code: 375 ms
  in_dwell = (now_s - last_handover_time) < 0.375

  # Non-emergency HO allowed only if low ping-pong AND not in dwell
  non_emergency_allowed = (ping_pong_risk < 0.40) AND (NOT in_dwell)

Step 6: MUST-HO Flag (Emergency Handover Required)
  MUST_HO = (signal_quality == "POOR")
            OR (zone == "EDGE_ZONE")
            OR (rlf_risk >= 0.60)

  # MUST_HO can proceed even when non-emergency is blocked

Step 7: TTT and HYS Parameters
  TTT_eff and HYS_eff are used directly from input (no clipping in code)

  TTT_eff in [200, 300] ms   # Time-To-Trigger
  HYS_eff in [4, 5] dB       # A3 hysteresis margin

Step 8: A3/A5 Holding and Eligibility Check
  Determine neighbors that satisfy A3/A5 + TTT + stability guard

  eligible_candidates = []
  K_distance_max = 1500  # meters

  IF there is at least one neighbor THEN
      current_strongest = argmax over neighbor_ids using rsrp_n_f[nid]
  ELSE
      current_strongest = None
  END IF

  FOR i = 0..K-1 DO
      nid = neighbor_ids[i]

      # Filter 1: Distance
      IF distance_neighbors[i] >= K_distance_max THEN
          CONTINUE
      END IF

      # A3 event
      A3_condition = (rsrp_n_f[nid] - rsrp_s_f) >= HYS_eff

      # A5 event
      A5_condition = (rsrp_s_f <= -110) AND (rsrp_n_f[nid] >= -108)

      # Hold timers (100 ms per tick)
      IF A3_condition THEN
          a3_hold_ms[nid] += 100
      ELSE
          a3_hold_ms[nid] = 0
      END IF

      IF A5_condition THEN
          a5_hold_ms[nid] += 100
      ELSE
          a5_hold_ms[nid] = 0
      END IF

      # Eligibility requires TTT and strongest-neighbor stability for 2 ticks
      IF (a3_hold_ms[nid] >= TTT_eff) OR (a5_hold_ms[nid] >= TTT_eff) THEN
          IF (current_strongest == nid) AND (previous_strongest_neighbor == nid) THEN
              eligible_candidates.append(i)
          END IF
      END IF
  END FOR

Step 9: Rank Eligible Candidates by PREDICTED Signal Margin
  Sort candidates using slope-based near-future prediction

  +- STEP 9A: Compute margin predictions
  |
  T_PRED = 0.3  # predict 300 ms ahead
  W = 10        # use 10 samples (~1 second at 100 ms tick)

  IF len(rsrp_history_serving) >= W THEN
      past_s = rsrp_history_serving[-W]
      rsrp_s_past = past_s[1]
      time_diff_s = max(now_s - past_s[0], 0.1)
      slope_s = (rsrp_s_f - rsrp_s_past) / time_diff_s

      FOR i IN eligible_candidates:
          nid = neighbor_ids[i]
          history_n = rsrp_history_neighbors[nid]

          IF len(history_n) >= W THEN
              past_n = history_n[-W]
              rsrp_n_past = past_n[1]
              time_diff_n = max(now_s - past_n[0], 0.1)
              slope_n = (rsrp_n_f[nid] - rsrp_n_past) / time_diff_n

              margin_delta = (slope_n - slope_s) * T_PRED
              margin_pred[i] = (rsrp_n_f[nid] - rsrp_s_f) + margin_delta
          ELSE
              margin_pred[i] = rsrp_n_f[nid] - rsrp_s_f
          END IF
      END FOR
  ELSE
      FOR i IN eligible_candidates:
          nid = neighbor_ids[i]
          margin_pred[i] = rsrp_n_f[nid] - rsrp_s_f
      END FOR
  END IF

  +- STEP 9B: Sort candidates
  |
  Sort eligible_candidates by descending tuple:
    1. margin_pred[i]
    2. rsrp_n_f[neighbor_ids[i]]
    3. -distance_neighbors[i]

  Effective tie-break meaning:
    - higher margin is better
    - then stronger neighbor RSRP
    - then shorter neighbor distance

Step 10: HANDOVER DECISION ENGINE (REFINED WITH SOFT PENALTIES)

+- EMERGENCY CONDITION: MUST_HO
|
IF MUST_HO == TRUE THEN
    IF eligible_candidates is NOT EMPTY THEN
        target_idx = eligible_candidates[0]
        action = target_idx + 1
        target_cell_id = neighbor_ids[target_idx]

        base_confidence = 0.90

        IF ping_pong_risk >= 0.70 THEN
            confidence = base_confidence * 0.75
            reason = "MUST-HO: Critical signal loss (extreme ping-pong caution)"
        ELSE IF ping_pong_risk >= 0.40 THEN
            confidence = base_confidence * 0.85
            reason = "MUST-HO: Critical signal loss (moderate ping-pong caution)"
        ELSE
            confidence = base_confidence
            reason = "MUST-HO: Critical signal loss, RLF imminent"
        END IF

        finalize_ho()  # records handoff timestamp + updates last_handover_time
        RETURN
    ELSE
        reason = "MUST-HO triggered but no eligible candidates"
    END IF
END IF

+- PROACTIVE CONDITION 1: Elevated RLF Risk
|
IF (rlf_risk >= 0.35) AND (eligible_candidates NOT EMPTY) AND (non_emergency_allowed) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]

    base_confidence = 0.82

    IF in_dwell THEN
        confidence = base_confidence * 0.60
        reason = "Proactive HO: RLF risk 0.35+ (dwell constraint active)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.65
        reason = "Proactive HO: RLF risk 0.35+ (high ping-pong caution)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.80
        reason = "Proactive HO: RLF risk 0.35+ (moderate ping-pong caution)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: RLF risk elevated (0.35+)"
    END IF

    finalize_ho()
    RETURN
END IF

+- PROACTIVE CONDITION 2: High Mobility + Marginal Signal
|
IF (velocity >= 10) AND (zone == "CRITICAL_ZONE") AND (rlf_risk >= 0.25)
   AND (eligible_candidates NOT EMPTY) AND (non_emergency_allowed) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]

    base_confidence = 0.78

    IF in_dwell THEN
        confidence = base_confidence * 0.55
        reason = "Proactive HO: High mobility in critical zone (dwell active)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.60
        reason = "Proactive HO: High mobility in critical zone (extreme ping-pong)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.75
        reason = "Proactive HO: High mobility in critical zone (moderate ping-pong)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: High mobility in critical zone with signal degradation"
    END IF

    finalize_ho()
    RETURN
END IF

+- PROACTIVE CONDITION 3: FAIR Signal + Meaningful RLF Risk
|
IF (signal_quality == "FAIR") AND (rlf_risk >= 0.25) AND (rlf_risk < 0.50)
   AND (eligible_candidates NOT EMPTY) AND (non_emergency_allowed) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]

    base_confidence = 0.75

    IF in_dwell THEN
        confidence = base_confidence * 0.50
        reason = "Proactive HO: Fair signal + RLF risk (dwell active)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.55
        reason = "Proactive HO: Fair signal + RLF risk (extreme ping-pong)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.70
        reason = "Proactive HO: Fair signal + RLF risk (moderate ping-pong)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: Fair signal with moderate RLF risk"
    END IF

    finalize_ho()
    RETURN
END IF

+- DEFAULT: STAY
|
action = 0
target_cell_id = None
confidence = CLAMP(1.0 - (rlf_risk + ping_pong_risk)/2, 0.4, 1.0)
reason = "No HO trigger"

# Keep strongest-neighbor continuity only on non-handover path
previous_strongest_neighbor = current_strongest
RETURN

END ALGORITHM
================================================================================
