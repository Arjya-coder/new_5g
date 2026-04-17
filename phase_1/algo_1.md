INPUT:
1. rsrp_serving (dBm)
2. sinr_serving (dB)
3. cqi_serving (0–15)
4. distance_serving (meters)
5. rsrp_neighbors (list of dBm values)
6. neighbor_ids (list)
7. distance_neighbors (list of meters)
8. velocity (m/s)
9. los_probability (0–1) [optional]
10. recent_handoffs (list of timestamps, seconds)
11. now_s (current time in seconds)
12. TTT_eff (Time-To-Trigger in ms) ∈ [200, 300]
13. HYS_eff (Hysteresis in dB) ∈ [4, 5]

PERSISTENT STATE (kept across ticks):
1. rsrp_s_f (EMA-filtered serving RSRP)
2. sinr_s_f (EMA-filtered serving SINR)
3. rsrp_n_f[i] (EMA-filtered neighbor RSRPs)
4. a3_hold_ms[i] (A3 hold timer per neighbor)
5. a5_hold_ms[i] (A5 hold timer per neighbor)
6. last_handover_time (seconds)

OUTPUT:
1. action (0 = STAY, i+1 = HO to neighbor at index i)
2. target_cell_id (or None)
3. confidence (0.0–1.0)
4. reason (short text)
5. rlf_risk (0.0–1.0) [diagnostic]
6. ping_pong_risk (0.0–1.0) [diagnostic]
7. signal_quality ("POOR" / "FAIR" / "GOOD") [diagnostic]
8. zone ("CELL_CENTER" / "HANDOFF_ZONE" / "CRITICAL_ZONE" / "EDGE_ZONE") [diagnostic]
9. eligible_candidates (ranked list of indices) [diagnostic]

================================================================================

PROCESS:

Step 0: Sanitize Neighbor Lists
  K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
  Truncate all neighbor lists to length K

Step 1: EMA Filtering (Reduce measurement noise)
  α = 0.2 (smoothing factor)
  
  rsrp_s_f = α·rsrp_serving + (1−α)·rsrp_s_f
  sinr_s_f = α·sinr_serving + (1−α)·sinr_s_f
  
  FOR i = 0..K−1:
    rsrp_n_f[i] = α·rsrp_neighbors[i] + (1−α)·rsrp_n_f[i]
  END FOR

Step 2: Signal Quality Classification
  Classify current serving cell signal as POOR / FAIR / GOOD
  
  IF (rsrp_s_f < −120) OR (sinr_s_f < −10) OR (cqi_serving < 5) THEN
      signal_quality = "POOR"
  ELSE IF (rsrp_s_f >= −100) AND (sinr_s_f >= 0) AND (cqi_serving >= 10) THEN
      signal_quality = "GOOD"
  ELSE
      signal_quality = "FAIR"
  END IF

Step 3: Coverage Zone Classification
  Determine UE's position within serving cell
  (Cell radius R = 500 m assumed)
  
  IF distance_serving < 350 THEN
      zone = "CELL_CENTER"          # Strong, stable coverage
  ELSE IF distance_serving < 480 THEN
      zone = "HANDOFF_ZONE"         # Approaching edge, neighbors visible
  ELSE IF distance_serving <= 500 THEN
      zone = "CRITICAL_ZONE"        # ← RENAMED from DANGER_ZONE
                                    # Very close to cell edge, may experience degradation
  ELSE
      zone = "EDGE_ZONE"            # ← RENAMED from OUT_OF_COVERAGE
                                    # Beyond nominal cell boundary
  END IF

Step 4: RLF (Radio Link Failure) Risk Estimation
  Quantify likelihood of connection loss if no HO occurs (0.0–1.0)
  
  rlf_risk = 0.0
  
  # RSRP component (strongest indicator of RLF risk)
  IF rsrp_s_f < −120 THEN
      rlf_risk += 0.30        # Very weak signal: high RLF probability
  ELSE IF rsrp_s_f < −110 THEN
      rlf_risk += 0.15        # Weak signal: moderate RLF probability
  END IF
  
  # SINR component (interference impact)
  IF sinr_s_f < −10 THEN
      rlf_risk += 0.30        # Severe interference: high RLF probability
  ELSE IF sinr_s_f < −5 THEN
      rlf_risk += 0.15        # Moderate interference: moderate RLF probability
  END IF
  
  # Distance component (coverage edge effect)
  IF distance_serving >= 480 THEN
      rlf_risk += 0.15        # ← REDUCED from 0.25 (was too aggressive)
                              # Being at edge increases RLF, but not catastrophically
  END IF
  
  rlf_risk = MIN(rlf_risk, 1.0)  # Cap at 1.0

Step 5: Ping-Pong Risk Estimation & Dwell Guard
  Quantify risk of rapid re-handover oscillation (0.0–1.0)
  Prevent HO decisions immediately after recent handover
  
  handoff_count_60s = COUNT(τ ∈ recent_handoffs : now_s−60 ≤ τ ≤ now_s)
  
  # Classify handoff frequency
  IF handoff_count_60s == 0 THEN
      ping_pong_risk = 0.0        # No recent HOs: safe to HO
  ELSE IF handoff_count_60s <= 2 THEN
      ping_pong_risk = 0.10       # 1-2 HOs in 60s: low risk
  ELSE IF handoff_count_60s <= 5 THEN
      ping_pong_risk = 0.40       # 3-5 HOs in 60s: moderate risk
  ELSE IF handoff_count_60s <= 10 THEN
      ping_pong_risk = 0.70       # 6-10 HOs in 60s: high risk
  ELSE
      ping_pong_risk = 1.0        # >10 HOs in 60s: extreme risk (oscillation)
  END IF
  
  # Dwell timer: block HO for 250 ms after last handover (prevents immediate re-HO)
  in_dwell = (now_s − last_handover_time) < 0.250
  
  # Non-emergency HO allowed only if ping-pong is low AND not in dwell period
  non_emergency_allowed = (ping_pong_risk < 0.40) AND (NOT in_dwell)

Step 6: MUST-HO Flag (Emergency Handover Required)
  Determine if HO is mandatory for survival (critical condition)
  
  MUST_HO = (signal_quality == "POOR") 
            OR (zone == "EDGE_ZONE") 
            OR (rlf_risk >= 0.50)
  
  # MUST_HO overrides all other logic and dwell guards
  # Used only when RLF is imminent

Step 7: TTT & HYS Parameters
  These come from RL agent (or baseline fixed values for Algorithm-1-only test)
  
  TTT_eff ∈ [200, 300] ms          # Time-To-Trigger (hold time for A3/A5)
  HYS_eff ∈ [4, 5] dB              # Hysteresis margin (A3 event threshold offset)

Step 8: A3/A5 Holding & Eligibility Check
  Determine which neighbors meet handover criteria (A3 or A5 events)
  Apply TTT hold timers to prevent flapping
  
  eligible_candidates = []
  K_distance_max = 1500  # meters, max distance to consider neighbor
  
  FOR i = 0..K−1 DO
      # Filter 1: Distance constraint
      IF distance_neighbors[i] >= K_distance_max THEN
          CONTINUE  # Neighbor too far, skip
      END IF

      # A3 Event: Neighbor signal is HYS_eff dB stronger than serving
      # Indicates neighbor has better coverage
      A3_condition = (rsrp_n_f[i] − rsrp_s_f) >= HYS_eff

      # A5 Event: Serving RSRP is weak AND neighbor RSRP is acceptable
      # Emergency condition: serving cell is failing, neighbor can help
      A5_condition = (rsrp_s_f <= −110) AND (rsrp_n_f[i] >= −108)

      # A3 Hold Timer: Count consecutive 100ms ticks where A3 condition is true
      IF A3_condition THEN
          a3_hold_ms[i] += 100     # Increment by one tick duration (100ms)
      ELSE
          a3_hold_ms[i] = 0        # Reset if condition becomes false
      END IF

      # A5 Hold Timer: Count consecutive 100ms ticks where A5 condition is true
      IF A5_condition THEN
          a5_hold_ms[i] += 100     # Increment by one tick duration
      ELSE
          a5_hold_ms[i] = 0        # Reset if condition becomes false
      END IF

      # Eligibility: Either A3 or A5 hold timer has exceeded TTT
      # Means condition has been stable for TTT duration
      # PLUS: Require best neighbor to remain strongest for >= 2 consecutive ticks
      IF (a3_hold_ms[i] >= TTT_eff) OR (a5_hold_ms[i] >= TTT_eff) THEN
          IF neighbor_is_strongest_for_2_ticks THEN
              eligible_candidates.append(i)
          END IF
      END IF
  END FOR

Step 9: Rank Eligible Candidates by PREDICTED Signal Margin
  Sort candidates by quality using slope-based future prediction
  
  ┌─ STEP 9A: Compute margin predictions
  │
  T_PRED = 0.3  # Look 300ms into future
  W = 10        # Use 1-second history window for slope
  
  IF (rsrp_history_buffer.size() >= W) THEN
      # Compute slope (rate of change) for serving cell
      rsrp_s_past = rsrp_history_serving[t − (W−1)]
      rsrp_s_now = rsrp_s_f
      slope_s = (rsrp_s_now − rsrp_s_past) / ((W−1) * 0.1)  # dB/second
      
      # Compute slope and predict future margin for each eligible neighbor
      FOR i IN eligible_candidates:
          rsrp_n_past = rsrp_history_neighbors[i][t − (W−1)]
          rsrp_n_now = rsrp_n_f[i]
          slope_n[i] = (rsrp_n_now − rsrp_n_past) / ((W−1) * 0.1)  # dB/second
          
          # Predicted relative change in margin
          margin_delta[i] = (slope_n[i] − slope_s) * T_PRED
          
          # Predicted future margin = current + delta
          current_margin[i] = rsrp_n_f[i] − rsrp_s_f
          margin_pred[i] = current_margin[i] + margin_delta[i]
      END FOR
      
      slope_available = TRUE
  ELSE
      # Not enough history yet, fall back to current margin
      FOR i IN eligible_candidates:
          margin_pred[i] = rsrp_n_f[i] − rsrp_s_f
      END FOR
      
      slope_available = FALSE
  END IF
  
  ┌─ STEP 9B: Sort candidates
  │
  Sort eligible_candidates by:
    1. margin_pred[i] (descending)           # Higher predicted margin = better future coverage
    2. rsrp_n_f[i] (descending)              # Strongest current neighbor as tiebreaker
    3. distance_neighbors[i] (ascending)     # Nearest neighbor as final tiebreaker
  
  # Output diagnostic
  best_candidate_idx = eligible_candidates[0]
  best_margin_predicted = margin_pred[best_candidate_idx]
  slope_info = slope_available  # For dataset logging
  
Step 10: HANDOVER DECISION ENGINE (REFINED WITH SOFT PENALTIES)

┌─ EMERGENCY CONDITION: MUST_HO
│
IF MUST_HO == TRUE THEN
    IF eligible_candidates is NOT EMPTY THEN
        target_idx = eligible_candidates[0]
        action = target_idx + 1
        target_cell_id = neighbor_ids[target_idx]
        
        # SOFT PENALTY: MUST_HO still happens, but confidence reduced if ping-pong risk high
        base_confidence = 0.90
        
        # Apply soft ping-pong penalty
        IF ping_pong_risk >= 0.70 THEN
            confidence = base_confidence * 0.75  # Reduce by 25%
            reason = "MUST-HO (extreme ping-pong risk, caution advised)"
        ELSE IF ping_pong_risk >= 0.40 THEN
            confidence = base_confidence * 0.85  # Reduce by 15%
            reason = "MUST-HO (moderate ping-pong risk)"
        ELSE
            confidence = base_confidence
            reason = "MUST-HO: Critical signal loss, RLF imminent"
        END IF
        
        RETURN
    ELSE
        # Emergency fallback logic (unchanged)
        ...
    END IF
END IF

┌─ PROACTIVE CONDITION 1: Moderate RLF Risk
│
IF (rlf_risk >= 0.35) AND (eligible_candidates NOT EMPTY) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]
    
    base_confidence = 0.82
    
    # SOFT DWELL PENALTY: Recent HO → reduce confidence, but allow HO
    in_dwell = (now_s − last_handover_time) < 1.0
    IF in_dwell THEN
        confidence = base_confidence * 0.60  # Reduce by 40% if in dwell
        reason = "Proactive HO: RLF risk 0.35+, CAUTION (within 1s of last HO)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.65  # Reduce by 35% if extreme ping-pong
        reason = "Proactive HO: RLF risk 0.35+, CAUTION (high ping-pong activity)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.80  # Reduce by 20% if moderate ping-pong
        reason = "Proactive HO: RLF risk 0.35+, CAUTION (moderate ping-pong risk)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: RLF risk elevated (0.35+), eligible neighbor available"
    END IF
    
    RETURN
END IF

┌─ PROACTIVE CONDITION 2: High Mobility + Marginal Signal
│
IF (velocity >= 10) AND (zone == "CRITICAL_ZONE") AND (rlf_risk >= 0.25)
   AND (eligible_candidates NOT EMPTY) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]
    
    base_confidence = 0.78
    
    # SOFT PENALTIES applied similarly
    in_dwell = (now_s − last_handover_time) < 1.0
    IF in_dwell THEN
        confidence = base_confidence * 0.55
        reason = "Proactive HO: High mobility + signal dip, CAUTION (recent HO)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.60
        reason = "Proactive HO: High mobility + signal dip, CAUTION (extreme ping-pong)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.75
        reason = "Proactive HO: High mobility + signal dip, CAUTION (moderate ping-pong)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: High mobility in critical zone with signal degradation"
    END IF
    
    RETURN
END IF

┌─ PROACTIVE CONDITION 3: Fair Signal + Meaningful RLF Risk
│
IF (signal_quality == "FAIR") AND (rlf_risk >= 0.25) AND (rlf_risk < 0.50)
   AND (eligible_candidates NOT EMPTY) THEN
    target_idx = eligible_candidates[0]
    action = target_idx + 1
    target_cell_id = neighbor_ids[target_idx]
    
    base_confidence = 0.75
    
    # SOFT PENALTIES
    in_dwell = (now_s − last_handover_time) < 1.0
    IF in_dwell THEN
        confidence = base_confidence * 0.50
        reason = "Proactive HO: Fair signal + RLF risk, CAUTION (within dwell)"
    ELSE IF ping_pong_risk >= 0.70 THEN
        confidence = base_confidence * 0.55
        reason = "Proactive HO: Fair signal + RLF risk, CAUTION (extreme ping-pong)"
    ELSE IF ping_pong_risk >= 0.40 THEN
        confidence = base_confidence * 0.70
        reason = "Proactive HO: Fair signal + RLF risk, CAUTION (moderate ping-pong)"
    ELSE
        confidence = base_confidence
        reason = "Proactive HO: Fair signal with moderate RLF risk, eligible neighbor ready"
    END IF
    
    RETURN
END IF

┌─ DEFAULT: STAY
│
action = 0
target_cell_id = None
confidence = CLAMP(1.0 − (rlf_risk + ping_pong_risk)/2, 0.4, 1.0)
reason = "No HO trigger: Signal adequate or no eligible candidates"
RETURN
END ALGORITHM
================================================================================