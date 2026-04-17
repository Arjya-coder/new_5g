ALGORITHM 2 — PPO ONLINE LEARNING (REFINED & FINAL)
WITH ALL 5 REFINEMENTS INTEGRATED

================================================================================

OVERVIEW:
This is the complete, production-ready Algorithm 2 for training an RL agent
to learn optimal TTT/HYS parameters in a live ns-3 environment.

Key improvements:
  ✅ Removed confidence term from reward (cleaner signal)
  ✅ Added QoE/SINR improvement reward
  ✅ Balanced RLF vs ping-pong penalties
  ✅ Integrated curriculum learning
  ✅ Added soft MUST_HO bias (instead of hard restriction)

================================================================================

INPUT (Per Interaction Step):

State s_t from ns-3 (24 dimensions - LEAN & FOCUSED):

a) Serving cell features (5 dims):
   ├─ rsrp_serving_norm ∈ [0, 1]     (0 = -130dBm, 1 = -70dBm)
   ├─ sinr_serving_norm ∈ [0, 1]     (0 = -20dB, 1 = 10dB)
   ├─ cqi_serving_norm ∈ [0, 1]      (CQI 0→15)
   ├─ rlf_risk ∈ [0, 1]              (from Algorithm 1)
   └─ zone_one_hot (4 dims):         [CELL_CENTER, HANDOFF_ZONE, CRITICAL_ZONE, EDGE_ZONE]

b) Neighbor margin features (10 dims, 5 strongest neighbors):
   For each of top-5 neighbors by RSRP:
   ├─ margin_current (dB, clipped [-20, +20])
   └─ margin_predicted (dB, clipped [-20, +20])

c) Risk & decision context (6 dims):
   ├─ ping_pong_risk ∈ [0, 1]
   ├─ signal_quality_one_hot (3: POOR, FAIR, GOOD)
   ├─ MUST_HO flag (binary)
   └─ time_since_last_ho (normalized, clipped 0-10s)

d) Scenario context (3 dims):
   ├─ velocity_norm ∈ [0, 1]         (0 = stationary, 1 = 50 m/s)
   ├─ los_probability ∈ [0, 1]
   └─ neighbor_count_norm ∈ [0, 1]   (K/10)

TOTAL: 5 + 10 + 6 + 3 = 24 dimensions

Algorithm 1 outputs (for reward & diagnostics):
  ├─ action, confidence
  ├─ rlf_risk, ping_pong_risk, signal_quality
  ├─ sinr_serving (SINR at current step, for r_qoe)
  ├─ rsrp_serving (RSRP at current step, optional)
  ├─ a3_hold_progress_max, a5_hold_progress_max
  └─ eligible_candidates_count

Current parameters:
  ├─ TTT_rule(t) ∈ {100, 160, 320} ms
  └─ HYS_rule(t) ∈ [1, 6] dB

================================================================================

OUTPUT (Per Interaction Step):

1. Action a_t ∈ {0, 1, 2, ..., 8}  (discrete)
2. Effective parameters:
   ├─ TTT_eff(t) ∈ {100, 160, 320}
   └─ HYS_eff(t) ∈ [1, 6]
3. Value estimate: V(s_t)

================================================================================

PROCESS:

Step 0: NETWORK INITIALIZATION

Actor Network π_θ(s):
  Input: s_t (24 dims)
  ├─ Dense(128, ReLU)
  ├─ Dense(128, ReLU)
  └─ Dense(9, Linear)  ← logits for 9 actions
  
Critic Network V_ξ(s):
  Input: s_t (24 dims)
  ├─ Dense(128, ReLU)
  ├─ Dense(128, ReLU)
  └─ Dense(1, Linear)  ← scalar value

Hyperparameters:
  ├─ Learning rate (actor):  α_θ = 3e-4
  ├─ Learning rate (critic): α_ξ = 1e-3
  ├─ Entropy coefficient: β_entropy = 0.01
  ├─ PPO clip epsilon: ε = 0.2
  ├─ Discount γ: γ = 0.99
  ├─ GAE λ: λ = 0.95
  ├─ Target KL: target_kl = 0.015
  ├─ Batch size: 64
  ├─ Rollout horizon: 1024 steps
  ├─ Training epochs per update: 10
  └─ Max training episodes: 1000

Step 1: ACTION DECODING (Same as before)

Decode action a_t ∈ {0..8} to parameter deltas:

  a_t = 0 → (ΔTTT_step = -1, ΔHYS_db = -1)
  a_t = 1 → (ΔTTT_step = -1, ΔHYS_db =  0)
  a_t = 2 → (ΔTTT_step = -1, ΔHYS_db = +1)
  a_t = 3 → (ΔTTT_step =  0, ΔHYS_db = -1)
  a_t = 4 → (ΔTTT_step =  0, ΔHYS_db =  0)  ← NO-OP
  a_t = 5 → (ΔTTT_step =  0, ΔHYS_db = +1)
  a_t = 6 → (ΔTTT_step = +1, ΔHYS_db = -1)
  a_t = 7 → (ΔTTT_step = +1, ΔHYS_db =  0)
  a_t = 8 → (ΔTTT_step = +1, ΔHYS_db = +1)

Step 2: SOFT MUST_HO BIAS (NEW - REFINEMENT 5)

Instead of hard blocking, apply soft bias to action probabilities:

IF MUST_HO(t) == TRUE:
    
    IF rlf_risk(t) >= 0.80:
        # Extreme emergency: Strong bias toward aggressive (TTT-1)
        logits_bias = [+2.0, +2.0, +1.5,  0, 0, 0, -2.0, -2.0, -2.0]
        # Actions 0,1: (TTT-1, ...) get +2.0 boost
        # Actions 6,7,8: (TTT+1, ...) get -2.0 penalty
        
    ELSE IF rlf_risk(t) >= 0.50:
        # Moderate emergency: Gentle bias
        logits_bias = [+1.0, +0.5, +0.0,  0, 0, 0, -0.5, -1.0, -1.5]
        
    ELSE:
        # Low emergency: Minimal bias
        logits_bias = [+0.5, +0.2, +0.0,  0, 0, 0, -0.2, -0.5, -0.8]
    
    END IF
    
    logits_adjusted = logits_raw + logits_bias
    
ELSE:
    logits_adjusted = logits_raw  # No adjustment
END IF

# Sample from adjusted logits
a_t ~ Categorical(softmax(logits_adjusted))

Why this works:
  ✅ Prevents obviously wrong actions (TTT+1 when RLF critical)
  ✅ Still allows exploration
  ✅ Soft guidance, not hard blocking

Step 3: COMPUTE EFFECTIVE PARAMETERS

HYS_eff(t) = CLAMP(HYS_rule(t) + ΔHYS_db, 1, 6)

TTT_eff(t) = STEP(TTT_rule(t), ΔTTT_step):
  IF ΔTTT_step == -1:
      IF TTT_rule == 100: TTT_eff = 100
      IF TTT_rule == 160: TTT_eff = 100
      IF TTT_rule == 320: TTT_eff = 160
  ELSE IF ΔTTT_step == 0:
      TTT_eff = TTT_rule
  ELSE IF ΔTTT_step == +1:
      IF TTT_rule == 100: TTT_eff = 160
      IF TTT_rule == 160: TTT_eff = 320
      IF TTT_rule == 320: TTT_eff = 320
  END IF

Step 4: ONLINE TRAINING LOOP (MAIN LOOP)

FOR episode e = 1 to total_episodes (e.g., 1000):
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: SELECT SCENARIO (CURRICULUM LEARNING - REFINEMENT 4)
    # ═══════════════════════════════════════════════════════════════
    
    IF e <= 150:
        # Phase 1: Easy scenarios only
        scenario = RANDOM_CHOICE(["highway", "suburban"])
        curriculum_phase = "EASY"
        
    ELSE IF e <= 300:
        # Phase 2: Add medium scenarios
        scenario = RANDOM_CHOICE(["highway", "suburban", "mixed_urban", "canyon"])
        curriculum_phase = "MEDIUM"
        
    ELSE IF e <= 500:
        # Phase 3: Add hard scenarios
        scenario = RANDOM_CHOICE(["mixed_urban", "canyon", "dense_grid", "intersection"])
        curriculum_phase = "HARD"
        
    ELSE IF e <= 750:
        # Phase 4: Extreme scenario
        scenario = RANDOM_CHOICE(["tunnel", "nlos_heavy"])
        curriculum_phase = "EXTREME"
        
    ELSE:
        # Phase 5: Full random (all scenarios)
        scenario = RANDOM_CHOICE(all_scenarios)
        curriculum_phase = "FULL"
    
    END IF
    
    seed = RANDOM_SEED()
    
    env = NS3_ENVIRONMENT(scenario, seed)
    s_t = env.reset()
    
    D = []  # Rollout buffer
    episode_reward = 0.0
    episode_rlf_count = 0
    episode_ho_count = 0
    episode_length = 0
    
    done = FALSE
    
    # Store for computing r_qoe (QoE improvement reward)
    sinr_prev = None
    rsrp_prev = None
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: COLLECT ROLLOUT (Interact with ns-3)
    # ═══════════════════════════════════════════════════════════════
    
    FOR step t = 1 to max_rollout_length (e.g., 1024):
        
        # ─────────────────────────────────────────────────────────
        # STEP 4A: INFERENCE
        # ─────────────────────────────────────────────────────────
        
        logits_t = π_θ(s_t)
        V_t = V_ξ(s_t)
        
        probs_t = softmax(logits_t)
        a_t ~ Categorical(probs_t)
        log_prob_a_t = log(probs_t[a_t])
        
        # Decode action
        (ΔTTT_step, ΔHYS_db) = DECODE_ACTION(a_t)
        
        # Apply soft MUST_HO bias (Step 2)
        IF MUST_HO_signal from s_t:
            rlf_risk_val = s_t.rlf_risk
            (logits_adjusted, logits_bias) = APPLY_SOFT_BIAS(logits_t, rlf_risk_val)
            # Resample with adjusted logits (for consistency)
        ELSE:
            logits_adjusted = logits_t
        END IF
        
        # Compute effective parameters
        (TTT_eff, HYS_eff) = COMPUTE_EFFECTIVE_PARAMETERS(TTT_rule, HYS_rule, ΔTTT_step, ΔHYS_db)
        
        # ─────────────────────────────────────────────────────────
        # STEP 4B: ENVIRONMENT STEP
        # ─────────────────────────────────────────────────────────
        
        (s_next,
         algo1_action,
         algo1_confidence,
         rlf_risk,
         ping_pong_risk,
         signal_quality,
         sinr_current,        ← NEW (from Algo 1)
         rsrp_current,        ← NEW (for reference)
         a3_hold_progress,    ← NEW (from Algo 1)
         a5_hold_progress,    ← NEW (from Algo 1)
         handover_occurred,
         rlf_event,
         done,
         info) = env.step(TTT_eff, HYS_eff)
        
        # ─────────────────────────────────────────────────────────
        # STEP 4C: REWARD COMPUTATION (5 REFINEMENTS INTEGRATED)
        # ─────────────────────────────────────────────────────────
        
        r_t = COMPUTE_REWARD_REFINED(
            algo1_action,
            algo1_confidence,
            rlf_risk,
            ping_pong_risk,
            signal_quality,
            handover_occurred,
            rlf_event,
            sinr_current,
            sinr_prev,          ← For r_qoe
            rsrp_current,
            rsrp_prev,          ← For r_qoe (optional)
            scenario             ← For scenario-adaptive weights
        )
        
        # (See Step 5 for COMPUTE_REWARD_REFINED details)
        
        # ─────────────────────────────────────────────────────────
        # STEP 4D: STORE TRANSITION
        # ─────────────────────────────────────────────────────────
        
        D.append({
            's': s_t,
            'a': a_t,
            'r': r_t,
            'V': V_t,
            'logit': logits_t,
            'log_prob': log_prob_a_t,
            's_next': s_next,
            'done': done
        })
        
        episode_reward += r_t
        episode_rlf_count += (1 if rlf_event else 0)
        episode_ho_count += (1 if handover_occurred else 0)
        episode_length += 1
        
        # Update for next iteration
        s_t = s_next
        sinr_prev = sinr_current
        rsrp_prev = rsrp_current
        
        IF done:
            BREAK
        END IF
    
    END FOR (step loop)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: ADVANTAGE ESTIMATION
    # ═══════════════════════════════════════════════════════════════
    
    IF done:
        V_bootstrap = 0.0
    ELSE:
        V_bootstrap = V_ξ(s_t)
    END IF
    
    FOR t = len(D)-1 DOWN TO 0:
        IF t == len(D)-1:
            δ_t = D[t].r + γ * V_bootstrap - D[t].V
        ELSE:
            δ_t = D[t].r + γ * D[t+1].V - D[t].V
        END IF
        
        IF t == len(D)-1:
            A_t = δ_t
        ELSE:
            A_t = δ_t + γ * λ * A_{t+1}
        END IF
        
        G_t = A_t + D[t].V
        D[t].A = A_t
        D[t].G = G_t
    END FOR
    
    # Normalize advantages
    A_all = [D[t].A for t in 0..len(D)-1]
    A_mean = mean(A_all)
    A_std = std(A_all) + 1e-8
    FOR t = 0..len(D)-1:
        D[t].A_norm = (D[t].A - A_mean) / A_std
    END FOR
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: PPO UPDATE
    # ═══════════════════════════════════════════════════════════════
    
    old_logits = [D[t].logit for t in 0..len(D)-1]
    
    FOR epoch = 1 TO num_epochs:
        
        SHUFFLE(D)
        
        FOR batch_start = 0 TO len(D)-1 STEP batch_size:
            batch_end = min(batch_start + batch_size, len(D))
            batch = D[batch_start:batch_end]
            
            FOR sample IN batch:
                
                # Actor update (PPO clipped)
                logits_new = π_θ(sample.s)
                probs_new = softmax(logits_new)
                log_prob_new = log(probs_new[sample.a])
                
                ratio = exp(log_prob_new - sample.log_prob)
                L_clip = min(
                    ratio * sample.A_norm,
                    CLAMP(ratio, 1-ε, 1+ε) * sample.A_norm
                )
                
                entropy = -sum(probs_new * log(probs_new + 1e-8))
                loss_actor = -L_clip - β_entropy * entropy
                
                # Critic update
                V_new = V_ξ(sample.s)
                V_clipped = sample.V + CLAMP(V_new - sample.V, -ε, ε)
                
                L_value = 0.5 * min(
                    (V_new - sample.G)^2,
                    (V_clipped - sample.G)^2
                )
                
                loss_total = loss_actor + 0.5 * L_value
                
                # Backprop
                ∇_θ = COMPUTE_GRADIENTS(loss_total, θ)
                ∇_ξ = COMPUTE_GRADIENTS(loss_total, ξ)
                
                θ ← θ - α_θ * ∇_θ
                ξ ← ξ - α_ξ * ∇_ξ
            
            END FOR
        
        END FOR
        
        # Early stopping on KL
        kl_div = mean([KL(old_logits[i] || π_θ(D[i].s)) for i in 0..len(D)-1])
        
        IF kl_div > 1.5 * target_kl:
            LOG("KL divergence exceeded, stopping epoch early")
            BREAK
        END IF
    
    END FOR
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: LOGGING & MONITORING
    # ═══════════════════════════════════════════════════════════════
    
    LOG_METRICS({
        'episode': e,
        'curriculum_phase': curriculum_phase,
        'scenario': scenario,
        'seed': seed,
        'total_reward': episode_reward,
        'episode_length': episode_length,
        'rlf_count': episode_rlf_count,
        'ho_count': episode_ho_count,
        'avg_reward_per_step': episode_reward / episode_length,
        'kl_divergence': kl_div,
        'learning_rate_actor': α_θ,
        'learning_rate_critic': α_ξ
    })
    
    # Periodic evaluation
    IF e % 50 == 0:
        EVALUATE_ON_TEST_SET(π_θ, e)
    END IF

END FOR (episode loop)

================================================================================

Step 5: REWARD FUNCTION (ALL 5 REFINEMENTS INTEGRATED)

COMPUTE_REWARD_REFINED(algo1_action, algo1_confidence, rlf_risk, ping_pong_risk,
                        signal_quality, handover_occurred, rlf_event,
                        sinr_current, sinr_prev, rsrp_current, rsrp_prev, scenario):

  r_t = 0.0
  
  # ═════════════════════════════════════════════════════════════════
  # REFINEMENT 1: Removed r_conf completely (no confidence term)
  # ═════════════════════════════════════════════════════════════════
  
  # OLD: r_conf = ±0.15
  # NEW: No r_conf term at all
  
  # Reason: Confidence is Algorithm 1 heuristic, not ground truth
  
  # ═════════════════════════════════════════════════════════════════
  # COMPONENT 1: RLF PREVENTION (Primary objective)
  # ═════════════════════════════════════════════════════════════════
  
  IF rlf_event == TRUE:
      # RLF occurred: MAJOR penalty
      r_rlf = -2.3  ← REFINED (was -3.0, now -2.3 for balance)
  ELSE IF rlf_risk >= 0.70:
      r_rlf = -0.8
  ELSE IF rlf_risk >= 0.50:
      r_rlf = -0.4
  ELSE IF rlf_risk >= 0.30:
      r_rlf = -0.1
  ELSE IF rlf_risk < 0.20:
      r_rlf = +0.1
  ELSE:
      r_rlf = 0.0
  END IF
  
  # ═════════════════════════════════════════════════════════════════
  # COMPONENT 2: PING-PONG PREVENTION (Secondary objective)
  # ═════════════════════════════════════════════════════════════════
  
  IF ping_pong_risk >= 0.80:
      r_pp = -2.0
  ELSE IF ping_pong_risk >= 0.60:
      r_pp = -1.0
  ELSE IF ping_pong_risk >= 0.40:
      r_pp = -0.3
  ELSE:
      r_pp = +0.1
  END IF
  
  # ═════════════════════════════════════════════════════════════════
  # COMPONENT 3: QoE / SINR IMPROVEMENT (NEW - REFINEMENT 2)
  # ═════════════════════════════════════════════════════════════════
  
  # CRITICAL FIX: Reward actual signal improvement, not just stability
  
  IF sinr_prev is NOT NULL:
      sinr_delta = sinr_current - sinr_prev
      
      IF sinr_delta > 1.5:
          # Meaningful SINR improvement (>1.5dB)
          r_qoe = +0.20
      ELSE IF sinr_delta > 0.5:
          # Slight improvement (0.5-1.5dB)
          r_qoe = +0.08
      ELSE IF sinr_delta > -0.5:
          # Stable (no significant change)
          r_qoe = 0.0
      ELSE IF sinr_delta > -1.5:
          # Slight degradation (-0.5 to -1.5dB)
          r_qoe = -0.08
      ELSE:
          # Significant degradation (>-1.5dB)
          r_qoe = -0.20
      END IF
  ELSE:
      # First step: no previous SINR
      r_qoe = 0.0
  END IF
  
  # Alternative: Use RSRP if SINR not available
  # (Similar logic, thresholds ±2-3dB)
  
  # ═════════════════════════════════════════════════════════════════
  # COMPONENT 4: PROACTIVE HANDOVER BONUS
  # ═════════════════════════════════════════════════════════════════
  
  IF (signal_quality == "GOOD") AND (handover_occurred == TRUE):
      # Unnecessary HO in good signal
      r_proactive = -0.2
  ELSE IF (signal_quality == "FAIR") AND (algo1_action != 0) AND (rlf_risk < 0.50):
      # Proactive HO before emergency
      r_proactive = +0.3
  ELSE IF (signal_quality == "POOR") AND (algo1_action != 0):
      # HO after signal already poor (too late)
      r_proactive = -0.2
  ELSE:
      r_proactive = 0.0
  END IF
  
  # ═════════════════════════════════════════════════════════════════
  # COMPONENT 5: SAFETY GATE PENALTY
  # ═════════════════════════════════════════════════════════════════
  
  # Soft penalty if action violates MUST_HO guidance (from Step 2 bias)
  # Not a hard block, but a reward penalty
  
  IF MUST_HO == TRUE AND rlf_risk >= 0.60:
      IF a_t IN {6, 7, 8}:  # Actions that increase TTT
          r_safe = -0.3
      ELSE:
          r_safe = 0.0
      END IF
  ELSE:
      r_safe = 0.0
  END IF
  
  # ═════════════════════════════════════════════════════════════════
  # FINAL REWARD (5 Components)
  # ═════════════════════════════════════════════════════════════════
  
  # REFINEMENT 3: Scenario-adaptive weights (balanced RLF/PP)
  
  IF scenario IN ["dense_grid", "intersection"]:
      # High ping-pong risk in dense areas
      w_rlf = 1.5, w_pp = 3.0, w_proactive = 1.0, w_safe = 0.5, w_qoe = 1.5
  ELSE IF scenario IN ["highway"]:
      # RLF risk higher, ping-pong lower
      w_rlf = 3.0, w_pp = 1.0, w_proactive = 0.5, w_safe = 0.3, w_qoe = 1.5
  ELSE IF scenario IN ["suburban"]:
      # Balanced scenario
      w_rlf = 2.0, w_pp = 1.5, w_proactive = 1.0, w_safe = 0.5, w_qoe = 1.5
  ELSE IF scenario IN ["tunnel", "nlos_heavy"]:
      # Extreme RLF risk
      w_rlf = 4.0, w_pp = 0.5, w_proactive = 0.5, w_safe = 1.0, w_qoe = 1.0
  ELSE:
      # Default (urban canyon, mixed)
      w_rlf = 2.0, w_pp = 2.0, w_proactive = 1.0, w_safe = 0.5, w_qoe = 1.5
  END IF
  
  r_t = (w_rlf * r_rlf) + (w_pp * r_pp) + (w_qoe * r_qoe) 
        + (w_proactive * r_proactive) + (w_safe * r_safe)
  
  # Clamp to reasonable range
  r_t = CLAMP(r_t, -5.0, +1.0)
  
  RETURN r_t

END COMPUTE_REWARD_REFINED

================================================================================

Step 6: EVALUATION (During & After Training)

PHASE 6A: Periodic Evaluation (every 50 episodes)

FOR each test scenario (held-out scenarios):
    FOR num_test_runs (e.g., 3):
        env = NS3_ENVIRONMENT(test_scenario, fixed_seed)
        s_t = env.reset()
        
        WHILE NOT done:
            a_t = argmax_a π_θ(s_t)  # Deterministic (no sampling)
            (s_next, ..., rlf_event, done) = env.step(a_t)
            s_t = s_next
        END WHILE
        
        Collect metrics:
        ├─ total_reward
        ├─ rlf_count
        ├─ ho_count
        └─ avg_sinr_improvement
    END FOR
    
    LOG test performance
END FOR

PHASE 6B: Final Evaluation (after training)

Compare on held-out test set:

FOR each test scenario:
    
    RL Agent (learned π_θ):
    ├─ Run 10 episodes (different seeds)
    ├─ Collect: RLF rate, HO count, avg reward, avg SINR
    └─ Report: RL performance
    
    Baseline Algorithm 1 (fixed TTT=160, HYS=3):
    ├─ Run 10 episodes (same seeds)
    ├─ Collect: RLF rate, HO count, reward, avg SINR
    └─ Report: Baseline performance
    
    Calculate improvements:
    ├─ ΔRLFrate = (RLF_baseline - RLF_RL) / RLF_baseline × 100%
    ├─ ΔHOcount = (HO_baseline - HO_RL) / HO_baseline × 100%
    ├─ ΔReward = (Reward_RL - Reward_baseline) / |Reward_baseline| × 100%
    └─ ΔSINRavg = (SINR_RL - SINR_baseline) / |SINR_baseline| × 100%

END EVALUATION

================================================================================

HYPERPARAMETER TUNING GUIDE (For Fine-Tuning):

If training unstable:
  ├─ Lower β_entropy to 0.005 (reduce exploration noise)
  ├─ Lower α_θ to 1e-4 (more conservative updates)
  └─ Increase target_kl to 0.02 (allow larger KL divergence)

If convergence too slow:
  ├─ Increase batch_size to 128
  ├─ Increase num_epochs to 15
  └─ Lower ε (clip epsilon) to 0.15 for more aggressive updates

If agent becomes too conservative:
  ├─ Increase w_qoe to 2.0 (reward signal improvement more)
  ├─ Decrease r_rlf penalty (lower w_rlf)
  └─ Reduce r_pp penalty weight

If agent oscillates (ping-pong):
  ├─ Increase w_pp to 4.0 (heavily penalize ping-pong)
  ├─ Increase curriculum phase duration
  └─ Add more episodes to training (1500+ total)

================================================================================

END ALGORITHM 2 (FINAL, ALL 5 REFINEMENTS INTEGRATED)