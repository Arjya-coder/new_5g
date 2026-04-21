ALGORITHM 2 - PPO ONLINE LEARNING (RLF CRITICAL FIX VERSION)
LINE-BY-LINE TONE STYLE MATCH FOR algo_2.py

================================================================================

OVERVIEW:
This document describes the implemented, production-style Algorithm 2 workflow
as coded in algo_2.py, using the same explanatory rhythm as algo_2.md.

This version is intentionally a minimal-change RL training variant.
Main objective:
  Train PPO to adapt TTT/HYS while preserving stable HO/PP behavior,
  with a critical reward fix for RLF severity under high mobility.

Key points in this implementation:
  - Velocity-aware RLF penalty is applied in reward
  - HO penalty and ping-pong penalty are preserved
  - Control-interval rollout is used (step_n)
  - Actor-Critic PPO remains standard clipped PPO

================================================================================

INPUT (Per Interaction Step):

State s_t from TrainingEnv/RLModule (23 dimensions):

a) Serving and position context (8 dims):
   - rsrp_serving_norm in [0, 1]      (from -130 to -70 dBm)
   - sinr_serving_norm in [0, 1]      (from -20 to 10 dB)
   - cqi_serving_norm in [0, 1]       (CQI 0 to 15)
   - distance_serving_norm in [0, 1]  (distance / 500 m)
   - zone_one_hot (4 dims): [CELL_CENTER, HANDOFF_ZONE, CRITICAL_ZONE, EDGE_ZONE]

b) Neighbor margin context (3 dims):
   Top-3 neighbors by RSRP margin over serving:
   - margin_1_norm in [-1, 1]
   - margin_2_norm in [-1, 1]
   - margin_3_norm in [-1, 1]
   (fallback to -20 dB when not enough neighbors)

c) Signal/decision context (7 dims):
   - signal_quality_one_hot (3 dims): [POOR, FAIR, GOOD]
   - time_since_last_ho_norm in [0, 1]  (clipped at 10 s)
   - ttt_norm in [0, 1]                 (100 to 320 ms)
   - hys_norm in [0, 1]                 (2.0 to 5.0 dB)
   - rsrp_trend_norm in [-1, 1]         (delta over 0.1 s)

d) Scenario-rate context (5 dims):
   - velocity_norm in [0, 1]          (0 to 50 m/s)
   - neighbor_count_norm in [0, 1]    (K/10)
   - recent_rlf_rate in [0, 1]        (10-step window)
   - recent_pp_rate in [0, 1]         (10-step window)
   - recent_ho_rate in [0, 1]         (10-step window)

TOTAL: 23 dimensions

Algorithm1 outputs consumed by environment/reward flow:
  - action, target_cell_id
  - rsrp_serving_dbm, sinr_serving_db
  - zone, signal_quality
  - ho_occurred, serving_cell_id
  - velocity, distance_serving
  - neighbor summaries

Current rule parameters:
  - TTT_rule(t) from [100, 160, 200, 240, 320] ms
  - HYS_rule(t) in [2.0, 5.0] dB

================================================================================

OUTPUT (Per Interaction Step):

1. PPO action a_t in {0, 1, ..., 14}
2. Effective parameters:
   - TTT_eff(t) in [100, 160, 200, 240, 320]
   - HYS_eff(t) in [2.0, 5.0]
3. Value estimate V(s_t)
4. Transition tuple for PPO update

Environment-level outputs per step/control block:
  - reward (discounted accumulation over control interval)
  - rlf_count, pp_count, ho_count
  - done flag and next state

================================================================================

PROCESS:

Step 0: NETWORK INITIALIZATION

Actor Network pi_theta(s):
  Input: s_t (23 dims)
  - Dense(128, ReLU)
  - Dense(128, ReLU)
  - Dense(15, Linear logits)

Critic Network V_xi(s):
  Input: s_t (23 dims)
  - Dense(128, ReLU)
  - Dense(128, ReLU)
  - Dense(1, Linear)

Hyperparameters in current implementation:
  - Learning rate (actor): 1e-3
  - Learning rate (critic): 3e-3
  - PPO clip epsilon: 0.2
  - Discount gamma: 0.99
  - GAE lambda: 0.95
  - Gradient norm clip: 0.5
  - Batch size: 64
  - Epochs per update: 10
  - Target KL configured: 0.015
  - Entropy coefficient schedule: 0.05 -> 0.005
  - Default episodes: 350
  - Rollout horizon: 2048
  - Control interval: 5 ticks (default)

Step 1: ACTION DECODING (15-Action Grid)

Decode action a_t in {0..14} to deltas:

  0  -> (delta_ttt_step = -2, delta_hys_db = -0.5)
  1  -> (delta_ttt_step = -2, delta_hys_db =  0.0)
  2  -> (delta_ttt_step = -2, delta_hys_db = +0.5)

  3  -> (delta_ttt_step = -1, delta_hys_db = -0.5)
  4  -> (delta_ttt_step = -1, delta_hys_db =  0.0)
  5  -> (delta_ttt_step = -1, delta_hys_db = +0.5)

  6  -> (delta_ttt_step =  0, delta_hys_db = -0.5)
  7  -> (delta_ttt_step =  0, delta_hys_db =  0.0)   # NO-OP in HYS/TTT
  8  -> (delta_ttt_step =  0, delta_hys_db = +0.5)

  9  -> (delta_ttt_step = +1, delta_hys_db = -0.5)
  10 -> (delta_ttt_step = +1, delta_hys_db =  0.0)
  11 -> (delta_ttt_step = +1, delta_hys_db = +0.5)

  12 -> (delta_ttt_step = +2, delta_hys_db = -0.5)
  13 -> (delta_ttt_step = +2, delta_hys_db =  0.0)
  14 -> (delta_ttt_step = +2, delta_hys_db = +0.5)

Step 2: COMPUTE EFFECTIVE PARAMETERS

Given current rule parameters (TTT_rule, HYS_rule):

A) TTT update on discrete ladder:
  ttt_steps = [100, 160, 200, 240, 320]
  idx_new = CLAMP(idx_old + delta_ttt_step, 0, 4)
  TTT_eff = ttt_steps[idx_new]

B) HYS update as clipped continuous:
  HYS_eff = CLAMP(HYS_rule + delta_hys_db, 2.0, 5.0)

Step 3: ALGORITHM1 HANDOVER INFERENCE (Inside env.step)

Algorithm1 runs per 100 ms tick with:
  - EMA filtering (alpha = 0.2)
  - Zone and quality classification
  - Internal RLF-risk estimate
  - A3 hold timer logic
  - Candidate ranking by margin

Core decision in this implementation:
  - MUST_HO if POOR quality OR EDGE zone OR rlf_risk >= 0.60
  - If MUST_HO and eligible candidates exist: handover
  - If MUST_HO and no eligible candidate:
      fallback to best neighbor with rsrp > -115 dBm
  - Else proactive HO when rlf_risk >= 0.35 and not in dwell

Dwell in this file:
  in_dwell = (now_s - last_handover_time) < 0.5

Step 4: ONLINE TRAINING LOOP (MAIN LOOP)

FOR episode e = 1 to num_episodes (default 350):

  PHASE 1: RESET AND PREPARE
  - entropy_coef from linear schedule
  - reset TrainingEnv and Algorithm1 state
  - initialize rule parameters:
      TTT_rule = 160, HYS_rule = 3.0
  - clear trajectory buffer

  PHASE 2: COLLECT ROLLOUT WITH CONTROL INTERVAL

  WHILE tick < rollout_horizon:

    4A) POLICY INFERENCE
    - logits_t = actor(s_t)
    - V_t = critic(s_t)
    - sample action a_t from softmax(logits_t)
    - decode (delta_ttt_step, delta_hys_db)
    - no_op = true only when both deltas are zero

    4B) PARAMETER COMPUTE
    - (TTT_eff, HYS_eff) = compute_effective_parameters(...)

    4C) MULTI-STEP ENV INTERACTION
    - run step_n for up to control_interval_steps
    - collect:
        s_next, reward_accum,
        rlf_count, pp_count, ho_count,
        steps_executed, done

    4D) STORE TRANSITION
    D.append({
      s, a, r, V, log_prob, n_steps
    })

    4E) UPDATE EPISODE ACCUMULATORS
    - reward, RLF, PP, HO, episode length
    - carry TTT_rule and HYS_rule forward

    IF done: BREAK

  END WHILE

  PHASE 3: ADVANTAGE ESTIMATION (n-step aware)

  - bootstrap value = 0 if done else critic(last_state)

  FOR t from end to start:
    n = trajectories[t].n_steps
    gamma_n = gamma^n
    gae_discount = (gamma * lambda)^n

    delta_t = r_t + gamma_n * V_{t+1} - V_t
    gae_t = delta_t + gae_discount * gae_{t+1}

    G_t = gae_t + V_t
  END FOR

  - normalize all advantages
  - attach A_norm and G to each trajectory sample

  PHASE 4: PPO UPDATE
  - train_step(trajectories, entropy_coef)
  - mini-batch PPO over epochs

  PHASE 5: LOGGING AND CHECKPOINTING
  - append episode metrics
  - every 10 episodes: print summary and save checkpoint

END FOR

After training:
  - save final actor/critic
  - save metrics JSON
  - save parameter history JSON

================================================================================

Step 5: REWARD FUNCTION (RLF CRITICAL FIX IMPLEMENTED)

COMPUTE_REWARD(
  rlf_event,
  ping_pong_event,
  sinr_delta,
  sinr_current,
  handover_occurred,
  delta_ttt_step,
  delta_hys_db,
  recent_ho_count,
  velocity,
  no_op
)

  r_t = 0.0

  COMPONENT 1: RLF PENALTY (CRITICAL FIX)

  IF rlf_event == TRUE:
      base_rlf_penalty = -2.0
      velocity_scale = 1.0 + velocity / 50.0
      r_rlf = CLAMP(base_rlf_penalty * velocity_scale, -5.0, -1.5)
  ELSE:
      r_rlf = 0.0
  END IF

  Interpretation:
    velocity = 0   -> r_rlf = -2.0
    velocity = 50  -> r_rlf = -4.0
    velocity = 70  -> r_rlf = -4.8

  COMPONENT 2: PING-PONG PENALTY (UNCHANGED)

  IF ping_pong_event == TRUE:
      r_pp = -1.0
  ELSE:
      r_pp = 0.0
  END IF

  COMPONENT 3: SINR SIGNAL TERM (UNCHANGED)

  r_sinr_delta = CLAMP(0.1 * sinr_delta, -0.2, +0.2)
  r_sinr_abs   = CLAMP(((sinr_current + 20) / 30) * 0.1, 0.0, 0.1)
  r_sinr = r_sinr_delta + r_sinr_abs

  COMPONENT 4: HANDOVER PENALTY (UNCHANGED)

  IF handover_occurred == TRUE:
      r_ho = -0.5
  ELSE:
      r_ho = 0.0
  END IF

  COMPONENT 5: PARAMETER SMOOTHNESS PENALTY (UNCHANGED)

  r_smooth = -0.15 * (abs(delta_ttt_step) + abs(delta_hys_db))

  COMPONENT 6: NO-OP PENALTY (UNCHANGED)

  IF no_op == TRUE:
      r_noop = -0.01
  ELSE:
      r_noop = 0.0
  END IF

  COMPONENT 7: HO FREQUENCY PENALTY (UNCHANGED)

  r_ho_freq = -0.1 * recent_ho_count

  FINAL REWARD:

  r_t = r_rlf + r_pp + r_sinr + r_ho + r_smooth + r_noop + r_ho_freq
  r_t = CLAMP(r_t, -5.0, +1.0)

  RETURN r_t

END COMPUTE_REWARD

================================================================================

Step 6: PPO UPDATE DETAILS (Inside train_step)

FOR each epoch in epochs_per_update:
  SHUFFLE trajectory indices

  FOR each mini-batch:

    ACTOR LOSS (PPO clipped):
      ratio = exp(log_prob_new - log_prob_old)
      surr1 = ratio * A_norm
      surr2 = CLAMP(ratio, 1-eps, 1+eps) * A_norm
      loss_clip = -mean(min(surr1, surr2))

      entropy = -sum(p * log(p + 1e-8))
      loss_entropy = -entropy_coef * mean(entropy)

      loss_actor = loss_clip + loss_entropy

    CRITIC LOSS (value clipping style):
      V_new = critic(s)
      V_clipped = CLAMP(V_new, G-eps, G+eps)
      loss_value = 0.5 * mean(min((V_new-G)^2, (V_clipped-G)^2))

    BACKPROP:
      - compute gradients
      - clip by max_grad_norm = 0.5
      - apply optimizer step

  END FOR
END FOR

================================================================================

Step 7: EVALUATION (evaluate_agent)

FOR each eval episode:
  - reset environment
  - initialize TTT/HYS rules (160, 3.0)

  WHILE NOT done:
    IF greedy:
      action = argmax(actor_logits)
    ELSE:
      action sampled from softmax

    decode action
    compute TTT_eff/HYS_eff
    execute step_n
    accumulate reward and event counts
  END WHILE

  log: reward, RLF count, PP count, HO count
END FOR

RETURN evaluation metrics list

================================================================================

Step 8: FILE RUNTIME NOTES

The runtime banner in __main__ emphasizes:
  - this is an RLF-fix-focused version
  - formula: r_rlf = -2.0 * (1 + velocity/50)
  - HO and PP logic are intentionally preserved

================================================================================

HYPERPARAMETER TUNING GUIDE (For This Implementation):

If updates are unstable:
  - lower actor learning rate (for example 5e-4)
  - reduce entropy start coefficient
  - reduce clip epsilon (for example 0.15)

If policy becomes too passive:
  - reduce handover penalty magnitude
  - reduce ho_frequency penalty weight
  - increase rollout horizon for richer trajectories

If RLF remains high at high speeds:
  - increase base_rlf_penalty magnitude
  - increase velocity scaling factor
  - increase control frequency (lower control interval)

If ping-pong becomes frequent:
  - increase ping_pong penalty magnitude
  - increase dwell window in Algorithm1 logic
  - add stricter candidate stability filtering

================================================================================

END ALGORITHM 2 (RLF CRITICAL FIX VERSION, TONE-MATCHED)
