ALGORITHM 2 - PPO ONLINE LEARNING (RLF CRITICAL FIX VERSION)
WORKING DESCRIPTION ALIGNED TO algo_2.py

================================================================================

OVERVIEW:
This is the implemented Phase 2 training system in algo_2.py.
It combines:
  1) Algorithm1 (handover baseline logic)
  2) RLModule (state + reward)
  3) PPOAgent (policy/value learning)
  4) TrainingEnv (ns-3 wrapper for events and reward rollout)

Design intent in this version:
  - Keep handover and ping-pong control logic stable
  - Apply a minimal but critical fix to RLF penalty
  - Make RLF penalty velocity-aware (higher speed => stronger penalty)

Key implementation profile:
  - State size: 23
  - Action size: 15
  - PPO with clipped objective
  - Control-interval training using step_n (default 5 ticks per decision)

================================================================================

ALGORITHM 2 IN REQUESTED NUMBERED STRUCTURE:

1: theta <- Initialize weights of actor net
2: xi <- Initialize weights of critic net
3: for episode i = 1 to N_episodes do
4:   Reset environment to time step 0 and initialize TTT_rule <- 160, HYS_rule <- 3.0
5:   Choose one scenario/dataset trace and initialize RSRP/SINR mapping for this run
6:   D <- empty memory buffer
7:   while episode not done and rollout budget not exhausted do
8:      Observe state s_t (23-dim), then compute actor logits and critic value V(s_t)
9:      Sample action a_t, decode to (delta_ttt_step, delta_hys_db), and compute (TTT_eff, HYS_eff)
10:     Execute environment interaction through step_n(...) for control interval k
11:     Receive transition outputs: (s_{t+1}, r_t, rlf_count, pp_count, ho_count, done)
12:     Store sample tuple in D: (s_t, a_t, s_{t+1}, r_t, log_prob_t, V_t, n_steps)
13:     Update counters/metrics and set s_t <- s_{t+1}
14:     if strict_reset_mode and (HOF or PP occurs) then
15:         Termination - reset environment to time step 0
16:     end if
17:   end while
18:   Compute bootstrap value V_bootstrap (0 if done, else critic estimate on last state)
19:   Compute A_hat and returns for all samples in D using n-step GAE
20:   Normalize A_hat over D
21:   m <- number of samples stored in D
22:   Shuffle D and split into mini-batches of size n
23:   for u = 1 to ceil(m/n) do
24:      Select one mini-batch B_u containing n samples (or remaining samples)
25:      Calculate current actor probabilities and critic V values for B_u
26:      for each sample (vectorized in implementation) in B_u do
27:          Calculate PPO clipped objective L_CLIP using ratio and A_hat
28:          theta <- theta_new using actor gradient update + entropy term
29:          xi <- xi_new using critic clipped value loss update
30:      end for
31:   end for
32:   Log metrics, save periodic checkpoints, and write episode summary
33:   Clear memory D
34: end for

Implementation note:
- Current algo_2.py does not hard-stop every episode on PP/HOF by default; it mainly penalizes these in reward and continues until done/horizon.

================================================================================

INPUT (Per Tick / Per Control Decision):

A) Algorithm1 tick input (from ns-3 context):
1. rsrp_serving (dBm)
2. sinr_serving (dB)
3. cqi_serving (0-15)
4. distance_serving (meters)
5. rsrp_neighbors (list)
6. neighbor_ids (list)
7. distance_neighbors (list)
8. velocity (m/s)
9. now_s (seconds)
10. TTT_eff (ms)
11. HYS_eff (dB)

B) PPO control input:
1. state vector (23 dims)
2. current rule parameters:
   - ttt_rule (one of [100, 160, 200, 240, 320])
   - hys_rule (continuous, clipped to [2.0, 5.0])
3. control_interval_steps (default 5)
4. rollout_horizon (default 2048)

C) RLF threshold context (scenario-aware in TrainingEnv):
scenario_id -> rlf_threshold_dbm
  1:-122, 2:-121, 3:-124, 4:-122, 5:-125, 6:-123, 7:-118, default:-122

================================================================================

OUTPUT:

Per Algorithm1 tick:
1. action (0 stay, 1 handover)
2. target_cell_id (or None)
3. serving/neighbor radio diagnostics
4. zone and signal_quality
5. ho_occurred flag

Per PPO control decision:
1. selected discrete action a_t in [0..14]
2. decoded parameter deltas (delta_ttt_step, delta_hys_db)
3. effective parameters (ttt_eff, hys_eff)
4. value estimate V(s_t)
5. log probability for PPO update

Per training episode:
1. total_reward
2. episode_length
3. rlf_count
4. pp_count
5. ho_count
6. avg_reward_per_step
7. entropy coefficient
8. final TTT/HYS values

Saved artifacts:
1. checkpoints every 10 episodes
2. final actor/critic model
3. metrics_<timestamp>.json
4. param_history_<timestamp>.json (from current env object)

================================================================================

PROCESS:

Step 0: Initialize Components

0A) Algorithm1 persistent state:
  - rsrp_s_f, sinr_s_f
  - rsrp_n_f[nid]
  - a3_hold_ms[nid]
  - previous_strongest_neighbor
  - recent_handoffs
  - last_handover_time
  - rsrp_history_serving
  - rsrp_history_neighbors[nid]
  - serving_cell_id

0B) RLModule settings:
  - entropy_coef_start = 0.05
  - entropy_coef_end = 0.005
  - recent event buffers (maxlen=10)

0C) PPOAgent defaults:
  - state_dim = 23
  - action_dim = 15
  - actor lr = 1e-3
  - critic lr = 3e-3
  - gamma = 0.99
  - gae_lambda = 0.95
  - clip_eps = 0.2
  - max_grad_norm = 0.5
  - batch_size = 64
  - epochs_per_update = 10
  - target_kl = 0.015

0D) Networks:
Actor pi_theta(s):
  Dense(128, relu) -> Dense(128, relu) -> Dense(15, linear logits)

Critic V_xi(s):
  Dense(128, relu) -> Dense(128, relu) -> Dense(1, linear)

--------------------------------------------------------------------------------

Step 1: Algorithm1 Tick Logic (Handover Baseline)

1A) Sanitize and cleanup:
  K = min(len(neighbor_ids), len(rsrp_neighbors), len(distance_neighbors))
  truncate neighbor arrays to K
  remove disappeared neighbors from:
    - rsrp_n_f
    - a3_hold_ms
    - rsrp_history_neighbors

1B) EMA filtering (alpha = 0.2):
  if first tick:
    rsrp_s_f = rsrp_serving
    sinr_s_f = sinr_serving
  else:
    rsrp_s_f = alpha*rsrp_serving + (1-alpha)*rsrp_s_f
    sinr_s_f = alpha*sinr_serving + (1-alpha)*sinr_s_f

  for each neighbor nid:
    initialize or EMA-update rsrp_n_f[nid]

1C) History update:
  append (now_s, value) and keep only last 5 seconds

1D) Signal quality classification:
  if rsrp_s_f < -120 or sinr_s_f < -10 or cqi < 5:
    signal_quality = POOR
  elif rsrp_s_f >= -100 and sinr_s_f >= 0 and cqi >= 10:
    signal_quality = GOOD
  else:
    signal_quality = FAIR

1E) Zone classification:
  if distance < 350: CELL_CENTER
  elif distance < 480: HANDOFF_ZONE
  elif distance <= 500: CRITICAL_ZONE
  else: EDGE_ZONE

1F) RLF risk estimate (internal, used for decision):
  rlf_risk = 0
  if rsrp_s_f < -120: +0.40
  elif rsrp_s_f < -110: +0.20
  if sinr_s_f < -10: +0.40
  elif sinr_s_f < -5: +0.20
  if distance >= 480: +0.05
  clamp to [0,1]

1G) Dwell and emergency flag:
  in_dwell = (now_s - last_handover_time) < 0.5
  MUST_HO = (signal_quality == POOR) or (zone == EDGE_ZONE) or (rlf_risk >= 0.60)

1H) Candidate eligibility (A3-only hold in this file):
  - max neighbor distance = 1500m
  - A3_condition = (rsrp_n_f[nid] - rsrp_s_f) >= HYS_eff
  - a3_hold_ms[nid] += 100 if true else reset to 0
  - eligible if:
      a3_hold_ms[nid] >= TTT_eff
      and current_strongest == nid
      and previous_strongest_neighbor == nid

1I) Candidate ranking:
  margin_pred[i] = rsrp_n_f[nid] - rsrp_s_f
  sort desc by:
    1) margin_pred
    2) neighbor rsrp
    3) shorter distance (implemented with -distance and reverse sort)

1J) Handover decision:
  default: action=0, target=None, ho_occurred=False

  if MUST_HO:
    if eligible exists:
      handover to best eligible
    else:
      fallback: choose best neighbor with rsrp_n_f > -115 (if available)

  elif rlf_risk >= 0.35 and eligible exists and not in_dwell:
    handover to best eligible

  on handover:
    action = 1
    target_cell_id updated
    append now_s to recent_handoffs
    last_handover_time = now_s
    serving_cell_id = target
    ho_occurred = True

  finally:
    previous_strongest_neighbor = current_strongest

--------------------------------------------------------------------------------

Step 2: Build 23-Dimensional RL State Vector

state = [
  1) rsrp_norm,
  2) sinr_norm,
  3) cqi_norm,
  4) distance_norm,
  5-8) zone one-hot (4 dims),
  9-11) top-3 neighbor margins normalized,
  12-14) signal quality one-hot (3 dims),
  15) time_since_last_ho_norm,
  16) ttt_norm,
  17) hys_norm,
  18) velocity_norm,
  19) neighbor_count_norm,
  20) recent_rlf_rate,
  21) recent_pp_rate,
  22) recent_ho_rate,
  23) rsrp_trend_norm
]

Normalization details:
  rsrp_norm = clip((rsrp - (-130))/60, 0, 1)
  sinr_norm = clip((sinr - (-20))/30, 0, 1)
  cqi_norm = clip(cqi/15, 0, 1)
  distance_norm = clip(distance/500, 0, 1)
  margin_norm = clip(margin/30, -1, 1)
  time_norm = clip(time_since_last_ho/10, 0, 1)
  ttt_norm = clip((ttt_eff-100)/(320-100), 0, 1)
  hys_norm = clip((hys_eff-2)/(5-2), 0, 1)
  velocity_norm = clip(velocity/50, 0, 1)
  neighbor_norm = clip(num_neighbors/10, 0, 1)
  rates = clip(count/10, 0, 1)
  rsrp_trend_norm = clip(((rsrp-rsrp_prev)/0.1)/5, -1, 1)

Assertions in code:
  - len(state) must be 23
  - each state value must be within [-1, 1]

--------------------------------------------------------------------------------

Step 3: Reward Function (Critical Fix Version)

compute_reward(
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

3A) Velocity-aware RLF penalty (core fix):
  if rlf_event:
    base = -2.0
    velocity_scale = 1 + velocity/50
    r_rlf = clip(base * velocity_scale, -5.0, -1.5)
  else:
    r_rlf = 0

  Examples implied by formula:
    velocity=0   -> -2.0
    velocity=50  -> -4.0
    velocity=70  -> -4.8

3B) Other components (kept stable):
  r_pp = -1.0 if ping_pong_event else 0

  r_sinr_delta = clip(sinr_delta * 0.1, -0.2, 0.2)
  r_sinr_abs = clip((sinr_current + 20)/30 * 0.1, 0, 0.1)
  r_sinr = r_sinr_delta + r_sinr_abs

  r_ho = -0.5 if handover_occurred else 0
  r_smooth = -0.15 * (abs(delta_ttt_step) + abs(delta_hys_db))
  r_noop = -0.01 if no_op else 0
  r_ho_freq = -0.1 * recent_ho_count

3C) Total reward:
  r = r_rlf + r_pp + r_sinr + r_ho + r_smooth + r_noop + r_ho_freq
  r = clip(r, -5.0, +1.0)

--------------------------------------------------------------------------------

Step 4: Action Space and Parameter Update

4A) Discrete action space (15 actions):
  action -> (delta_ttt_step, delta_hys_db)
  0: (-2, -0.5)   1: (-2, 0.0)   2: (-2, +0.5)
  3: (-1, -0.5)   4: (-1, 0.0)   5: (-1, +0.5)
  6: ( 0, -0.5)   7: ( 0, 0.0)   8: ( 0, +0.5)
  9: (+1, -0.5)  10: (+1, 0.0)  11: (+1, +0.5)
 12: (+2, -0.5)  13: (+2, 0.0)  14: (+2, +0.5)

4B) Effective parameter computation:
  allowed TTT steps = [100, 160, 200, 240, 320]
  move index by delta_ttt_step with clamping to [0,4]
  hys_eff = clip(hys_rule + delta_hys_db, 2.0, 5.0)

4C) Policy sampling:
  logits = actor(state)
  probs = softmax(logits)
  action sampled from categorical(probs)

--------------------------------------------------------------------------------

Step 5: TrainingEnv Single-Step Event Logic

TrainingEnv.step(ttt_eff, hys_eff):

1. call ns3_env.step(ttt_eff, hys_eff, algo1)
2. read algo1 output fields (rsrp, sinr, zone, velocity, now_s, target, ho flag)
3. detect RLF event:
   - if rsrp_current < scenario_threshold: rlf_counter += 1 else reset
   - rlf_event = (rlf_counter >= 2)
   - if triggered, reset rlf_counter to 0
4. detect handover_occurred using ho_event + in_handover guard
5. detect ping_pong_event if rapid return to previous serving cell within <1.0s
6. update recent event deques
7. compute sinr_delta
8. build next state using RLModule
9. update time_since_last_ho (+0.1 each step, reset on HO)
10. append parameter/radio snapshot to param_history
11. return state and events including velocity

--------------------------------------------------------------------------------

Step 6: Multi-Step Control Interval (step_n)

TrainingEnv.step_n(..., n_steps, gamma):

for micro-step in [0..n_steps-1]:
  - call step(ttt_eff, hys_eff)
  - reward per tick via compute_reward(..., velocity=velocity)
  - only first micro-step applies action deltas and no_op penalty
  - later micro-steps use dt_step=0, dh_db=0, no_op=False
  - accumulate discounted reward with gamma^micro
  - accumulate rlf/pp/ho counts
  - break early if done

returns:
  (state_next, reward_accum, rlf_count, pp_count, ho_count, steps_executed, done)

--------------------------------------------------------------------------------

Step 7: PPO Training Loop (train_ppo)

Defaults:
  num_episodes=350
  rollout_horizon=2048
  control_interval_steps=5

Per episode:
1. entropy_coef = linear decay from 0.05 to 0.005
2. reset env and initialize ttt_rule=160, hys_rule=3.0
3. rollout until horizon or done:
   - select action from actor
   - decode deltas
   - compute effective TTT/HYS
   - execute step_n over control interval
   - store trajectory item:
       s, a, r, V, log_prob, n_steps
   - update episode metrics
   - carry forward rule parameters
4. bootstrap value = 0 if done else critic(state)
5. compute n-step GAE backward:
   gamma_n = gamma^n_steps
   gae_discount = (gamma*lambda)^n_steps
   delta_t = r_t + gamma_n*V_next - V_t
   gae_t = delta_t + gae_discount*gae_{t+1}
6. normalize advantages
7. call agent.train_step(trajectories, entropy_coef)
8. log episode metrics
9. every 10 episodes:
   - print summary line
   - save checkpoint model

After all episodes:
  - save final model
  - dump metrics JSON
  - dump param_history JSON

--------------------------------------------------------------------------------

Step 8: PPO Update Details (agent.train_step)

For each epoch and mini-batch:

Actor loss:
  ratio = exp(log_prob_new - log_prob_old)
  surr1 = ratio * A_norm
  surr2 = clip(ratio, 1-eps, 1+eps) * A_norm
  loss_clip = -mean(min(surr1, surr2))

Entropy regularization:
  loss_entropy = -entropy_coef * mean(entropy)

Total actor loss:
  loss_actor = loss_clip + loss_entropy

Critic loss:
  values = critic(s)
  v_clipped = clip(values, returns-eps, returns+eps)
  loss_value = 0.5 * mean(min((values-returns)^2, (v_clipped-returns)^2))

Gradient handling:
  - compute gradients
  - clip each by max_grad_norm=0.5
  - apply Adam updates for actor and critic

--------------------------------------------------------------------------------

Step 9: Evaluation Loop (evaluate_agent)

Per evaluation episode:
1. reset env
2. run until done
3. choose action:
   - greedy=True: argmax(logits)
   - else: sample from softmax
4. decode and apply parameters through step_n
5. collect total_reward, rlf_count, pp_count, ho_count
6. print per-episode evaluation summary

returns list of episode metrics

--------------------------------------------------------------------------------

Step 10: Runtime Entry Behavior

In __main__, current file prints the RLF-fix summary banner.
The shown block is informational and describes:
  - velocity-aware RLF penalty formula
  - preserved HO/PP logic

================================================================================

NOTES ON WHAT CHANGED VS PRIOR VARIANTS:
1. Reward change is focused on RLF severity scaling with velocity.
2. Handover and ping-pong logic are intentionally preserved.
3. State/action/training pipeline remains stable in overall structure.
4. Control-interval rollout (step_n) is central for policy interaction timing.

================================================================================

END ALGORITHM 2 (RLF CRITICAL FIX IMPLEMENTATION)
