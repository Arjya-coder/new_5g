import os
import json
from datetime import datetime
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import the Phase-1 baseline so PPO adapts TTT/HYS on top of the same
# deterministic handover logic that Phase-1 evaluates against.
from phase_1.algo_1 import Algorithm1 as BaselineAlgorithm1


# ============================================================================
# RL MODULE
# ============================================================================

class RLModule:
    """
    PPO state + reward module.
    """

    def __init__(self):
        self.entropy_coef_start = 0.05
        self.entropy_coef_end = 0.005

    def build_state_vector(
        self,
        algo1_output,
        ttt_eff: int,
        hys_eff: float,
        time_since_last_ho=0.0,
        recent_rlf_count=0,
        recent_pp_count=0,
        recent_ho_count=0,
        rsrp_prev=None,
    ):
        state = []

        rsrp = float(algo1_output["rsrp_serving_dbm"])
        state.append(np.clip((rsrp - (-130.0)) / 60.0, 0.0, 1.0))

        sinr = float(algo1_output["sinr_serving_db"])
        state.append(np.clip((sinr - (-20.0)) / 30.0, 0.0, 1.0))

        cqi = int(algo1_output["cqi_serving"])
        state.append(np.clip(cqi / 15.0, 0.0, 1.0))

        distance = float(algo1_output["distance_serving"])
        state.append(np.clip(distance / 500.0, 0.0, 1.0))

        zone = algo1_output["zone"]
        zone_map = {
            "CELL_CENTER": [1, 0, 0, 0],
            "HANDOFF_ZONE": [0, 1, 0, 0],
            "CRITICAL_ZONE": [0, 0, 1, 0],
            "EDGE_ZONE": [0, 0, 0, 1],
        }
        state.extend(zone_map.get(zone, [0, 1, 0, 0]))

        rsrp_neighbors = algo1_output["rsrp_neighbors"]
        distance_neighbors = algo1_output["distance_neighbors"]
        neighbors_data = list(zip(rsrp_neighbors, distance_neighbors))
        neighbors_data.sort(key=lambda x: x[0], reverse=True)

        for i in range(3):
            if i < len(neighbors_data):
                margin = float(neighbors_data[i][0]) - rsrp
            else:
                margin = -20.0
            state.append(np.clip(margin / 30.0, -1.0, 1.0))

        signal_quality = algo1_output["signal_quality"]
        sq_map = {
            "POOR": [1, 0, 0],
            "FAIR": [0, 1, 0],
            "GOOD": [0, 0, 1],
        }
        state.extend(sq_map.get(signal_quality, [0, 1, 0]))

        state.append(np.clip(float(time_since_last_ho) / 10.0, 0.0, 1.0))

        # Keep PPO versatile range normalization
        state.append(np.clip((float(ttt_eff) - 100.0) / 220.0, 0.0, 1.0))
        state.append(np.clip((float(hys_eff) - 2.0) / 3.0, 0.0, 1.0))

        velocity = float(algo1_output["velocity"])
        state.append(np.clip(velocity / 50.0, 0.0, 1.0))

        neighbor_count = int(algo1_output["num_neighbors"])
        state.append(np.clip(neighbor_count / 10.0, 0.0, 1.0))

        state.append(np.clip(float(recent_rlf_count) / 10.0, 0.0, 1.0))
        state.append(np.clip(float(recent_pp_count) / 10.0, 0.0, 1.0))
        state.append(np.clip(float(recent_ho_count) / 10.0, 0.0, 1.0))

        if rsrp_prev is not None:
            rsrp_trend = (rsrp - float(rsrp_prev)) / 0.1
            rsrp_trend_norm = np.clip(rsrp_trend / 5.0, -1.0, 1.0)
        else:
            rsrp_trend_norm = 0.0
        state.append(rsrp_trend_norm)

        state = np.array(state, dtype=np.float32)
        assert len(state) == 23, f"State dim should be 23, got {len(state)}"
        return state

    def compute_reward(
        self,
        rlf_event,
        ping_pong_event,
        sinr_delta,
        sinr_current,
        handover_occurred,
        delta_ttt_step,
        delta_hys_db,
        recent_ho_count,
        velocity=0.0,
        no_op=False
    ):
        # Stronger RLF protection (velocity-aware)
        if rlf_event:
            base_rlf_penalty = -2.0
            velocity_scale = 1.0 + (float(velocity) / 50.0)
            r_rlf = np.clip(base_rlf_penalty * velocity_scale, -5.0, -1.5)
        else:
            r_rlf = 0.0

        r_pp = -1.0 if ping_pong_event else 0.0
        r_ho = -0.5 if handover_occurred else 0.0

        r_sinr_delta = np.clip(float(sinr_delta) * 0.1, -0.2, 0.2)
        r_sinr_abs = np.clip((float(sinr_current) + 20.0) / 30.0 * 0.1, 0.0, 0.1)
        r_sinr = r_sinr_delta + r_sinr_abs

        r_smooth = -0.15 * (abs(float(delta_ttt_step)) + abs(float(delta_hys_db)))
        r_noop = -0.01 if no_op else 0.0
        r_ho_freq = -0.1 * float(recent_ho_count)

        r = r_rlf + r_pp + r_sinr + r_ho + r_smooth + r_noop + r_ho_freq
        return float(np.clip(r, -5.0, 1.0))

    def get_entropy_coef(self, episode, total_episodes):
        progress = min(float(episode) / float(total_episodes), 1.0)
        return self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """
    PPO with 23-dim state and 15-action control grid.
    """

    def __init__(self, state_dim=23, action_dim=15, learning_rate_actor=1e-3, learning_rate_critic=3e-3):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

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
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='linear'),
        ])
        return model

    def _build_critic(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        return model

    def select_action(self, state):
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        logits = self.actor(state, training=False)[0].numpy()
        value = self.critic(state, training=False)[0].numpy()[0]

        probs = tf.nn.softmax(logits).numpy()
        action = int(np.random.choice(self.action_dim, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, float(value)

    def decode_action(self, action):
        # Keep your 15-action versatile grid
        grid = [
            (-2, -0.5), (-2, 0.0), (-2, +0.5),
            (-1, -0.5), (-1, 0.0), (-1, +0.5),
            (0, -0.5),  (0, 0.0),  (0, +0.5),
            (+1, -0.5), (+1, 0.0), (+1, +0.5),
            (+2, -0.5), (+2, 0.0), (+2, +0.5),
        ]
        return grid[int(action)]

    def compute_effective_parameters(self, ttt_rule, hys_rule, delta_ttt_step, delta_hys_db):
        # Keep versatile ladder
        ttt_steps = [100, 160, 200, 240, 320]

        if int(ttt_rule) not in ttt_steps:
            ttt_rule = 160

        idx = ttt_steps.index(int(ttt_rule))
        idx = max(0, min(4, idx + int(delta_ttt_step)))
        ttt_eff = int(ttt_steps[idx])

        hys_eff = float(np.clip(float(hys_rule) + float(delta_hys_db), 2.0, 5.0))
        return ttt_eff, hys_eff

    def train_step(self, trajectories, entropy_coef):
        states = np.array([t["s"] for t in trajectories], dtype=np.float32)
        actions = np.array([t["a"] for t in trajectories], dtype=np.int32)
        advantages = np.array([t["A_norm"] for t in trajectories], dtype=np.float32)
        returns = np.array([t["G"] for t in trajectories], dtype=np.float32)
        old_log_probs = np.array([t["log_prob"] for t in trajectories], dtype=np.float32)

        for _ in range(self.epochs_per_update):
            idx = np.arange(len(trajectories))
            np.random.shuffle(idx)

            for start in range(0, len(idx), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]

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
                    loss_entropy = -float(entropy_coef) * tf.reduce_mean(entropy)

                    loss_actor = loss_clip + loss_entropy

                grads = tape.gradient(loss_actor, self.actor.trainable_variables)
                grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in grads]
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    values = tf.squeeze(self.critic(s_batch, training=True), axis=1)
                    v_clipped = tf.clip_by_value(values, ret_batch - self.clip_eps, ret_batch + self.clip_eps)
                    loss_value = 0.5 * tf.reduce_mean(
                        tf.minimum((values - ret_batch) ** 2, (v_clipped - ret_batch) ** 2)
                    )

                grads = tape.gradient(loss_value, self.critic.trainable_variables)
                grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in grads]
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.actor.save(f"{path}/actor.h5")
        self.critic.save(f"{path}/critic.h5")

    def load(self, path):
        self.actor = keras.models.load_model(f"{path}/actor.h5")
        self.critic = keras.models.load_model(f"{path}/critic.h5")


# ============================================================================
# TRAINING ENVIRONMENT
# ============================================================================

class TrainingEnv:
    """
    Offline environment wrapper around Algorithm1.
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
        self.last_ho_time = -float("inf")

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
        algo1_output = self.ns3_env.step(ttt_eff, hys_eff, self.algo1)
        done = self.ns3_env.is_done()

        rsrp_current = float(algo1_output["rsrp_serving_dbm"])
        sinr_current = float(algo1_output["sinr_serving_db"])
        target_cell = algo1_output.get("target_cell_id", None)
        now_s = float(algo1_output["now_s"])
        zone = algo1_output["zone"]
        velocity = float(algo1_output["velocity"])
        distance = float(algo1_output["distance_serving"])
        ho_event = bool(algo1_output.get("ho_occurred", False))

        scenario_id = int(algo1_output.get("scenario_id", 1))
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

        if handover_occurred and self.serving_cell is not None and target_cell is not None:
            if (
                self.prev_serving_cell == target_cell
                and self.serving_cell != self.prev_serving_cell
                and now_s - self.last_ho_time < 1.0
            ):
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
            algo1_output=algo1_output,
            ttt_eff=int(ttt_eff),
            hys_eff=float(hys_eff),
            time_since_last_ho=float(self.time_since_last_ho),
            recent_rlf_count=int(recent_rlf_count),
            recent_pp_count=int(recent_pp_count),
            recent_ho_count=int(recent_ho_count),
            rsrp_prev=self.rsrp_prev,
        )

        self.sinr_prev = sinr_current
        self.rsrp_prev = rsrp_current
        self.time_since_last_ho += 0.1

        if handover_occurred and target_cell is not None:
            self.prev_serving_cell = self.serving_cell
            self.serving_cell = int(target_cell)
            self.target_cell = int(target_cell)
            self.last_ho_time = now_s
            self.time_since_last_ho = 0.0

        self.param_history.append({
            "timestamp": now_s,
            "ttt_eff": int(ttt_eff),
            "hys_eff": float(hys_eff),
            "zone": zone,
            "velocity": velocity,
            "distance": distance,
            "rsrp": rsrp_current,
            "sinr": sinr_current,
        })

        return state, rlf_event, ping_pong_event, sinr_delta, sinr_current, handover_occurred, recent_ho_count, done, velocity

    def step_n(self, ttt_eff, hys_eff, delta_ttt_step, delta_hys_db, no_op, n_steps: int, gamma: float = 0.99):
        n_steps = int(max(1, n_steps))
        reward_accum = 0.0
        discount = 1.0
        rlf_count = 0
        pp_count = 0
        ho_count = 0
        steps_executed = 0
        state_next = None
        done = False
        velocity_now = 0.0

        for micro in range(n_steps):
            state_next, rlf_event, ping_pong_event, sinr_delta, sinr_current, ho_occurred, recent_ho_count, done, velocity_now = self.step(ttt_eff, hys_eff)
            steps_executed += 1

            if micro == 0:
                dt_step = delta_ttt_step
                dh_db = delta_hys_db
                no_op_flag = bool(no_op)
            else:
                dt_step = 0
                dh_db = 0.0
                no_op_flag = False

            r_tick = self.rl_module.compute_reward(
                rlf_event=rlf_event,
                ping_pong_event=ping_pong_event,
                sinr_delta=sinr_delta,
                sinr_current=sinr_current,
                handover_occurred=ho_occurred,
                delta_ttt_step=dt_step,
                delta_hys_db=dh_db,
                recent_ho_count=recent_ho_count,
                velocity=velocity_now,
                no_op=no_op_flag,
            )

            reward_accum += discount * float(r_tick)
            discount *= float(gamma)

            rlf_count += 1 if rlf_event else 0
            pp_count += 1 if ping_pong_event else 0
            ho_count += 1 if ho_occurred else 0

            if done:
                break

        return state_next, float(reward_accum), int(rlf_count), int(pp_count), int(ho_count), int(steps_executed), bool(done)

    def reset(self):
        self.algo1 = BaselineAlgorithm1()
        self.time_since_last_ho = 0.0
        self.sinr_prev = None
        self.rsrp_prev = None
        self.prev_serving_cell = None
        self.serving_cell = None
        self.target_cell = None
        self.last_ho_time = -float("inf")
        self.rlf_counter = 0
        self.recent_rlf_events.clear()
        self.recent_pp_events.clear()
        self.recent_ho_events.clear()
        self.param_history = []
        self.in_handover = False

        algo1_output = self.ns3_env.reset(self.algo1)
        ttt_eff = 160
        hys_eff = 3.0
        state = self.rl_module.build_state_vector(
            algo1_output=algo1_output,
            ttt_eff=ttt_eff,
            hys_eff=hys_eff,
            time_since_last_ho=0.0,
            recent_rlf_count=0,
            recent_pp_count=0,
            recent_ho_count=0,
            rsrp_prev=None,
        )
        return state


# ============================================================================
# TRAIN / EVAL HELPERS
# ============================================================================

def train_ppo(agent, rl_module, training_env, num_episodes=350, rollout_horizon=2048,
              save_dir="models", log_dir="logs", control_interval_steps: int = 5):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    metrics_history = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                ttt_eff, hys_eff, delta_ttt, delta_hys, no_op, steps_to_run, gamma=agent.gamma
            )

            trajectories.append({
                "s": state,
                "a": action,
                "r": reward,
                "V": value,
                "log_prob": log_prob,
                "n_steps": steps_executed,
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
                next_value = trajectories[i + 1]["V"]

            n = int(trajectories[i].get("n_steps", 1))
            gamma_n = agent.gamma ** n
            gae_discount = (agent.gamma * agent.gae_lambda) ** n

            delta = trajectories[i]["r"] + gamma_n * next_value - trajectories[i]["V"]
            gae = delta + gae_discount * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + trajectories[i]["V"])

        advantages = np.array(advantages, dtype=np.float32)
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages_normalized = (advantages - adv_mean) / adv_std

        for i, traj in enumerate(trajectories):
            traj["A_norm"] = float(advantages_normalized[i])
            traj["G"] = float(returns[i])

        agent.train_step(trajectories, entropy_coef)

        metrics = {
            "episode": episode + 1,
            "total_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "rlf_count": int(episode_rlf),
            "pp_count": int(episode_pp),
            "ho_count": int(episode_ho),
            "avg_reward_per_step": float(episode_reward / max(1, episode_length)),
            "entropy_coef": float(entropy_coef),
            "ttt_final": int(ttt_rule),
            "hys_final": float(hys_rule),
        }
        metrics_history.append(metrics)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:4d} | Reward: {metrics['total_reward']:8.2f} | "
                f"RLF: {metrics['rlf_count']:2d} | PP: {metrics['pp_count']:2d} | "
                f"HO: {metrics['ho_count']:3d} | TTT: {ttt_rule} | HYS: {hys_rule:.2f}"
            )
            agent.save(f"{save_dir}/checkpoint_ep{episode + 1}")

    agent.save(f"{save_dir}/final")

    with open(f"{log_dir}/metrics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    with open(f"{log_dir}/param_history_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(training_env.param_history, f, indent=2)

    print("=" * 80)
    print(f"TRAINING COMPLETE - {num_episodes} EPISODES")
    print(f"Final model: {save_dir}/final")
    print(f"Logs: {log_dir}/metrics_{timestamp}.json")
    print("=" * 80)

    return metrics_history


def evaluate_agent(agent, rl_module, training_env, num_episodes=10, greedy=True, control_interval_steps: int = 5):
    all_metrics = []

    print("\n" + "=" * 80)
    print("EVALUATION PHASE")
    print("=" * 80)

    for ep in range(num_episodes):
        state = training_env.reset()

        total_reward = 0.0
        rlf_count = 0
        pp_count = 0
        ho_count = 0

        ttt_rule = 160
        hys_rule = 3.0

        control_interval_steps = int(max(1, control_interval_steps))

        while True:
            state_input = np.array(state, dtype=np.float32).reshape(1, -1)
            logits = agent.actor(state_input, training=False)[0].numpy()

            if greedy:
                action = int(np.argmax(logits))
            else:
                probs = tf.nn.softmax(logits).numpy()
                action = int(np.random.choice(agent.action_dim, p=probs))

            delta_ttt, delta_hys = agent.decode_action(action)
            no_op = (delta_ttt == 0 and delta_hys == 0)
            ttt_eff, hys_eff = agent.compute_effective_parameters(ttt_rule, hys_rule, delta_ttt, delta_hys)

            state_next, reward, rlf_c, pp_c, ho_c, steps_executed, done = training_env.step_n(
                ttt_eff, hys_eff, delta_ttt, delta_hys, no_op, control_interval_steps, gamma=agent.gamma
            )

            total_reward += reward
            rlf_count += int(rlf_c)
            pp_count += int(pp_c)
            ho_count += int(ho_c)

            ttt_rule = ttt_eff
            hys_rule = hys_eff
            state = state_next

            if done:
                break

        metrics = {
            "episode": ep + 1,
            "total_reward": float(total_reward),
            "rlf_count": int(rlf_count),
            "pp_count": int(pp_count),
            "ho_count": int(ho_count),
        }
        all_metrics.append(metrics)

        print(
            f"Eval {ep + 1:2d} | Reward: {total_reward:8.2f} | "
            f"RLF: {rlf_count:2d} | PP: {pp_count:2d} | HO: {ho_count:3d}"
        )

    print("=" * 80 + "\n")
    return all_metrics