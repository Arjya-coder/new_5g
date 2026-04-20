"""
dqn_agent.py — Double DQN + Dueling Network + Prioritized Experience Replay
============================================================================
Architecture:
  - Dueling Double DQN (Van Hasselt et al. + Wang et al.)
  - Prioritized Experience Replay with SumTree (Schaul et al.)
  - n-step returns (n=5) for better credit assignment over ping-pong delay
  - Soft target network updates (Polyak averaging)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================================
# SUM TREE  — O(log n) priority sampling
# ============================================================================
class SumTree:
    """Binary sum tree for O(log n) weighted sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.ptr      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else \
               self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        leaf = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float):
        leaf = self._retrieve(0, s)
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]


# ============================================================================
# PRIORITIZED EXPERIENCE REPLAY BUFFER
# ============================================================================
class PrioritizedReplayBuffer:
    """
    PER buffer with SumTree.

    Parameters
    ----------
    capacity : int    — max transitions stored
    alpha    : float  — prioritization exponent (0 = uniform, 1 = full priority)
    beta_start/end    — IS correction weight annealing schedule
    """

    def __init__(self, capacity: int = 100_000,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_end: float   = 1.0,
                 beta_steps: int   = 200_000,
                 eps: float = 1e-6):
        self.tree       = SumTree(capacity)
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.eps        = eps
        self._step      = 0
        self._max_prio  = 1.0

    @property
    def beta(self) -> float:
        frac = min(self._step / self.beta_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, transition: tuple, priority: float | None = None):
        """Add transition. New transitions get max priority for guaranteed replay."""
        p = (priority + self.eps) ** self.alpha if priority else self._max_prio
        self.tree.add(p, transition)

    def sample(self, batch_size: int):
        """
        Returns
        -------
        transitions : list of tuples
        weights     : np.ndarray  importance-sampling weights
        leaf_ids    : list of int  for priority updates
        """
        self._step += 1
        total    = self.tree.total
        segment  = total / batch_size

        transitions, weights, leaf_ids = [], [], []
        beta = self.beta
        min_prob = self.tree.tree[self.tree.capacity - 1 + np.argmin(
            self.tree.tree[self.tree.capacity - 1: self.tree.capacity - 1 + self.tree.size]
        )] / total + 1e-8
        max_weight = (min_prob * self.tree.size) ** (-beta)

        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment)
            leaf, prio, data = self.tree.get(s)
            if data is None:
                # Buffer not full yet — sample randomly from existing
                leaf, prio, data = self.tree.get(np.random.uniform(0, total))
            prob   = prio / total + 1e-8
            weight = (prob * self.tree.size) ** (-beta) / max_weight
            transitions.append(data)
            weights.append(weight)
            leaf_ids.append(leaf)

        return transitions, np.array(weights, dtype=np.float32), leaf_ids

    def update_priorities(self, leaf_ids: list, td_errors: np.ndarray):
        for leaf, err in zip(leaf_ids, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(leaf, p)
            self._max_prio = max(self._max_prio, p)

    def __len__(self):
        return self.tree.size


# ============================================================================
# N-STEP RETURN BUFFER
# ============================================================================
class NStepBuffer:
    """Accumulates n transitions and returns n-step (s, a, R_n, s_n, done)."""

    def __init__(self, n: int = 5, gamma: float = 0.99):
        self.n     = n
        self.gamma = gamma
        self.buf   = []

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))
        if len(self.buf) < self.n and not done:
            return None
        # Compute n-step return
        R     = 0.0
        gamma = 1.0
        final_done = False
        for i, (_, _, ri, _, di) in enumerate(self.buf):
            R += gamma * ri
            gamma *= self.gamma
            if di:
                final_done = True
                break
        s0, a0 = self.buf[0][0], self.buf[0][1]
        s_n    = self.buf[-1][3]
        self.buf.pop(0)
        return (s0, a0, R, s_n, final_done)

    def flush(self):
        """Return all remaining partial sequences at episode end."""
        out = []
        while self.buf:
            R, gamma = 0.0, 1.0
            for _, _, ri, _, _ in self.buf:
                R += gamma * ri
                gamma *= self.gamma
            s0, a0 = self.buf[0][0], self.buf[0][1]
            s_n    = self.buf[-1][3]
            done   = self.buf[-1][4]
            out.append((s0, a0, R, s_n, done))
            self.buf.pop(0)
        return out


# ============================================================================
# DUELING DQN NETWORK
# ============================================================================
class DuelingQNetwork(keras.Model):
    """
    Dueling architecture:
        shared stem → [Value stream V(s)] + [Advantage stream A(s,a)]
        Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int = 256, name: str = "dueling_q"):
        super().__init__(name=name)
        self.action_dim = action_dim

        # Shared feature extractor
        self.fc1 = keras.layers.Dense(hidden, activation='relu')
        self.fc2 = keras.layers.Dense(hidden, activation='relu')
        self.ln  = keras.layers.LayerNormalization()

        # Value stream
        self.v1  = keras.layers.Dense(128, activation='relu')
        self.v_out = keras.layers.Dense(1)

        # Advantage stream
        self.a1  = keras.layers.Dense(128, activation='relu')
        self.a_out = keras.layers.Dense(action_dim)

    def call(self, state, training=False):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.ln(x, training=training)

        v = self.v_out(self.v1(x))                          # (B, 1)
        a = self.a_out(self.a1(x))                          # (B, A)
        # Dueling combine: Q = V + (A - mean(A))
        q = v + a - tf.reduce_mean(a, axis=-1, keepdims=True)
        return q                                             # (B, A)


# ============================================================================
# DOUBLE DQN AGENT
# ============================================================================
class DoubleDQNAgent:
    """
    Double DQN with:
      - Dueling Q-network
      - Prioritized Experience Replay
      - n-step returns
      - Soft target network updates
    """

    def __init__(self,
                 state_dim:   int   = 22,
                 action_dim:  int   = 55,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 buffer_size: int   = 100_000,
                 batch_size:  int   = 64,
                 n_step:      int   = 5,
                 eps_start:   float = 1.0,
                 eps_end:     float = 0.05,
                 eps_decay_steps: int = 50_000):

        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.n_step      = n_step

        # Networks
        self.online = DuelingQNetwork(state_dim, action_dim, name="online")
        self.target = DuelingQNetwork(state_dim, action_dim, name="target")
        # Warm up both networks with a dummy forward pass
        dummy = tf.zeros((1, state_dim))
        self.online(dummy); self.target(dummy)
        self._hard_update()

        self.optimizer = keras.optimizers.Adam(lr, clipnorm=10.0)

        # PER buffer
        self.buffer   = PrioritizedReplayBuffer(capacity=buffer_size)
        self.n_buf    = NStepBuffer(n=n_step, gamma=gamma)

        # Epsilon schedule
        self.eps      = eps_start
        self.eps_end  = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps
        self.steps    = 0

    # ── Action selection ────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy during training; pure greedy for evaluation."""
        if not greedy and np.random.random() < self.eps:
            return np.random.randint(self.action_dim)
        q = self.online(state[None].astype(np.float32), training=False)[0].numpy()
        return int(np.argmax(q))

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps - self.eps_decay)
        self.steps += 1

    # ── Store transition (with n-step wrapping) ─────────────────────────────
    def store(self, s, a, r, s_next, done):
        result = self.n_buf.push(s, a, r, s_next, done)
        if result is not None:
            self.buffer.push(result)
        if done:
            for exp in self.n_buf.flush():
                self.buffer.push(exp)

    # ── Training step ───────────────────────────────────────────────────────
    @tf.function
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights):
        # Double DQN: online selects action, target evaluates value
        next_q_online = self.online(next_states, training=False)
        best_actions  = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        next_q_target = self.target(next_states, training=False)
        batch_idx     = tf.range(tf.shape(best_actions)[0], dtype=tf.int32)
        gather_idx    = tf.stack([batch_idx, best_actions], axis=1)
        next_q_val    = tf.gather_nd(next_q_target, gather_idx)

        gamma_n = self.gamma ** self.n_step
        td_target = rewards + gamma_n * next_q_val * (1.0 - dones)
        td_target = tf.stop_gradient(td_target)

        with tf.GradientTape() as tape:
            q_vals   = self.online(states, training=True)
            act_idx  = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            q_chosen = tf.gather_nd(q_vals, act_idx)
            td_err   = td_target - q_chosen
            loss     = tf.reduce_mean(weights * tf.square(td_err))  # weighted MSE

        grads = tape.gradient(loss, self.online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        return loss, td_err

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        transitions, weights, leaf_ids = self.buffer.sample(self.batch_size)
        s, a, r, s_n, d = map(np.array, zip(*transitions))

        states      = tf.constant(np.stack(s),   dtype=tf.float32)
        actions     = tf.constant(a,             dtype=tf.int32)
        rewards     = tf.constant(r,             dtype=tf.float32)
        next_states = tf.constant(np.stack(s_n), dtype=tf.float32)
        dones       = tf.constant(d.astype(np.float32), dtype=tf.float32)
        is_weights  = tf.constant(weights,       dtype=tf.float32)

        loss, td_errors = self._compute_loss(states, actions, rewards, next_states, dones, is_weights)
        self.buffer.update_priorities(leaf_ids, td_errors.numpy())
        self._soft_update()
        return float(loss)

    # ── Target network updates ───────────────────────────────────────────────
    def _hard_update(self):
        self.target.set_weights(self.online.get_weights())

    def _soft_update(self):
        tau = self.tau
        for w_o, w_t in zip(self.online.weights, self.target.weights):
            w_t.assign(tau * w_o + (1 - tau) * w_t)

    # ── Serialization ───────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.online.save_weights(os.path.join(path, 'online.weights.h5'))
        self.target.save_weights(os.path.join(path, 'target.weights.h5'))
        np.save(os.path.join(path, 'meta.npy'),
                {'eps': self.eps, 'steps': self.steps})

    def load(self, path: str):
        dummy = tf.zeros((1, self.state_dim))
        self.online(dummy); self.target(dummy)
        self.online.load_weights(os.path.join(path, 'online.weights.h5'))
        self.target.load_weights(os.path.join(path, 'target.weights.h5'))
        meta = np.load(os.path.join(path, 'meta.npy'), allow_pickle=True).item()
        self.eps   = float(meta.get('eps', self.eps_end))
        self.steps = int(meta.get('steps', 0))
