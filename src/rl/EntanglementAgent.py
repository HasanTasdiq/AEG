"""
EntanglementAgent — Deep Q-Network for RL-based entanglement link selection.

State space (per edge): (N×3 + 2) × N matrix, where N = number of nodes.
    Row block 0..N-1   : Adjacency — how many assignable qubits each edge has
    Row block N..2N-1  : Request demand — how many pending requests route over each edge
    Row block 2N..3N-1 : Distance matrix — Euclidean distance between every node pair
    Row 3N             : One-hot target edge indicator (source / destination nodes = 1)
    Row 3N+1           : Remaining qubit capacity at each node

This full state (including distances) is the primary agent described in the paper (Fig. 2).
Epsilon decays from 0.5 → 0 over 250 time slots.

Rewards (defined in RoutingEnv.find_reward_ent):
    +25  if the entangled link was used in a successful end-to-end connection
    -5   if the entangled link expired without being used
"""

import numpy as np
import time
import random
import os
from collections import deque

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Flatten
    from keras.optimizers import Adam

from RoutingEnv import RoutingEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Tuned for ttime=200, times=10 (2000 time slots per sweep value).
# The model is saved/loaded across runs so learning accumulates.
#
# Paper values: β=0.1 (learning rate, handled by Adam), γ=0.95 (discount),
#               200,000 total time slots.

DISCOUNT               = 0.95  # γ — paper value ✓

# Replay buffer: keep enough history for varied sampling.
# With ~150 edges × 200 slots = 30,000 transitions per trial; 10k cap avoids
# over-weighting stale transitions from early random exploration.
REPLAY_MEMORY_SIZE     = 10_000

# Start training quickly: with 150 edges/slot we hit 200 after 1–2 time slots,
# so the DQN begins learning almost immediately rather than waiting.
MIN_REPLAY_MEMORY_SIZE = 200

MINIBATCH_SIZE         = 64    # smaller batch → more frequent weight updates early on

# Sync the target network less often for more stable Bellman targets.
UPDATE_TARGET_EVERY    = 100

# Exploration schedule:
#   Explore for ~25% of the recommended 2000-slot run (first 500 slots ≈ 2–3 trials).
#   After that, exploit with 5% residual exploration so the policy never fully freezes.
EPSILON_START          = 0.5
EPSILON_MIN            = 0.05  # never drop below 5% random actions
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = 500   # full state (incl. distances) — decays slower
EPSILON_DECAY_VALUE    = (EPSILON_START - EPSILON_MIN) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

random.seed(1)
np.random.seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')


class EntanglementAgent:
    """
    DQN agent that decides which physical links to entangle each time slot.
    Uses the full state representation (topology + requests + distances + qubit counts).
    This is the primary agent used in AEG_LS and AEG_PES.
    """

    def __init__(self, algo, pid=0):
        print(f'[EntanglementAgent] initialising for algorithm: {algo.name}')
        self.env = RoutingEnv(algo)
        N = self.env.SIZE

        # State: (N adjacency rows) + (N request rows) + (N distance rows) + 2 extra rows
        self.OBSERVATION_SPACE_VALUES = (N * 3 + 2, N)

        self.model_name = (
            f"{algo.name}_{len(algo.topo.nodes)}"
            f"_{algo.topo.alpha}_{algo.topo.q}_EntanglementAgent.keras"
        )

        self.model        = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory       = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.last_action_table   = {}   # link → [(action, timeSlot, state, next_state)]
        self.link_qs             = {}   # link → (state, q_values)
        self.epsilon             = EPSILON_START

    # ── Model construction ────────────────────────────────────────────────────

    def _create_model(self):
        try:
            model = load_model(self.model_name)
            print(f'[EntanglementAgent] loaded saved model: {self.model_name}')
            return model
        except Exception:
            print('[EntanglementAgent] no saved model found — building new model')

        model = Sequential([
            Flatten(input_shape=self.OBSERVATION_SPACE_VALUES),
            Dense(72, activation='relu'),
            Dense(48, activation='relu'),
            Dense(24, activation='relu'),
            Dense(7,  activation='linear'),   # Q-values for action indices
        ])
        model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
        return model

    # ── Replay memory ─────────────────────────────────────────────────────────

    def _update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # ── Training step ─────────────────────────────────────────────────────────

    def _train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch       = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states  = np.array([t[0] for t in minibatch])
        new_states      = np.array([t[3] for t in minibatch])

        current_qs_list = self.model.predict(current_states, verbose=0, batch_size=64)
        future_qs_list  = self.target_model.predict(new_states, verbose=0, batch_size=64)

        X, y = [], []
        for idx, (state, action, reward, _, done) in enumerate(minibatch):
            new_q = reward if done else reward + DISCOUNT * np.max(future_qs_list[idx])
            qs         = current_qs_list[idx].copy()
            qs[action] = new_q
            X.append(state)
            y.append(qs)

        t0 = time.time()
        self.model.fit(np.array(X), np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        print(f'[EntanglementAgent] training step done in {time.time()-t0:.2f}s')

        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # ── Batch Q-value computation ─────────────────────────────────────────────

    def _get_link_qs_batch(self, links, time_slot):
        if not links:
            return
        states = [(link, self.env.ent_state(link, time_slot)) for link in links]
        qs_all = self.model.predict(
            np.array([s for _, s in states]), verbose=0, batch_size=50)
        for i, (link, state) in enumerate(states):
            self.link_qs[link] = (state, qs_all[i])

    # ── Main learning loop (called once per time slot in p2) ─────────────────

    def learn_and_predict(self):
        """Assign qubits to links greedily until no more assignments are possible."""
        while self._learn_and_predict_step():
            pass

    def _learn_and_predict_step(self):
        """One pass: predict Q-values, sort edges, assign qubits. Returns True if any assigned."""
        t0         = time.time()
        edges      = self.env.algo.topo.edges
        time_slot  = self.env.algo.timeSlot
        assignable = False

        self._get_link_qs_batch(edges, time_slot)

        link_action_q = []
        for link, (state, qs) in self.link_qs.items():
            if np.random.random() > self.epsilon:
                action = int(np.argmax(qs))
            else:
                action = np.random.randint(0, 2)
            link_action_q.append((link, action, qs[action], state))

        # Prioritise edges with highest predicted Q-value
        link_action_q.sort(key=lambda x: x[2], reverse=True)

        for link, action, q, state in link_action_q:
            next_state, did_assign = self.env.assignQubitEdge(link, action, time_slot)
            next_state = next_state if next_state is not None else state
            if did_assign:
                assignable = True
            entry = (action, time_slot, state, next_state)
            self.last_action_table.setdefault(link, []).append(entry)



        self.link_qs = {}
        print(f'[EntanglementAgent] learn_and_predict step done in {time.time()-t0:.2f}s')
        return assignable

    # ── Reward update (called once per time slot in p4) ──────────────────────

    def update_reward(self):
        """Look up rewards from topo.reward_ent, push to replay, run a training step."""
        t0 = time.time()
        print(f'[EntanglementAgent] update_reward — {len(self.last_action_table)} links')

        for link, history in self.last_action_table.items():
            for action, time_slot, state, next_state in history:
                reward = self.env.find_reward_ent(link, time_slot, action)
                if reward:
                    self._update_replay_memory((state, action, reward, next_state, False))

        self.env.algo.topo.reward_ent = {}
        self._train(terminal_state=False)

        # Prune entries older than entanglement lifetime
        lifetime = 10
        for link in self.last_action_table:
            self.last_action_table[link] = [
                e for e in self.last_action_table[link]
                if self.env.algo.timeSlot - e[1] < lifetime
            ]
        # Decay exploration
        if START_EPSILON_DECAYING <= time_slot <= END_EPSILON_DECAYING:
            self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_DECAY_VALUE)
        print(f'[EntanglementAgent] update_reward done in {time.time()-t0:.2f}s')

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self):
        self.model.save(self.model_name)
        print(f'[EntanglementAgent] model saved: {self.model_name}')
