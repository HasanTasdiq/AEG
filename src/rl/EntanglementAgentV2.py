"""
EntanglementAgentV2 — Reduced-state variant of EntanglementAgent.

State space (per edge): (N×2 + 2) × N matrix, where N = number of nodes.
    Row block 0..N-1   : Adjacency — how many assignable qubits each edge has
    Row block N..2N-1  : Request demand — how many pending requests route over each edge
    Row 2N             : One-hot target edge indicator (source / destination nodes = 1)
    Row 2N+1           : Remaining qubit capacity at each node

Differs from EntanglementAgent in two ways:
  1. No distance matrix in state (N×2+2 rows instead of N×3+2)
  2. Faster exploration: epsilon decays to 0 over 100 time slots (vs 250)

Use this when distance information adds too much noise or for faster training experiments.
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
# V2 variant: no distance matrix in state → smaller state space → can converge
# faster and tolerate a shorter epsilon schedule.

DISCOUNT               = 0.95
REPLAY_MEMORY_SIZE     = 10_000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE         = 64
UPDATE_TARGET_EVERY    = 100

EPSILON_START          = 0.5
EPSILON_MIN            = 0.05   # floor for residual exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = 300    # simpler state → converges faster than V1's 500
EPSILON_DECAY_VALUE    = (EPSILON_START - EPSILON_MIN) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

random.seed(1)
np.random.seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')


class EntanglementAgentV2:
    """
    Reduced-state DQN agent for entanglement link selection.
    Uses topology + requests + qubit counts (no distance matrix).
    Faster exploration than EntanglementAgent.
    """

    def __init__(self, algo, pid=0):
        print(f'[EntanglementAgentV2] initialising for algorithm: {algo.name}')
        self.env = RoutingEnv(algo)
        N = self.env.SIZE

        # State: (N adjacency rows) + (N request rows) + 2 extra rows — NO distance block
        self.OBSERVATION_SPACE_VALUES = (N * 2 + 2, N)

        self.model_name = (
            f"{algo.name}_{len(algo.topo.nodes)}"
            f"_{algo.topo.alpha}_{algo.topo.q}_EntanglementAgentV2.keras"
        )

        self.model        = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory         = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.last_action_table     = {}
        self.link_qs               = {}
        self.epsilon               = EPSILON_START

    def _create_model(self):
        try:
            model = load_model(self.model_name)
            print(f'[EntanglementAgentV2] loaded saved model: {self.model_name}')
            return model
        except Exception:
            print('[EntanglementAgentV2] no saved model found — building new model')

        model = Sequential([
            Flatten(input_shape=self.OBSERVATION_SPACE_VALUES),
            Dense(72, activation='relu'),
            Dense(48, activation='relu'),
            Dense(24, activation='relu'),
            Dense(7,  activation='linear'),
        ])
        model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
        return model

    def _update_replay_memory(self, transition):
        self.replay_memory.append(transition)

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
        print(f'[EntanglementAgentV2] training step done in {time.time()-t0:.2f}s')

        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def _get_link_qs_batch(self, links, time_slot):
        if not links:
            return
        states = [(link, self.env.ent_state(link, time_slot)) for link in links]
        qs_all = self.model.predict(
            np.array([s for _, s in states]), verbose=0, batch_size=50)
        for i, (link, state) in enumerate(states):
            self.link_qs[link] = (state, qs_all[i])

    def learn_and_predict(self):
        while self._learn_and_predict_step():
            pass

    def _learn_and_predict_step(self):
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

        link_action_q.sort(key=lambda x: x[2], reverse=True)

        for link, action, q, state in link_action_q:
            next_state, did_assign = self.env.assignQubitEdge(link, action, time_slot)
            next_state = next_state if next_state is not None else state
            if did_assign:
                assignable = True
            self.last_action_table.setdefault(link, []).append(
                (action, time_slot, state, next_state))

        if START_EPSILON_DECAYING <= time_slot <= END_EPSILON_DECAYING:
            self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_DECAY_VALUE)

        self.link_qs = {}
        print(f'[EntanglementAgentV2] learn_and_predict step done in {time.time()-t0:.2f}s')
        return assignable

    def update_reward(self):
        t0 = time.time()
        print(f'[EntanglementAgentV2] update_reward — {len(self.last_action_table)} links')

        for link, history in self.last_action_table.items():
            for action, time_slot, state, next_state in history:
                reward = self.env.find_reward_ent(link, time_slot, action)
                if reward:
                    self._update_replay_memory((state, action, reward, next_state, False))

        self.env.algo.topo.reward_ent = {}
        self._train(terminal_state=False)

        lifetime = 10
        for link in self.last_action_table:
            self.last_action_table[link] = [
                e for e in self.last_action_table[link]
                if self.env.algo.timeSlot - e[1] < lifetime
            ]

        print(f'[EntanglementAgentV2] update_reward done in {time.time()-t0:.2f}s')

    def save_model(self):
        self.model.save(self.model_name)
        print(f'[EntanglementAgentV2] model saved: {self.model_name}')
