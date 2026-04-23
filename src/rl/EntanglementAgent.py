"""
EntanglementAgent — Deep Q-Network for RL-based entanglement link selection.

State space (per edge): (N×3 + 2) × N matrix, where N = number of nodes.
    Row block 0..N-1   : Adjacency — how many assignable qubits each edge has
    Row block N..2N-1  : Request demand — how many pending requests route over each edge
    Row block 2N..3N-1 : Distance matrix — Euclidean distance between every node pair
    Row 3N             : One-hot target edge indicator (source / destination nodes = 1)
    Row 3N+1           : Remaining qubit capacity at each node

Full state (including distances) is the primary agent described in the paper (Fig. 2).
Epsilon decays from 1.0 → 0.05 over 50,000 time slots (~25% of paper's 200k horizon).

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
# Paper values: γ=0.95 (discount), β=0.1 (learning rate, handled by Adam).
#
# Memory notes: each transition stores two states of shape (N*3+2, N).
# For N=50: 152×50×4 bytes ≈ 30 KB per state → ~61 KB per transition.
# REPLAY_MEMORY_SIZE=2000 → ~122 MB per agent, safe for multi-worker runs.
# Reduce further (e.g. 1000) if memory pressure remains.

DISCOUNT               = 0.95

# Replay buffer: 50k cap — holds ~330 slots of history at ~150 transitions/slot.
# Provides diverse experience for stable gradient estimates during full training.
REPLAY_MEMORY_SIZE     = 50_000

# Start training after 2000 transitions (~13 slots) — proper warm-up before SGD.
MIN_REPLAY_MEMORY_SIZE = 2_000

MINIBATCH_SIZE         = 64

# Sync target network every 50 steps.
UPDATE_TARGET_EVERY    = 50

# Exploration schedule:
#   Full exploration (ε=1.0) decays to 0.05 over first 50,000 slots
#   (~25% of paper's 200,000-slot training horizon).
#   Residual 5% exploration prevents policy from fully freezing.
EPSILON_START          = 1.0
EPSILON_MIN            = 0.05   # floor — always keep some exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = 50_000
EPSILON_DECAY_VALUE    = (EPSILON_START - EPSILON_MIN) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

if not os.path.isdir('models'):
    os.makedirs('models')


class EntanglementAgent:
    """
    DQN agent that decides which physical links to entangle each time slot.
    Uses the full state representation (topology + requests + distances + qubit counts).
    Primary agent for AEG_LS, AEG_EC, and AEG_PES.
    """

    def __init__(self, algo, pid=0, global_slot_offset=0, eval_mode: bool = False):
        """
        Args:
            algo:               the routing algorithm instance (provides topo & state)
            pid:                process id (unused internally, kept for API compat)
            global_slot_offset: total slots already trained before this worker starts
                                (= round_idx × ttime in FedAvg mode).
                                Used to initialise epsilon at the correct point in the
                                decay schedule so exploration does not restart from
                                EPSILON_START=0.5 at the beginning of every round.
        """
        print(f'[EntanglementAgent] init  algo={algo.name}  '
              f'global_slot_offset={global_slot_offset}')
        self.env = RoutingEnv(algo)
        N = self.env.SIZE

        # State shape: (N adjacency + N requests + N distances + 2 extra) × N
        self.OBSERVATION_SPACE_VALUES = (N * 3 + 2, N)

        self.model_name = (
            f"{algo.name}_{len(algo.topo.nodes)}"
            f"_{algo.topo.alpha}_{algo.topo.q}_EntanglementAgent.keras"
        )

        self.model        = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.eval_mode             = eval_mode
        self.replay_memory         = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.last_action_table     = {}   # link → [(action, time_slot, state, next_state)]
        self.link_qs               = {}   # link → (state, q_values)

        # Compute starting epsilon from the global training position.
        # Without this, every FedAvg round would restart at 0.5 (pure exploration).
        elapsed      = max(0, global_slot_offset - START_EPSILON_DECAYING)
        self.epsilon = max(EPSILON_MIN, EPSILON_START - EPSILON_DECAY_VALUE * elapsed)
        print(f'[EntanglementAgent] starting epsilon = {self.epsilon:.4f}')

    # ── Model construction ────────────────────────────────────────────────────

    def _create_model(self):
        try:
            model = load_model(self.model_name)
            expected_in  = self.OBSERVATION_SPACE_VALUES
            expected_out = 8
            actual_in  = model.input_shape[1:]   # drop batch dim
            actual_out = model.output_shape[-1]
            if actual_in != expected_in or actual_out != expected_out:
                raise ValueError(
                    f'stale model: input {actual_in} (want {expected_in}), '
                    f'output {actual_out} (want {expected_out})')
            print(f'[EntanglementAgent] loaded saved model: {self.model_name}')
            return model
        except Exception as e:
            print(f'[EntanglementAgent] no saved model or shape mismatch ({e}) — building fresh')

        model = Sequential([
            Flatten(input_shape=self.OBSERVATION_SPACE_VALUES),
            Dense(72, activation='relu'),
            Dense(48, activation='relu'),
            Dense(24, activation='relu'),
            Dense(8,  activation='linear'),  # Q-values; 8 = l+1 where l_max=7 (paper §III-A)
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

        minibatch      = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([t[0] for t in minibatch])
        new_states     = np.array([t[3] for t in minibatch])

        current_qs_list = self.model.predict(current_states, verbose=0, batch_size=64)
        future_qs_list  = self.target_model.predict(new_states, verbose=0, batch_size=64)

        X, y = [], []
        for idx, (state, action, reward, _, done) in enumerate(minibatch):
            new_q      = reward if done else reward + DISCOUNT * np.max(future_qs_list[idx])
            qs         = current_qs_list[idx].copy()
            qs[action] = new_q
            X.append(state)
            y.append(qs)

        t0 = time.time()
        self.model.fit(np.array(X), np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        print(f'[EntanglementAgent] train step {time.time()-t0:.2f}s')

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

    # ── Main learning loop (called once per time slot from p2) ───────────────

    def learn_and_predict(self):
        """
        Run the RL model ONCE per timeslot to score all edges, then assign
        qubits greedily in Q-value priority order until no more assignments
        can be made — without calling the model again.

        This replaces the old multi-step loop that re-ran inference on every
        pass, cutting p2 runtime from O(passes × edges) model calls to O(1).
        """
        t0        = time.time()
        edges     = self.env.algo.topo.edges
        time_slot = self.env.algo.timeSlot

        # ── Single inference pass ─────────────────────────────────────────
        self._get_link_qs_batch(edges, time_slot)

        link_action_q = []
        for link, (state, qs) in self.link_qs.items():
            if np.random.random() > self.epsilon:
                action = int(np.argmax(qs))   # exploit
            else:
                action = np.random.randint(0, 8)  # explore full action space (0..l_max)
            link_action_q.append((link, action, qs[action], state))

        # Sort once: highest Q-value first — priority is fixed for this slot
        link_action_q.sort(key=lambda x: x[2], reverse=True)
        self.link_qs = {}
        print(f'[EntanglementAgent] inference {time.time()-t0:.3f}s  ε={self.epsilon:.4f}')

        # ── Greedy assignment loop (no further model calls) ───────────────
        # Re-visit edges in the same Q-value order each pass; stop when a
        # full pass produces no new assignment (capacity exhausted).
        while True:
            assigned_this_pass = False
            for link, action, q, state in link_action_q:
                next_state, did_assign = self.env.assignQubitEdge(link, action, time_slot)
                if did_assign:
                    next_state = next_state if next_state is not None else state
                    assigned_this_pass = True
                    self.last_action_table.setdefault(link, []).append(
                        (action, time_slot, state, next_state))
            if not assigned_this_pass:
                break

    # ── Reward update (called once per time slot from p4) ────────────────────

    def update_reward(self):
        """
        After p4 has run BSM swapping and recorded rewards in topo.reward_ent:
          1. Look up reward for each (link, action) pair taken this slot
          2. Push (state, action, reward, next_state) into replay memory
          3. Run one SGD training step on a random minibatch
          4. Prune old entries from last_action_table (beyond entanglement lifetime)

        Note: epsilon decay happens in _learn_and_predict_step, NOT here.
        """
        if self.eval_mode:
            return
        t0 = time.time()
        print(f'[EntanglementAgent] update_reward — {len(self.last_action_table)} links')

        for link, history in self.last_action_table.items():
            for action, time_slot, state, next_state in history:
                reward = self.env.find_reward_ent(link, time_slot, action)
                if reward:
                    self._update_replay_memory(
                        (state, action, reward, next_state, False))

        self.env.algo.topo.reward_ent = {}
        self._train(terminal_state=False)

        # Prune transitions older than the entanglement lifetime so the table
        # does not grow unboundedly across time slots.
        current_slot = self.env.algo.timeSlot
        lifetime = self.env.algo.topo.entanglementLifetime
        for link in self.last_action_table:
            self.last_action_table[link] = [
                e for e in self.last_action_table[link]
                if current_slot - e[1] < lifetime
            ]
        # Decay epsilon once per time slot (after all link decisions are made)
        if START_EPSILON_DECAYING <= current_slot <= END_EPSILON_DECAYING:
            self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_DECAY_VALUE)
        print(f'[EntanglementAgent] update_reward done {time.time()-t0:.2f}s')

    # ── Persistence & FedAvg support ─────────────────────────────────────────

    def save_model(self):
        """Save to the shared model file (loaded by the next trial/round)."""
        self.model.save(self.model_name)
        print(f'[EntanglementAgent] model saved: {self.model_name}')

    def save_model_to(self, path):
        """Save to an arbitrary path (FedAvg worker snapshots)."""
        self.model.save(path)

    def get_weights(self):
        """Return a copy of model weights for FedAvg averaging."""
        return [w.copy() for w in self.model.get_weights()]

    def set_weights(self, weights):
        """Apply averaged weights to both the prediction and target networks."""
        self.model.set_weights(weights)
        self.target_model.set_weights(weights)  # keep target in sync
