"""
EntanglementAgent — Per-edge value regressor for RL-based entanglement link selection.

Architecture: contextual-bandit / off-policy value regression.

State space (per edge): (N×3 + 2) × N matrix, where N = number of nodes.
    Row block 0..N-1   : Adjacency — how many assignable qubits each edge has
    Row block N..2N-1  : Request demand — how many pending requests route over each edge
    Row block 2N..3N-1 : Distance matrix — Euclidean distance between every node pair
    Row 3N             : One-hot target edge indicator (source / destination nodes = 1)
    Row 3N+1           : Remaining qubit capacity at each node

Network output: scalar value per edge (not 8 Q-values).
Loss: MSE directly against observed reward — no discount, no bootstrapping, no target network.

Training behaviour policy: uniform random action per edge (randint(0, MAX_ACTION)).
Evaluation policy: score all edges, sort descending by predicted value, greedily
    assign qubits to highest-value edges until node qubit budget is exhausted.

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

# Replay buffer: 50k transitions — provides diverse training data for the
# value regressor across many random-behavior slots.
REPLAY_MEMORY_SIZE     = 50_000

# Start training after 2000 transitions so the buffer holds a diverse sample
# before the first SGD step.
MIN_REPLAY_MEMORY_SIZE = 2_000

MINIBATCH_SIZE         = 64

# Maximum qubit action per edge — matches the original DQN action range [0..7].
# assignQubitEdge clamps to min(action, available_links) so this is an upper bound.
MAX_ACTION             = 7

if not os.path.isdir('models'):
    os.makedirs('models')


class EntanglementAgent:
    """
    Per-edge value regressor.  Behaviour policy during training is uniform-random;
    evaluation policy is greedy by predicted value.
    Used for AEG_LS in the Option-2 (contextual-bandit) variant.
    """

    def __init__(self, algo, pid: int = 0, global_slot_offset: int = 0,
                 eval_mode: bool = False):
        """
        Args:
            algo:               the routing algorithm instance (provides topo & state)
            pid:                process id (unused internally, kept for API compat)
            global_slot_offset: ignored in value-regressor mode (no epsilon schedule);
                                kept for API compatibility with Run.py
            eval_mode:          if True, learn_and_predict uses the greedy value policy
                                and update_reward is a no-op.
        """
        print(f'[EntanglementAgent] init  algo={algo.name}  '
              f'global_slot_offset={global_slot_offset}  eval_mode={eval_mode}')
        self.env = RoutingEnv(algo)
        N = self.env.SIZE

        self.OBSERVATION_SPACE_VALUES = (N * 3 + 2, N)

        self.model_name = (
            f"{algo.name}_{len(algo.topo.nodes)}"
            f"_{algo.topo.alpha}_{algo.topo.q}_EntanglementAgentValue.keras"
        )

        self.model   = self._create_model()
        self.eval_mode             = eval_mode
        self.replay_memory         = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.last_action_table: dict = {}   # edge → [(time_slot, state)]

    # ── Model construction ────────────────────────────────────────────────────

    def _create_model(self):
        try:
            model = load_model(self.model_name)
            expected_in  = self.OBSERVATION_SPACE_VALUES
            expected_out = 1
            actual_in  = model.input_shape[1:]
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
            Dense(1,  activation='linear'),
        ])
        model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])
        return model

    # ── Replay memory ─────────────────────────────────────────────────────────

    def _update_replay_memory(self, transition: tuple) -> None:
        self.replay_memory.append(transition)

    # ── Training step ─────────────────────────────────────────────────────────

    def _train(self, terminal_state: bool) -> None:
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        states    = np.array([t[0] for t in minibatch])
        rewards   = np.array([t[1] for t in minibatch],
                             dtype=np.float32).reshape(-1, 1)

        t0  = time.time()
        fit = self.model.fit(states, rewards,
                             batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        loss = fit.history['loss'][0]
        print(f'[EntanglementAgent] train step {time.time()-t0:.2f}s  '
              f'loss={loss:.4f}  buf={len(self.replay_memory)}')

    # ── Main learning loop (called once per time slot from p2) ───────────────

    def learn_and_predict(self) -> None:
        """
        Eval mode  — score every edge once, sort by predicted value, greedy-saturate.
        Train mode — behaviour policy is uniform-random per edge; states recorded
                     for reward attribution in update_reward().
        """
        t0        = time.time()
        edges     = self.env.algo.topo.edges
        time_slot = self.env.algo.timeSlot

        if self.eval_mode:
            states = [(edge, self.env.ent_state(edge, time_slot)) for edge in edges]
            values = self.model.predict(
                np.array([s for _, s in states]), verbose=0, batch_size=50
            ).flatten()
            ranked = sorted(
                zip(edges, [s for _, s in states], values),
                key=lambda x: x[2], reverse=True,
            )
            print(f'[EntanglementAgent] eval inference {time.time()-t0:.3f}s')
            while True:
                assigned_this_pass = False
                for edge, state, _v in ranked:
                    _, did_assign = self.env.assignQubitEdge(
                        edge, MAX_ACTION, time_slot)
                    if did_assign:
                        assigned_this_pass = True
                if not assigned_this_pass:
                    break
            return

        edge_states = [(edge, self.env.ent_state(edge, time_slot)) for edge in edges]
        print(f'[EntanglementAgent] train state-build {time.time()-t0:.3f}s')

        while True:
            assigned_this_pass = False
            for edge, state in edge_states:
                action = np.random.randint(0, MAX_ACTION + 1)
                _, did_assign = self.env.assignQubitEdge(edge, action, time_slot)
                if did_assign:
                    assigned_this_pass = True
                    self.last_action_table.setdefault(edge, []).append(
                        (time_slot, state))
            if not assigned_this_pass:
                break

    # ── Reward update (called once per time slot from p4) ────────────────────

    def update_reward(self) -> None:
        """
        After p4 has run BSM swapping and recorded rewards in topo.reward_ent:
          1. Look up reward for each edge that was assigned this slot
          2. Push (state, reward) into replay memory
          3. Run one SGD training step on a random minibatch
          4. Prune old entries from last_action_table (beyond entanglement lifetime)
        """
        if self.eval_mode:
            return
        t0 = time.time()
        print(f'[EntanglementAgent] update_reward — {len(self.last_action_table)} edges')

        for edge, history in self.last_action_table.items():
            for time_slot, state in history:
                reward = self.env.find_reward_ent(edge, time_slot, action=0)
                self._update_replay_memory((state, reward))

        self.env.algo.topo.reward_ent = {}
        self._train(terminal_state=False)

        current_slot = self.env.algo.timeSlot
        lifetime = self.env.algo.topo.entanglementLifetime
        for edge in self.last_action_table:
            self.last_action_table[edge] = [
                e for e in self.last_action_table[edge]
                if current_slot - e[0] < lifetime
            ]
        print(f'[EntanglementAgent] update_reward done {time.time()-t0:.2f}s')

    # ── Persistence & FedAvg support ─────────────────────────────────────────

    def save_model(self) -> None:
        self.model.save(self.model_name)
        print(f'[EntanglementAgent] model saved: {self.model_name}')

    def save_model_to(self, path: str) -> None:
        self.model.save(path)

    def get_weights(self) -> list:
        return [w.copy() for w in self.model.get_weights()]

    def set_weights(self, weights: list) -> None:
        self.model.set_weights(weights)
