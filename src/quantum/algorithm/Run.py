"""
Run.py — Simulation entry point for AEG paper experiments.

Algorithm lineup (6 algorithms — full ablation):
  ILP          — Integer Linear Programming link selection (= REPS baseline)
  Random       — Random link selection (50% per link), same path selection as ILP
  SP           — Shortest-path (greedy hop-count) routing
  AEG_LS       — AEG Link Selection only (RL, no caching, no proactive swap)
  AEG_EC       — AEG Entanglement Caching (RL + caching, no proactive swap)
  AEG_PES      — Full AEG: RL + entanglement caching + proactive swapping

Paper figures reproduced:
  Fig. 4  → entanglement lifetime sweep (runLabel index 8)
  Fig. 5  → requests/slot sweep (runLabel index 0)
  Fig. 5b → swap probability sweep (runLabel index 4)
  Fig. 5c → alpha sweep (runLabel index 5)
  Fig. 6  → runtime comparison (same data as Fig. 5, logged separately)

Recommended training settings for DQN convergence:
  ttime = 200, times = 10  →  2,000 slots per sweep value.
  The DQN model is saved/loaded across runs so learning accumulates.
  Minimum to see convergence: ttime ≥ 100 (otherwise replay buffer
  never fills MIN_REPLAY_MEMORY_SIZE=200 with ~150 edges/slot).
  Paper used ~200,000 total time slots for full convergence.
"""

import multiprocessing
import sys
import copy
sys.path.append("../..")

from AlgorithmBase import AlgorithmResult
from ILP      import ILP
from Random   import RandomLinkSelection
from SP       import SP
from AEG_LS   import AEG_LS
from AEG_EC   import AEG_EC
from AEG_PES  import AEG_PES
from SEER_cache3_3 import SEERCACHE3_3    # kept for GENI cloud experiments (silly.sh)
from CachedEntanglement import CachedEntanglement
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link

from random import sample
import numpy as np
import random
import time
import os.path

# ── Simulation parameters ─────────────────────────────────────────────────────
# FedAvg (Federated Averaging) parallel training for AEG-LS.
#
# Training structure:
#   rounds  — sequential FedAvg rounds; model accumulates across rounds
#   workers — parallel workers *within* each round; weights averaged after
#
#   Round 1: workers 0,1,2 run in parallel (different request patterns)
#            → each saves worker snapshot → main process averages weights
#            → shared model updated
#   Round 2: all workers load the averaged model → train 200 more slots
#            → average again → ...
#
# Total training = rounds × workers × ttime slots
#   Default (rounds=5, workers=3, ttime=200): 3,000 slots × ~150 edges
#   = 450,000 transitions  — 2× paper target, rounds × speedup vs sequential
#
# Epsilon continuity: each worker is initialised at the correct point in the
#   decay schedule via slot_offset = round_idx × ttime, so exploration decays
#   smoothly across rounds rather than restarting from EPSILON_START each round.
#
# Quick smoke-test: rounds=1, workers=2, ttime=20
ttime   = 200     # time slots per worker trial
ttime2  = 200     # cap for non-DQN algorithms (matches ttime when running AEG-LS only)
step    = 50      # timeslot chart sample interval → points at 0, 50, 100, 150
rounds  = 5       # sequential FedAvg rounds (model saved/averaged between rounds)
workers = 3       # parallel workers per round (set to cpu_count() for max speed)
nodeNo  = 50      # nodes (paper: 50-node Waxman network)
alpha_  = 0.0002  # default entanglement-generation alpha (P≈0.819 at 100 km)
degree  = 6

# Sweep ranges — one list per X-axis in the paper
numOfRequestPerRound  = [25, 30, 35]                    # Fig. 5 / Fig. 6
totalRequest          = [10, 20, 30, 40, 50]
numOfNodes            = [50, 75, 100]
r                     = [0, 2, 4, 6, 8, 10]
q                     = [0.7, 0.8, 0.9]                # Fig. 5b
alpha                 = [0.0001, 0.0002, 0.0003]        # Fig. 5c (×10⁻⁴ in paper)
SocialNetworkDensity  = [0.25, 0.5, 0.75, 1]
preSwapFraction       = [0.4, 0.6, 0.8, 1]
entanglementLifetimes = [1, 2, 3, 4, 5, 6, 7, 8]       # Fig. 4
requestTimeouts       = [100, 200, 300]
preSwapCapacity       = [0.2, 0.4, 0.5, 0.6, 0.8]

Xlabels = [
    "#RequestPerRound",     # 0  → Fig. 5 / Fig. 6
    "totalRequest",         # 1
    "#nodes",               # 2
    "r",                    # 3
    "swapProbability",      # 4  → Fig. 5b
    "alpha",                # 5  → Fig. 5c
    "SocialNetworkDensity", # 6
    "preSwapFraction",      # 7
    "entanglementLifetime", # 8  → Fig. 4
    "requestTimeout",       # 9
    "preSwapCapacity",      # 10
]

# Run all four paper sweeps: Fig.5 (requests), Fig.5b (swap prob),
# Fig.5c (alpha), Fig.4 (entanglement lifetime)
runLabel = [0, 4, 5, 8]

# No non-RL baselines in this run — no algorithms need a shortened window
toRunLessAlgos = []


# ── Per-trial worker ──────────────────────────────────────────────────────────
def runThread(algo, requests, algoIndex, ttime, pid, resultDict, shared_data,
              worker_model_path=None):
    """
    worker_model_path: if provided (FedAvg mode), always save the final model
    here regardless of performance so the main process can average weights.
    """
    timeSlot = ttime
    global ttime2
    if algo.name in toRunLessAlgos:
        timeSlot = min(ttime2, ttime)

    for i in range(timeSlot):
        result = algo.work(requests[i], i)

    resultDict[pid] = result

    success_req = sum(result.successfulRequestPerRound[i] for i in range(timeSlot))
    max_key     = (algo.name + str(len(algo.topo.nodes))
                   + str(algo.topo.alpha) + str(algo.topo.q) + 'max_success')

    print(f'pid={pid}  algo={algo.name}  success={success_req}  '
          f'best_so_far={shared_data[max_key] / timeSlot:.1f}')

    if hasattr(algo, 'entAgent') and algo.entAgent is not None:
        # FedAvg: always save worker snapshot so main process can average
        if worker_model_path:
            algo.entAgent.save_model_to(worker_model_path)
        # Also keep the global best for fallback / evaluation
        if success_req > shared_data[max_key]:
            algo.entAgent.save_model()
            shared_data[max_key] = success_req


# ── FedAvg weight averaging ───────────────────────────────────────────────────
def fedavg_models(worker_paths, shared_path):
    """
    Load models saved by each parallel worker, average their weights
    element-wise (FedAvg), and write the result to shared_path.
    This is the model that all workers will load at the start of the next round.
    Temporary worker files are deleted after averaging.
    """
    try:
        from tensorflow.keras.models import load_model as _load
    except ImportError:
        from keras.models import load_model as _load

    weight_lists = []
    valid_paths  = []
    for p in worker_paths:
        if os.path.exists(p):
            try:
                m = _load(p)
                weight_lists.append(m.get_weights())
                valid_paths.append(p)
            except Exception as e:
                print(f'[FedAvg] could not load {p}: {e}')

    if not weight_lists:
        print('[FedAvg] no valid worker snapshots — skipping averaging')
        return

    # Element-wise mean across all workers
    avg_weights = [
        np.mean([w[layer] for w in weight_lists], axis=0)
        for layer in range(len(weight_lists[0]))
    ]

    # Apply averaged weights to first valid model and save as shared model
    base = _load(valid_paths[0])
    base.set_weights(avg_weights)
    base.save(shared_path)
    print(f'[FedAvg] averaged {len(weight_lists)} workers → {shared_path}')

    # Clean up temporary worker snapshots
    for p in valid_paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _agent_model_path(algo_name, n_nodes, alpha_val, q_val):
    """Compute the shared model filename for a given (algo, topology) tuple."""
    return f'{algo_name}_{n_nodes}_{alpha_val}_{q_val}_EntanglementAgent.keras'


# ── Single-parameter simulation run ──────────────────────────────────────────
def Run(numOfRequestPerRound=30, numOfNode=0, r=7, q=0.9, alpha=alpha_,
        SocialNetworkDensity=0.5, rtime=ttime, topo=None,
        FixedRequests=None, results=[], name_suffix=''):

    if topo is None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6)
    numOfNode = len(topo.nodes)
    topo.setQ(q)
    topo.setAlpha(alpha)

    # ── AEG-LS only — full DQN training ──────────────────────────────────────
    # name_suffix makes each sweep value use a distinct .keras model file so
    # parallel sweep processes never overwrite each other's saved model.
    # To add baselines or other variants, uncomment the relevant lines below.
    algorithms = [
        AEG_LS(copy.deepcopy(topo), name=f'AEG_LS{name_suffix}'),

        # -- baselines (uncomment to compare) ---------------------------------
        # ILP(copy.deepcopy(topo),              name=f'ILP{name_suffix}'),
        # RandomLinkSelection(copy.deepcopy(topo), name=f'Random{name_suffix}'),
        # SP(copy.deepcopy(topo),               name=f'SP{name_suffix}'),

        # -- AEG ablation variants (uncomment to compare) ---------------------
        # AEG_EC(copy.deepcopy(topo),  param='ten', name=f'AEG_EC{name_suffix}'),
        # AEG_PES(copy.deepcopy(topo), param='ten', name=f'AEG_PES{name_suffix}'),
    ]

    # r and density are ILP-specific; set only if ILP is in the list
    if hasattr(algorithms[0], 'r'):
        algorithms[0].r       = r
        algorithms[0].density = SocialNetworkDensity

    global times
    results  = [[] for _ in range(len(algorithms))]
    ttime_   = rtime

    resultDicts = [multiprocessing.Manager().dict() for _ in algorithms]
    shared_data = multiprocessing.Manager().dict()
    for algo in algorithms:
        key = (algo.name + str(len(algo.topo.nodes))
               + str(algo.topo.alpha) + str(algo.topo.q) + 'max_success')
        shared_data[key] = 0

    pid = 0
    # ── FedAvg parallel training ──────────────────────────────────────────────
    # Structure:  rounds (sequential)  ×  workers (parallel within each round)
    #
    # Each round:
    #   1. `workers` processes start simultaneously, each training on an
    #      independently sampled request sequence — diversity helps generalisation
    #   2. Every worker saves its final model to a unique temp file
    #   3. Main process averages all worker weights (FedAvg)
    #   4. Averaged model saved as the shared .keras file
    #   5. Next round: every worker loads the shared model → continues training
    #
    # Epsilon continuity: each worker receives slot_offset = round_idx × ttime
    # so it initialises epsilon at the correct point in the global decay schedule
    # instead of restarting from EPSILON_START every round.
    global rounds, workers

    for round_idx in range(rounds):
        print(f'\n{"="*60}')
        print(f'[FedAvg] Round {round_idx + 1}/{rounds}  '
              f'({workers} parallel workers × {ttime_} slots)')
        print(f'{"="*60}')

        round_jobs          = []
        worker_model_paths  = [[] for _ in algorithms]   # per-algo worker snapshots
        slot_offset         = round_idx * ttime_         # for epsilon continuity

        for worker_i in range(workers):
            # Each worker gets a fresh random request sequence
            ids = {i: [] for i in range(ttime_)}
            if FixedRequests is not None:
                ids = FixedRequests
            else:
                for i in range(ttime_):
                    if i < rtime:
                        for _ in range(numOfRequestPerRound):
                            a = sample(range(numOfNode), 2)
                            ids[i].append((a[0], a[1]))

            for algoIndex, base_algo in enumerate(algorithms):
                algo = copy.deepcopy(base_algo)
                # Store slot_offset so prepare() initialises epsilon correctly
                algo.slot_offset = slot_offset

                requests = {i: [] for i in range(ttime_)}
                for i in range(rtime):
                    for (src, dst) in ids[i]:
                        requests[i].append(
                            (algo.topo.nodes[src], algo.topo.nodes[dst]))

                pid += 1
                # Temp snapshot path for this specific worker
                w_path = f'{algo.name}_w{worker_i}_r{round_idx}.keras'
                worker_model_paths[algoIndex].append(w_path)

                job = multiprocessing.Process(
                    target=runThread,
                    args=(algo, requests, algoIndex, ttime_, pid,
                          resultDicts[algoIndex], shared_data, w_path))
                round_jobs.append(job)

        # Start all workers in this round simultaneously
        for job in round_jobs:
            job.start()
        for job in round_jobs:
            job.join()   # wait for the full round to finish

        # FedAvg: average each algorithm's worker weights → update shared model
        for algoIndex, base_algo in enumerate(algorithms):
            if not hasattr(base_algo, 'entAgent'):
                continue   # non-DQN algorithm — no model to average
            shared = _agent_model_path(
                base_algo.name,
                len(base_algo.topo.nodes),
                base_algo.topo.alpha,
                base_algo.topo.q)
            fedavg_models(worker_model_paths[algoIndex], shared)

        print(f'[FedAvg] Round {round_idx + 1}/{rounds} complete')

    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(
            resultDicts[algoIndex].values(),
            numOfRequestPerRound,
            algorithms[0].topo)
    return results


# ── Per-sweep thread targets ──────────────────────────────────────────────────
# Each target appends a sweep-specific suffix to the algorithm name so that
# DQN models trained for different parameter values are stored in separate
# .keras files and do not overwrite each other.

def mainThreadReqPerTime(Xparam, topo, result):
    result.extend(Run(numOfRequestPerRound=Xparam,
                      topo=copy.deepcopy(topo),
                      name_suffix=f'_req{Xparam}'))

def mainThreadNumOfNode(Xparam, result):
    result.extend(Run(numOfNode=Xparam,
                      name_suffix=f'_n{Xparam}'))

def mainThreadSwapProb(Xparam, topo, result):
    result.extend(Run(q=Xparam, topo=copy.deepcopy(topo),
                      name_suffix=f'_q{Xparam}'))

def mainThreadAlpha(Xparam, topo, result):
    result.extend(Run(alpha=Xparam, topo=copy.deepcopy(topo),
                      name_suffix=f'_a{Xparam}'))

def mainThreadSwapFrac(Xparam, topo, result):
    topo.preSwapFraction = Xparam
    result.extend(Run(topo=copy.deepcopy(topo),
                      name_suffix=f'_sf{Xparam}'))

def mainThreadEntanglementLifetime(Xparam, topo, result):
    topo.entanglementLifetime = Xparam
    result.extend(Run(topo=copy.deepcopy(topo),
                      name_suffix=f'_lt{Xparam}'))

def mainThreadRequestTimeout(Xparam, topo, result):
    topo.requestTimeout = Xparam
    result.extend(Run(topo=copy.deepcopy(topo),
                      name_suffix=f'_rt{Xparam}'))

def mainThreadPreSwapCapacity(Xparam, topo, result):
    topo.preswap_capacity = Xparam
    result.extend(Run(topo=copy.deepcopy(topo),
                      name_suffix=f'_pc{Xparam}'))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Starting AEG simulation — generating paper results...')
    t1 = time.time()

    targetFilePath = '../../plot/data/'
    Ylabels        = AlgorithmResult().Ylabels

    Xparameters = [
        numOfRequestPerRound,   # 0
        totalRequest,           # 1
        numOfNodes,             # 2
        r,                      # 3
        q,                      # 4
        alpha,                  # 5
        SocialNetworkDensity,   # 6
        preSwapFraction,        # 7
        entanglementLifetimes,  # 8
        requestTimeouts,        # 9
        preSwapCapacity,        # 10
    ]

    topo = Topo.generate(nodeNo, 0.9, 5, alpha_, degree)

    for XlabelIndex, Xlabel in enumerate(Xlabels):
        if XlabelIndex not in runLabel:
            continue

        Ydata   = []
        jobs    = []
        results = {Xp: multiprocessing.Manager().list()
                   for Xp in Xparameters[XlabelIndex]}

        target_fn = {
            0: mainThreadReqPerTime,
            2: mainThreadNumOfNode,
            4: mainThreadSwapProb,
            5: mainThreadAlpha,
            7: mainThreadSwapFrac,
            8: mainThreadEntanglementLifetime,
            9: mainThreadRequestTimeout,
            10: mainThreadPreSwapCapacity,
        }.get(XlabelIndex)

        if target_fn is None:
            continue

        for Xp in Xparameters[XlabelIndex]:
            if XlabelIndex == 2:
                job = multiprocessing.Process(target=target_fn,
                                              args=(Xp, results[Xp]))
            else:
                job = multiprocessing.Process(target=target_fn,
                                              args=(Xp, topo, results[Xp]))
            jobs.append(job)

        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        for Xp in Xparameters[XlabelIndex]:
            Ydata.append(results[Xp])

        # ── Write timeslot success file ───────────────────────────────────────
        ts_filename  = 'Timeslot_#successRequest.txt'
        ts_filepath  = targetFilePath + ts_filename
        sampleRounds = list(range(0, ttime, step))
        print(f'\n{"="*60}')
        print(f'[WRITE] {ts_filepath}')
        print(f'{"="*60}')
        F = open(ts_filepath, 'w')
        for roundIndex in sampleRounds:
            Xaxis = str(roundIndex)
            Yaxis = []
            for result in Ydata[0]:
                try:
                    Yaxis.append(
                        sum(result.successfulRequestPerRound[
                            roundIndex:roundIndex + step]) / step)
                except Exception:
                    Yaxis.append(0)
            row = Xaxis + str(Yaxis).replace('[', ' ').replace(']', '\n').replace(',', '')
            F.write(row)
            print(f'  timeslot={roundIndex:>4}  values={[f"{v:.3f}" for v in Yaxis]}')
        F.close()
        print(f'[DONE]  {ts_filepath}')

        # ── Write per-metric files ────────────────────────────────────────────
        print(f'\n[WRITE] Per-metric files for X-axis: {Xlabel}')
        print(f'{"─"*60}')
        for Ylabel in Ylabels:
            filename = f'{Xlabel}_{Ylabel}.txt'
            filepath = targetFilePath + filename
            mode     = 'w' if os.path.isfile(filepath) else 'a'
            print(f'  {filename}  (mode={mode})')
            F = open(filepath, mode)
            for i, Xp in enumerate(Xparameters[XlabelIndex]):
                Xaxis = str(Xp)
                Yaxis = [ar.toDict()[Ylabel] for ar in Ydata[i]]
                row   = Xaxis + str(Yaxis).replace('[', ' ').replace(']', '\n').replace(',', '')
                F.write(row)
                algo_names = [type(a).__name__ for a in Ydata[i]] if Ydata[i] else []
                print(f'    X={Xp}  {Ylabel}={[f"{v:.4g}" for v in Yaxis]}')
            F.close()

    elapsed = time.time() - t1
    print(f'\n{"="*60}')
    print(f'Done — total time: {elapsed/3600:.2f} h  ({elapsed:.1f} s)')
    print(f'{"="*60}')
