"""
Run.py — Simulation entry point for AEG paper experiments.

Algorithm lineup (5 algorithms, matching paper figures):
  ILP          — Integer Linear Programming link selection (= REPS baseline)
  Random       — Random link selection (50% per link), same path selection as ILP
  SP           — Shortest-path (greedy hop-count) routing
  AEG_LS       — AEG with RL-based Link Selection only (no caching, no proactive swap)
  AEG_PES      — Full AEG: RL link selection + entanglement caching + proactive swapping

Paper figures reproduced:
  Fig. 4  → entanglement lifetime sweep (runLabel index 8)
  Fig. 5  → requests/slot sweep (runLabel index 0)  — ablation: ILP, Random, AEG_LS, SP, AEG_PES
  Fig. 5b → swap probability sweep (runLabel index 4)
  Fig. 5c → alpha sweep (runLabel index 5)
  Fig. 6  → runtime comparison (same data as Fig. 5, logged separately)
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
ttime  = 10       # time slots per trial
ttime2 = 50
step   = 50
times  = 1        # independent trials (set to 10 for paper-quality averaging)
nodeNo = 50       # nodes (paper: 50-node Waxman network)
alpha_ = 0.0002   # default entanglement-generation alpha
degree = 6

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

# Which X-sweeps to actually run  (0=Fig5, 4=Fig5b, 5=Fig5c, 8=Fig4)
# runLabel = [0, 4, 5, 8]
runLabel = [0]

# Algorithms that use the shorter ttime2 window
toRunLessAlgos = ['ILP']


# ── Per-trial worker ──────────────────────────────────────────────────────────
def runThread(algo, requests, algoIndex, ttime, pid, resultDict, shared_data):
    timeSlot = ttime
    global ttime2
    if algo.name in toRunLessAlgos:
        timeSlot = min(ttime2, ttime)

    for i in range(timeSlot):
        result = algo.work(requests[i], i)

    resultDict[pid] = result

    success_req   = sum(result.successfulRequestPerRound[i] for i in range(timeSlot))
    max_key       = (algo.name + str(len(algo.topo.nodes))
                     + str(algo.topo.alpha) + str(algo.topo.q) + 'max_success')

    print(f'pid={pid}  algo={algo.name}  success={success_req}  '
          f'best_so_far={shared_data[max_key] / timeSlot:.1f}')

    # Save the DQN model when a new best is reached
    if hasattr(algo, 'entAgent') and algo.entAgent is not None:
        if success_req > shared_data[max_key]:
            algo.entAgent.save_model()
            shared_data[max_key] = success_req


# ── Single-parameter simulation run ──────────────────────────────────────────
def Run(numOfRequestPerRound=30, numOfNode=0, r=7, q=0.9, alpha=alpha_,
        SocialNetworkDensity=0.5, rtime=ttime, topo=None,
        FixedRequests=None, results=[]):

    if topo is None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6)
    numOfNode = len(topo.nodes)
    topo.setQ(q)
    topo.setAlpha(alpha)

    # ── Algorithm definitions ─────────────────────────────────────────────────
    algorithms = [
        # Paper baseline: Integer Linear Programming link selection
        ILP(copy.deepcopy(topo), name='ILP'),

        # Paper baseline: Random link selection (50% per link)
        RandomLinkSelection(copy.deepcopy(topo), name='Random'),

        # Paper baseline: Shortest path (greedy hop-count) routing
        SP(copy.deepcopy(topo), name='SP'),

        # AEG ablation: RL link selection only (no caching, no proactive swap)
        # AEG_LS(copy.deepcopy(topo), name='AEG_LS'),

        # Full AEG: RL link selection + entanglement caching + proactive swapping
        # AEG_PES(copy.deepcopy(topo), name='AEG_PES'),
    ]

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

    jobs = []
    pid  = 0
    for _ in range(times):
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
            algo     = copy.deepcopy(base_algo)
            requests = {i: [] for i in range(ttime_)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    requests[i].append(
                        (algo.topo.nodes[src], algo.topo.nodes[dst]))
            pid += 1
            job = multiprocessing.Process(
                target=runThread,
                args=(algo, requests, algoIndex, ttime_, pid,
                      resultDicts[algoIndex], shared_data))
            jobs.append(job)

    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(
            resultDicts[algoIndex].values(),
            numOfRequestPerRound,
            algorithms[0].topo)
    return results


# ── Per-sweep thread targets ──────────────────────────────────────────────────
def mainThreadReqPerTime(Xparam, topo, result):
    result.extend(Run(numOfRequestPerRound=Xparam, topo=copy.deepcopy(topo)))

def mainThreadNumOfNode(Xparam, result):
    result.extend(Run(numOfNode=Xparam))

def mainThreadSwapProb(Xparam, topo, result):
    result.extend(Run(q=Xparam, topo=copy.deepcopy(topo)))

def mainThreadAlpha(Xparam, topo, result):
    result.extend(Run(alpha=Xparam, topo=copy.deepcopy(topo)))

def mainThreadSwapFrac(Xparam, topo, result):
    topo.preSwapFraction = Xparam
    result.extend(Run(topo=copy.deepcopy(topo)))

def mainThreadEntanglementLifetime(Xparam, topo, result):
    topo.entanglementLifetime = Xparam
    result.extend(Run(topo=copy.deepcopy(topo)))

def mainThreadRequestTimeout(Xparam, topo, result):
    topo.requestTimeout = Xparam
    result.extend(Run(topo=copy.deepcopy(topo)))

def mainThreadPreSwapCapacity(Xparam, topo, result):
    topo.preswap_capacity = Xparam
    result.extend(Run(topo=copy.deepcopy(topo)))


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
