import multiprocessing
import sys
import copy
sys.path.append("../..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from REPS import REPS
from REPS_cache_ent_dqrl_proswap import REPSCACHEENT_DQRL_PSWAP
from SEER_cache3_3 import SEERCACHE3_3
from CachedEntanglement import CachedEntanglement
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link

from random import sample
from numpy import log as ln
import numpy as np
import random
import time
import os.path

sys.path.insert(0, "../../rl")
from agent import Agent
from DQNAgentDistEnt import DQNAgentDistEnt
from DQNAgentDistEnt_2 import DQNAgentDistEnt_2

# ── Simulation parameters ────────────────────────────────────────────────────
ttime  = 50          # time slots per trial
ttime2 = 50
step   = 50
times  = 10          # independent trials
nodeNo = 50          # nodes (paper: 50)
alpha_ = 0.0002      # default entanglement-generation alpha
degree = 6

# Sweep ranges — one list per X-axis in the paper
numOfRequestPerRound  = [25, 30, 35]                      # Fig 5(a)
totalRequest          = [10, 20, 30, 40, 50]
numOfNodes            = [50, 75, 100]
r                     = [0, 2, 4, 6, 8, 10]
q                     = [0.7, 0.8, 0.9]                   # Fig 5(b)
alpha                 = [0.0001, 0.0002, 0.0003]           # Fig 5(c)  (×10⁻⁴ in paper)
SocialNetworkDensity  = [0.25, 0.5, 0.75, 1]
preSwapFraction       = [0.4, 0.6, 0.8, 1]
entanglementLifetimes = [1, 2, 3, 4, 5, 6, 7, 8]          # Fig 4
requestTimeouts       = [100, 200, 300]
preSwapCapacity       = [0.2, 0.4, 0.5, 0.6, 0.8]

Xlabels = [
    "#RequestPerRound",      # 0  → Fig 5(a)
    "totalRequest",          # 1
    "#nodes",                # 2
    "r",                     # 3
    "swapProbability",       # 4  → Fig 5(b)
    "alpha",                 # 5  → Fig 5(c)
    "SocialNetworkDensity",  # 6
    "preSwapFraction",       # 7
    "entanglementLifetime",  # 8  → Fig 4
    "requestTimeout",        # 9
    "preSwapCapacity",       # 10
]

# Indices of Xlabels to actually run (reproduce all paper figures)
runLabel = [0, 4, 5, 8]

# Algorithms that run for the shorter ttime2 window
toRunLessAlgos = ['REPS']


# ── Per-trial worker ─────────────────────────────────────────────────────────
def runThread(algo, requests, algoIndex, ttime, pid, resultDict, shared_data):
    if '_entdqrl' in algo.name:
        algo.entAgent = DQNAgentDistEnt(algo, pid)
    if '_2entdqrl' in algo.name:
        algo.entAgent = DQNAgentDistEnt_2(algo, pid)

    timeSlot = ttime
    global ttime2
    if algo.name in toRunLessAlgos:
        timeSlot = min(ttime2, ttime)

    for i in range(timeSlot):
        result = algo.work(requests[i], i)

    if 'SEER' in algo.name:
        for req in algo.requestState:
            if algo.requestState[req].state == 2:
                algo.requestState[req].intermediate.clearIntermediate()

    resultDict[pid] = result

    success_req = sum(result.successfulRequestPerRound[i] for i in range(timeSlot))
    max_success_key = (algo.name + str(len(algo.topo.nodes))
                       + str(algo.topo.alpha) + str(algo.topo.q) + 'max_success')

    print('=' * 52)
    print(f'pid: {pid}  success_req: {success_req}')
    print(f'pid: {pid}  max_success_rate: {shared_data[max_success_key] / timeSlot}')
    print('=' * 52)

    if ('_entdqrl' in algo.name or '_2entdqrl' in algo.name) and success_req > shared_data[max_success_key]:
        algo.entAgent.save_model()
        shared_data[max_success_key] = success_req


# ── Single-parameter run ─────────────────────────────────────────────────────
def Run(numOfRequestPerRound=30, numOfNode=0, r=7, q=0.9, alpha=alpha_,
        SocialNetworkDensity=0.5, rtime=ttime, topo=None, FixedRequests=None,
        results=[]):

    if topo is None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6)
    numOfNode = len(topo.nodes)

    topo.setQ(q)
    topo.setAlpha(alpha)

    algorithms = [
        REPS(copy.deepcopy(topo), name='REPS'),
        REPSCACHEENT_DQRL_PSWAP(
            copy.deepcopy(topo), param='ten',
            name='REPSCACHE_DQRL_PSWAP_entdqrl_1hop_distdqrl'),
    ]

    algorithms[0].r = r
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

    bias_weights = [x % 10 == 0 for x in range(numOfNode)]
    prob = np.array(bias_weights) / np.sum(bias_weights)

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

        for algoIndex in range(len(algorithms)):
            algo     = copy.deepcopy(algorithms[algoIndex])
            requests = {i: [] for i in range(ttime_)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst]))

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
            resultDicts[algoIndex].values(), numOfRequestPerRound, algorithms[0].topo)

    return results


# ── Per-sweep thread targets ─────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("start Run and Generate data.txt")
    t1 = time.time()
    targetFilePath = "../../plot/data/"
    temp    = AlgorithmResult()
    Ylabels = temp.Ylabels

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
    jobs = []

    for XlabelIndex in range(len(Xlabels)):
        if XlabelIndex not in runLabel:
            continue

        Xlabel = Xlabels[XlabelIndex]
        Ydata  = []
        jobs   = []
        results = {Xparam: multiprocessing.Manager().list()
                   for Xparam in Xparameters[XlabelIndex]}

        for Xparam in Xparameters[XlabelIndex]:
            if XlabelIndex == 0:
                job = multiprocessing.Process(
                    target=mainThreadReqPerTime,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 2:
                job = multiprocessing.Process(
                    target=mainThreadNumOfNode,
                    args=(Xparam, results[Xparam]))
            elif XlabelIndex == 4:
                job = multiprocessing.Process(
                    target=mainThreadSwapProb,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 5:
                job = multiprocessing.Process(
                    target=mainThreadAlpha,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 7:
                job = multiprocessing.Process(
                    target=mainThreadSwapFrac,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 8:
                job = multiprocessing.Process(
                    target=mainThreadEntanglementLifetime,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 9:
                job = multiprocessing.Process(
                    target=mainThreadRequestTimeout,
                    args=(Xparam, topo, results[Xparam]))
            elif XlabelIndex == 10:
                job = multiprocessing.Process(
                    target=mainThreadPreSwapCapacity,
                    args=(Xparam, topo, results[Xparam]))
            else:
                continue
            jobs.append(job)

        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        for Xparam in Xparameters[XlabelIndex]:
            Ydata.append(results[Xparam])

        # ── Write timeslot success file ──────────────────────────────────
        filename    = "Timeslot_#successRequest.txt"
        sampleRounds = list(range(0, ttime, step))
        F = open(targetFilePath + filename, "w")
        for roundIndex in sampleRounds:
            Xaxis = str(roundIndex)
            Yaxis = []
            for result in Ydata[0]:
                try:
                    Yaxis.append(
                        sum(result.successfulRequestPerRound[roundIndex:roundIndex + step]) / step)
                except Exception:
                    Yaxis.append(0)
            Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
            F.write(Xaxis + Yaxis)
        F.close()

        # ── Write per-metric files ───────────────────────────────────────
        for Ylabel in Ylabels:
            filename = Xlabel + "_" + Ylabel + ".txt"
            mode = "w" if os.path.isfile(targetFilePath + filename) else "a"
            F = open(targetFilePath + filename, mode)
            for i, Xparam in enumerate(Xparameters[XlabelIndex]):
                Xaxis = str(Xparam)
                Yaxis = [algoResult.toDict()[Ylabel] for algoResult in Ydata[i]]
                Yaxis = str(Yaxis).replace("[", " ").replace("]", "\n").replace(",", "")
                F.write(Xaxis + Yaxis)
            F.close()

    t2 = time.time()
    print(f'-----EXIT-----  total time: {(t2 - t1) / 3600:.2f} hours')
