"""
Random — Random link selection baseline.

Randomly assigns qubits to physical links (each link selected with 50%
probability). Path selection uses the same LP-based EPS/ELS pipeline as ILP
and AEG, so the only difference from ILP is that link entanglement is chosen
randomly rather than by Integer Linear Programming.

No entanglement caching, no proactive swapping, no RL.

This is the "Random" baseline in the paper's ablation study (Fig. 5 & 6).
"""

import sys
import math
import random
import gurobipy as gp
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from numpy import log as ln
from random import sample
import numpy as np

EPS = 1e-6


class RandomLinkSelection(AlgorithmBase):
    """
    Baseline that randomly assigns qubits to links for entanglement attempts,
    then uses REPS's LP2/EPS/ELS pipeline for path selection.
    """

    def __init__(self, topo, param=None, name='Random'):
        super().__init__(topo, param=param)
        self.name             = name
        self.requests         = []
        self.totalRequest     = 0
        self.totalUsedQubits  = 0
        self.totalWaitingTime = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gen_name_comma(self, var, par):
        return (var + str(par)).replace(' ', '')

    def _gen_name_bracket(self, var, par):
        return (var + str(par)).replace(' ', '').replace(',', '][')

    def _print_result(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest:
            self.result.waitingTime = self.totalWaitingTime / self.totalRequest
            self.result.usedQubits  = self.totalUsedQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
        print(f'[Random] time slot {self.timeSlot} | '
              f'successful: {self.result.successfulRequest} | '
              f'remaining: {len(self.requests)}')

    def _add_new_sd_pairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))

        self.srcDstPairs = []
        for req in self.requests:
            sd = (req[0], req[1])
            if sd not in self.srcDstPairs:
                self.srcDstPairs.append(sd)

    # ── Link selection (random) ───────────────────────────────────────────────

    def _random_assign_links(self):
        """Each assignable link is independently selected with 50% probability."""
        for link in self.topo.links:
            if link.assignable() and random.random() > 0.5:
                link.assignQubits()
                self.totalUsedQubits += 2

    # ── Phase 2: queue requests + random link selection ───────────────────────

    def p2(self):
        self._add_new_sd_pairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime  += len(self.requests)
        if self.srcDstPairs:
            self.result.numOfTimeslot += 1
            self._random_assign_links()

    # ── Phase 4: LP-based path selection + BSM swapping ──────────────────────

    def p4(self):
        if self.srcDstPairs:
            self.EPS()
            self.ELS()   # appends to successfulRequestPerRound AND entanglementPerRound
        else:
            # No pending requests — keep all three per-slot lists in sync
            self.result.successfulRequestPerRound.append(0)
            self.result.entanglementPerRound.append(0)
        self._print_result()   # always appends to remainRequestPerRound
        return self.result

    # ── edgeSuccessfulEntangle ────────────────────────────────────────────────

    def edgeSuccessfulEntangle(self, u, v):
        """Count currently entangled links between nodes u and v."""
        if u == v:
            return 0
        return sum(
            1 for link in u.links
            if link.contains(v) and link.entangled
        )

    # ── LP2 (path selection LP — identical to ILP's second LP) ───────────────

    def LP2(self):
        numOfNodes   = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow    = [9] * numOfSDpairs

        maxK = max(numOfFlow) if numOfFlow else 0
        self.fki_LP = {sd: [{} for _ in range(maxK)] for sd in self.srcDstPairs}
        self.tki_LP = {sd: [0] * maxK               for sd in self.srcDstPairs}

        edgeIndices = []
        notEdge     = []
        for edge in self.topo.edges:
            edgeIndices.append((edge[0].id, edge[1].id))
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))

        m = gp.Model('Random LP2')
        m.setParam('OutputFlag', 0)

        # Initialise f to 0 (not None) — non-edge entries stay 0 so gp.quicksum
        # can safely include them without a TypeError.
        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxK
            for k in range(maxK):
                f[i][k] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][k][u] = [0] * numOfNodes
                    for v in range(numOfNodes):
                        if k < numOfFlow[i] and (
                                (u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][k][u][v] = m.addVar(
                                lb=0, ub=1, vtype=gp.GRB.CONTINUOUS,
                                name='f[%d][%d][%d][%d]' % (i, k, u, v))

        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                ub = 1 if k < numOfFlow[i] else 0
                t[i][k] = m.addVar(lb=0, ub=ub, vtype=gp.GRB.CONTINUOUS,
                                   name='t[%d][%d]' % (i, k))

        m.update()
        m.setObjective(
            gp.quicksum(t[i][k] for k in range(maxK) for i in range(numOfSDpairs)),
            gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s, d = self.srcDstPairs[i][0].id, self.srcDstPairs[i][1].id
            for k in range(numOfFlow[i]):
                nbS = [e[1] for e in edgeIndices if e[0] == s] + \
                      [e[0] for e in edgeIndices if e[1] == s]
                nbD = [e[1] for e in edgeIndices if e[0] == d] + \
                      [e[0] for e in edgeIndices if e[1] == d]
                m.addConstr(gp.quicksum(f[i][k][s][v] for v in nbS) -
                            gp.quicksum(f[i][k][v][s] for v in nbS) == t[i][k])
                m.addConstr(gp.quicksum(f[i][k][d][v] for v in nbD) -
                            gp.quicksum(f[i][k][v][d] for v in nbD) == -t[i][k])
                for u in range(numOfNodes):
                    if u not in [s, d]:
                        # Only sum over actual neighbours — avoids iterating all
                        # N nodes and hitting 0-int entries in non-edge slots.
                        edgeUV = [v for v in range(numOfNodes)
                                  if (u, v) in edgeIndices or (v, u) in edgeIndices]
                        if edgeUV:
                            m.addConstr(
                                gp.quicksum(f[i][k][u][v] for v in edgeUV) -
                                gp.quicksum(f[i][k][v][u] for v in edgeUV) == 0)

        for (u, v) in edgeIndices:
            cap = self.edgeSuccessfulEntangle(
                self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(
                gp.quicksum(
                    (f[i][k][u][v] + f[i][k][v][u])
                    for k in range(maxK) for i in range(numOfSDpairs)
                ) <= cap)

        m.optimize()

        for i in range(numOfSDpairs):
            sd = self.srcDstPairs[i]
            for k in range(numOfFlow[i]):
                for edge in self.topo.edges:
                    u, v = edge[0], edge[1]
                    name = self._gen_name_bracket('f', [i, k, u.id, v.id])
                    self.fki_LP[sd][k][(u, v)] = m.getVarByName(name).x
                for edge in self.topo.edges:
                    v, u = edge[0], edge[1]
                    name = self._gen_name_bracket('f', [i, k, u.id, v.id])
                    self.fki_LP[sd][k][(u, v)] = m.getVarByName(name).x
                name = self._gen_name_bracket('t', [i, k])
                self.tki_LP[sd][k] = m.getVarByName(name).x

    # ── EPS / ELS (identical to ILP — shared path-selection logic) ────────────

    def EPS(self):
        self.LP2()
        numOfFlow = {sd: 9 for sd in self.srcDstPairs}
        self.fki  = {sd: [{} for _ in range(9)] for sd in self.srcDstPairs}
        self.tki  = {sd: [0]  * 9               for sd in self.srcDstPairs}
        self.pathForELS = {sd: [] for sd in self.srcDstPairs}

        for sd in self.srcDstPairs:
            for u in self.topo.nodes:
                for v in self.topo.nodes:
                    for k in range(9):
                        self.fki[sd][k][(u, v)] = 0

        for sd in self.srcDstPairs:
            for k in range(numOfFlow[sd]):
                tki_val = self.tki_LP[sd][k]
                # Bernoulli: commit to this flow instance with prob = tki_val
                self.tki[sd][k] = tki_val >= random.random()
                if not self.tki[sd][k]:
                    continue
                paths = self._find_paths_for_eps(sd, k)
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[sd][k][(u, v)] = 0
                for path in paths:
                    width = path[-1]
                    # Guard: if tki_val is 0 no path should have been found,
                    # but skip division if it somehow is zero to avoid ZeroDivisionError
                    if tki_val <= 0:
                        continue
                    select = (width / tki_val) >= random.random()
                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[sd].append(path)
                    for idx in range(len(path) - 1):
                        self.fki[sd][k][(path[idx], path[idx + 1])] = 1

    def ELS(self):
        Ci   = self.pathForELS
        self.y            = {(u, v): 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node: -ln(node.q) for node in self.topo.nodes}
        needLink  = {}
        nextLink  = {node: [] for node in self.topo.nodes}
        Pi        = {sd: [] for sd in self.srcDstPairs}
        T         = list(self.srcDstPairs)

        # First pass
        while T:
            for sd in self.srcDstPairs:
                to_remove = [p for p in Ci[sd]
                             if any(self.y[(p[j], p[j+1])] >=
                                    self.edgeSuccessfulEntangle(p[j], p[j+1])
                                    for j in range(len(p) - 1))]
                for p in to_remove:
                    Ci[sd].remove(p)
                if not Ci[sd] and sd in T:
                    T.remove(sd)

            if not T:
                break

            i = min(
                ((sd, path) for sd in T for path in Ci[sd]),
                key=lambda x: len(x[1])
            )[0]

            target = min(Ci[i], key=lambda p: sum(self.weightOfNode[n] for n in p))
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []
            Pi[i].append(target)

            for idx in range(1, len(target) - 2):
                prev, node, nxt = target[idx-1], target[idx], target[idx+1]
                link1 = next((l for l in node.links
                              if l.contains(nxt)  and l.entangled and l.notSwapped()), None)
                link2 = next((l for l in node.links
                              if l.contains(prev) and l.entangled and l.notSwapped()), None)
                self.y[(node, nxt)]  += 1
                self.y[(nxt,  node)] += 1
                self.y[(node, prev)] += 1
                self.y[(prev, node)] += 1
                nextLink[node].append(link1)
                if link1 and link2:
                    needLink[(i, pathIndex)].append((node, link1, link2))
            T.remove(i)

        # Second pass
        T = list(self.srcDstPairs)
        while T:
            for sd in self.srcDstPairs:
                to_remove = [p for p in Ci[sd]
                             if any(self.y[(p[j], p[j+1])] >=
                                    self.edgeSuccessfulEntangle(p[j], p[j+1])
                                    for j in range(len(p) - 1))]
                for p in to_remove:
                    Ci[sd].remove(p)

            i_sd   = None
            minLen = math.inf
            for sd in T:
                for p in Ci[sd]:
                    if len(p) - 1 < minLen:
                        minLen = len(p) - 1
                        i_sd   = sd
                if not Ci[sd] and i_sd is None:
                    i_sd = sd

            target    = self._find_path_for_els(i_sd)
            pathIndex = len(Pi[i_sd])
            needLink[(i_sd, pathIndex)] = []
            Pi[i_sd].append(target)

            for idx in range(1, len(target) - 1):
                prev, node, nxt = target[idx-1], target[idx], target[idx+1]
                link1 = next((l for l in node.links
                              if l.contains(nxt)  and l.entangled), None)
                link2 = next((l for l in node.links
                              if l.contains(prev) and l.entangled), None)
                self.y[(node, nxt)]  += 1
                self.y[(nxt,  node)] += 1
                self.y[(node, prev)] += 1
                self.y[(prev, node)] += 1
                nextLink[node].append(link1)
                if link1 and link2:
                    needLink[(i_sd, pathIndex)].append((node, link1, link2))
            T.remove(i_sd)

        # Execute swaps and count successes
        success_req = 0
        total_ent   = 0   # total established end-to-end entanglements this slot
        used_links  = set()
        for sd in self.srcDstPairs:
            src, dst = sd
            if Pi[sd]:
                self.result.idleTime -= 1
            for pathIndex, path in enumerate(Pi[sd]):
                for (node, l1, l2) in needLink[(sd, pathIndex)]:
                    used_links.add(l1)
                    used_links.add(l2)
                    node.attemptSwapping(l1, l2)

                success_paths = self.topo.getEstablishedEntanglementsWithLinks(src, dst)
                total_ent    += len(success_paths)
                for sp in success_paths:
                    for node, link in sp:
                        if link:
                            link.used = True
                            edge = self.topo.linktoEdgeSorted(link)
                            self.topo.reward_ent[edge] = (
                                self.topo.reward_ent.get(edge, 0) + self.topo.positive_reward)

                if success_paths:
                    for req in self.requests:
                        if (src, dst) == (req[0], req[1]):
                            self.requests.remove(req)
                            success_req += 1
                            break

                for (node, l1, l2) in needLink[(sd, pathIndex)]:
                    for lnk in (l1, l2):
                        if lnk and not lnk.used and lnk.entangled:
                            edge = self.topo.linktoEdgeSorted(lnk)
                            self.topo.reward_ent[edge] = (
                                self.topo.reward_ent.get(edge, 0)
                                + self.topo.negative_reward)
                        if lnk:
                            lnk.clearPhase4Swap()

        self.result.usedLinks                  += len(used_links)
        self.result.entanglementPerRound.append(total_ent)
        self.result.successfulRequestPerRound.append(success_req)
        self.result.successfulRequest          += success_req
        self._filter_requests()
        print(f'[Random] ELS done — slot {self.timeSlot} | '
              f'entanglements={total_ent} | successful={success_req}')

    def _filter_requests(self):
        self.requests = [
            r for r in self.requests
            if self.timeSlot - r[2] < self.topo.requestTimeout - 1
        ]

    # ── Dijkstra helpers (EPS / ELS) ──────────────────────────────────────────

    def _find_paths_for_eps(self, sd, k):
        path_list = []
        while self._dijkstra_eps(sd, k):
            path = []
            cur  = sd[1]
            while cur != self.topo.sentinel:
                path.append(cur)
                cur = self.parent[cur]
            path = path[::-1]
            width = self._width_eps(path, sd, k)
            for j in range(len(path) - 1):
                self.fki_LP[sd][k][(path[j], path[j+1])] -= width
            path.append(width)
            path_list.append(path[:])
        return path_list

    def _dijkstra_eps(self, sd, k):
        src, dst = sd
        self.parent   = {n: self.topo.sentinel for n in self.topo.nodes}
        adj           = {n: set() for n in self.topo.nodes}
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(n1, n2) > 0:
                    adj[n1].add(n2)
        dist    = {n: 0      for n in self.topo.nodes}
        visited = {n: False  for n in self.topo.nodes}
        pq      = PriorityQueue()
        pq.put((-math.inf, src.id))
        while not pq.empty():
            d, uid = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue
            if u == dst:
                return True
            dist[u]    = -d
            visited[u] = True
            for nxt in adj[u]:
                nd = min(dist[u], self.fki_LP[sd][k][(u, nxt)])
                if dist[nxt] < nd:
                    dist[nxt]    = nd
                    self.parent[nxt] = u
                    pq.put((-nd, nxt.id))
        return False

    def _width_eps(self, path, sd, k):
        return min(self.fki_LP[sd][k][(path[i], path[i+1])]
                   for i in range(len(path) - 1))

    def _find_path_for_els(self, sd):
        if self._dijkstra_els(sd):
            path, cur = [], sd[1]
            while cur != self.topo.sentinel:
                path.append(cur)
                cur = self.parent[cur]
            return path[::-1]
        return []

    def _dijkstra_els(self, sd):
        src, dst = sd
        self.parent   = {n: self.topo.sentinel for n in self.topo.nodes}
        adj           = {n: set() for n in self.topo.nodes}
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if self.y[(n1, n2)] < self.edgeSuccessfulEntangle(n1, n2):
                    adj[n1].add(n2)
        dist    = {n: math.inf for n in self.topo.nodes}
        visited = {n: False    for n in self.topo.nodes}
        pq      = PriorityQueue()
        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            d, uid = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue
            if u == dst:
                return True
            dist[u]    = d
            visited[u] = True
            for nxt in adj[u]:
                nd = dist[u] + self.weightOfNode[nxt]
                if dist[nxt] > nd:
                    dist[nxt]    = nd
                    self.parent[nxt] = u
                    pq.put((nd, nxt.id))
        return False
