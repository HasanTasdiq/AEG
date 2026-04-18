"""
SP (Shortest Path) — Greedy hop-count routing baseline.

p2(): For each pending request, builds the shortest path by hop count and
      assigns qubits to each link along the way. Repeats until no more paths
      can be built (exhaustive — mirrors AEG_LS / Random behaviour).

p4(): Uses the shared LP2 / EPS / ELS pipeline, identical to ILP, Random,
      and AEG variants, so results are directly comparable.

This is the "SP" baseline in the paper's ablation study.
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


class SP(AlgorithmBase):
    """Shortest-path baseline: greedy hop-count qubit assignment, shared EPS/ELS pipeline."""

    def __init__(self, topo, name='SP'):
        super().__init__(topo)
        self.name             = name
        self.requests         = []
        self.totalRequest     = 0
        self.totalUsedQubits  = 0
        self.totalWaitingTime = 0

    def prepare(self):
        self.requests.clear()
        self.updateNeighbors()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gen_name_bracket(self, var, par):
        return (var + str(par)).replace(' ', '').replace(',', '][')

    def printResult(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest:
            self.result.waitingTime = self.totalWaitingTime / self.totalRequest
            self.result.usedQubits  = self.totalUsedQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
        print(f'[SP] slot {self.timeSlot} | '
              f'successful: {self.result.successfulRequest} | '
              f'remaining: {len(self.requests)}')

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []
        for req in self.requests:
            sd = (req[0], req[1])
            if sd not in self.srcDstPairs:
                self.srcDstPairs.append(sd)

    # ── Phase 2: exhaustive greedy shortest-path qubit assignment ─────────────

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime  += len(self.requests)
        if not self.srcDstPairs:
            return
        self.result.numOfTimeslot += 1
        self._assign_qubits_shortest_paths()

    def _assign_qubits_shortest_paths(self):
        """Greedily assign qubits along min-hop paths, repeating until no more
        paths can be built — mirrors AEG_LS's exhaustive while-loop."""
        while True:
            found = False
            for (src, dst) in self.srcDstPairs:
                # Build shortest path by greedy min-hop
                p = [src]
                while True:
                    last = p[-1]
                    if last == dst:
                        break
                    candidates = []
                    for neighbor in last.neighbors:
                        if (neighbor.remainingQubits > 2
                                or (neighbor == dst and neighbor.remainingQubits > 1)):
                            for link in neighbor.links:
                                if link.contains(last) and not link.assigned:
                                    candidates.append(neighbor)
                                    break
                    next_node = self.topo.sentinel
                    min_hops  = sys.maxsize
                    for nb in candidates:
                        h = self.topo.hopsAway(nb, dst, 'Hop')
                        if h != -1 and h < min_hops:
                            min_hops  = h
                            next_node = nb
                    if next_node == self.topo.sentinel or next_node in p:
                        break
                    p.append(next_node)

                if p[-1] != dst:
                    continue
                width = self.topo.widthPhase2(p)
                if width == 0:
                    continue

                found = True
                for _ in range(width):
                    for s in range(len(p) - 1):
                        n1, n2 = p[s], p[s + 1]
                        for link in n1.links:
                            if link.contains(n2) and not link.assigned:
                                self.totalUsedQubits += 2
                                link.assignQubits()
                                break
            if not found:
                break

    # ── Phase 4: shared EPS / ELS pipeline (identical to ILP / Random / AEG) ─

    def p4(self):
        if self.srcDstPairs:
            self.EPS()
            self.ELS()
        else:
            # No pending requests — keep all three per-slot lists in sync
            self.result.successfulRequestPerRound.append(0)
            self.result.entanglementPerRound.append(0)
        self.printResult()
        return self.result

    # ── edgeSuccessfulEntangle ────────────────────────────────────────────────

    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        return sum(1 for link in u.links if link.contains(v) and link.entangled)

    # ── LP2 ──────────────────────────────────────────────────────────────────

    def LP2(self):
        numOfNodes   = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow    = [9] * numOfSDpairs
        maxK         = max(numOfFlow) if numOfFlow else 0

        self.fki_LP = {sd: [{} for _ in range(maxK)] for sd in self.srcDstPairs}
        self.tki_LP = {sd: [0]  * maxK               for sd in self.srcDstPairs}

        edgeIndices = [(e[0].id, e[1].id) for e in self.topo.edges]
        notEdge = [
            (u, v)
            for u in range(numOfNodes)
            for v in range(numOfNodes)
            if (u, v) not in edgeIndices and (v, u) not in edgeIndices
        ]

        m = gp.Model('SP LP2')
        m.setParam('OutputFlag', 0)

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
                        nbU = [v for v in range(numOfNodes)
                               if (u, v) in edgeIndices or (v, u) in edgeIndices]
                        if nbU:
                            m.addConstr(
                                gp.quicksum(f[i][k][u][v] for v in nbU) -
                                gp.quicksum(f[i][k][v][u] for v in nbU) == 0)

        for (u, v) in edgeIndices:
            cap = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
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
                    self.fki_LP[sd][k][(u, v)] = m.getVarByName(
                        self._gen_name_bracket('f', [i, k, u.id, v.id])).x
                for edge in self.topo.edges:
                    v, u = edge[0], edge[1]
                    self.fki_LP[sd][k][(u, v)] = m.getVarByName(
                        self._gen_name_bracket('f', [i, k, u.id, v.id])).x
                self.tki_LP[sd][k] = m.getVarByName(
                    self._gen_name_bracket('t', [i, k])).x

    # ── EPS ──────────────────────────────────────────────────────────────────

    def EPS(self):
        self.LP2()
        numOfFlow = {sd: 9 for sd in self.srcDstPairs}
        self.fki  = {sd: [{} for _ in range(9)] for sd in self.srcDstPairs}
        self.tki  = {sd: [0]  * 9               for sd in self.srcDstPairs}
        self.pathForELS = {sd: [] for sd in self.srcDstPairs}

        for sd in self.srcDstPairs:
            for k in range(9):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[sd][k][(u, v)] = 0

        for sd in self.srcDstPairs:
            for k in range(numOfFlow[sd]):
                tki_val = self.tki_LP[sd][k]
                self.tki[sd][k] = tki_val >= random.random()
                if not self.tki[sd][k]:
                    continue
                paths = self._find_paths_for_eps(sd, k)
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[sd][k][(u, v)] = 0
                for path in paths:
                    width = path[-1]
                    if tki_val <= 0:
                        continue
                    if (width / tki_val) < random.random():
                        continue
                    path = path[:-1]
                    self.pathForELS[sd].append(path)
                    for idx in range(len(path) - 1):
                        self.fki[sd][k][(path[idx], path[idx + 1])] = 1

    # ── ELS ──────────────────────────────────────────────────────────────────

    def ELS(self):
        Ci            = self.pathForELS
        self.y            = {(u, v): 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node: -ln(node.q) for node in self.topo.nodes}
        needLink  = {}
        nextLink  = {node: [] for node in self.topo.nodes}
        Pi        = {sd: [] for sd in self.srcDstPairs}
        T         = list(self.srcDstPairs)

        # First pass
        while T:
            for sd in self.srcDstPairs:
                to_rm = [p for p in Ci[sd]
                         if any(self.y[(p[j], p[j+1])] >=
                                self.edgeSuccessfulEntangle(p[j], p[j+1])
                                for j in range(len(p) - 1))]
                for p in to_rm:
                    Ci[sd].remove(p)
                if not Ci[sd] and sd in T:
                    T.remove(sd)
            if not T:
                break

            i = min(
                ((sd, path) for sd in T for path in Ci[sd]),
                key=lambda x: len(x[1])
            )[0]
            target    = min(Ci[i], key=lambda p: sum(self.weightOfNode[n] for n in p))
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []
            Pi[i].append(target)

            for idx in range(1, len(target) - 2):
                prev, node, nxt = target[idx-1], target[idx], target[idx+1]
                link1 = next((l for l in node.links
                              if l.contains(nxt)  and l.entangled and l.notSwapped()), None)
                link2 = next((l for l in node.links
                              if l.contains(prev) and l.entangled and l.notSwapped()), None)
                self.y[(node, nxt)]  += 1; self.y[(nxt,  node)] += 1
                self.y[(node, prev)] += 1; self.y[(prev, node)] += 1
                nextLink[node].append(link1)
                if link1 and link2:
                    needLink[(i, pathIndex)].append((node, link1, link2))
            T.remove(i)

        # Second pass
        T = list(self.srcDstPairs)
        while T:
            for sd in self.srcDstPairs:
                to_rm = [p for p in Ci[sd]
                         if any(self.y[(p[j], p[j+1])] >=
                                self.edgeSuccessfulEntangle(p[j], p[j+1])
                                for j in range(len(p) - 1))]
                for p in to_rm:
                    Ci[sd].remove(p)

            i_sd = None; minLen = math.inf
            for sd in T:
                for p in Ci[sd]:
                    if len(p) - 1 < minLen:
                        minLen = len(p) - 1; i_sd = sd
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
                self.y[(node, nxt)]  += 1; self.y[(nxt,  node)] += 1
                self.y[(node, prev)] += 1; self.y[(prev, node)] += 1
                nextLink[node].append(link1)
                if link1 and link2:
                    needLink[(i_sd, pathIndex)].append((node, link1, link2))
            T.remove(i_sd)

        # Execute swaps and count successes
        success_req = 0
        total_ent   = 0
        used_links  = set()
        for sd in self.srcDstPairs:
            src, dst = sd
            if Pi[sd]:
                self.result.idleTime -= 1
            for pathIndex, path in enumerate(Pi[sd]):
                for (node, l1, l2) in needLink[(sd, pathIndex)]:
                    used_links.add(l1); used_links.add(l2)
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
                                self.topo.reward_ent.get(edge, 0) + self.topo.negative_reward)
                        if lnk:
                            lnk.clearPhase4Swap()

        self.result.usedLinks                  += len(used_links)
        self.result.entanglementPerRound.append(total_ent)
        self.result.successfulRequestPerRound.append(success_req)
        self.result.successfulRequest          += success_req
        self._filter_requests()
        print(f'[SP] ELS done — slot {self.timeSlot} | '
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
            path = []; cur = sd[1]
            while cur != self.topo.sentinel:
                path.append(cur); cur = self.parent[cur]
            path = path[::-1]
            width = self._width_eps(path, sd, k)
            for j in range(len(path) - 1):
                self.fki_LP[sd][k][(path[j], path[j+1])] -= width
            path.append(width)
            path_list.append(path[:])
        return path_list

    def _dijkstra_eps(self, sd, k):
        src, dst = sd
        self.parent = {n: self.topo.sentinel for n in self.topo.nodes}
        adj = {n: set() for n in self.topo.nodes}
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(n1, n2) > 0:
                    adj[n1].add(n2)
        dist    = {n: 0     for n in self.topo.nodes}
        visited = {n: False for n in self.topo.nodes}
        pq = PriorityQueue()
        pq.put((-math.inf, src.id))
        while not pq.empty():
            d, uid = pq.get(); u = self.topo.nodes[uid]
            if visited[u]: continue
            if u == dst: return True
            dist[u] = -d; visited[u] = True
            for nxt in adj[u]:
                nd = min(dist[u], self.fki_LP[sd][k][(u, nxt)])
                if dist[nxt] < nd:
                    dist[nxt] = nd; self.parent[nxt] = u; pq.put((-nd, nxt.id))
        return False

    def _width_eps(self, path, sd, k):
        return min(self.fki_LP[sd][k][(path[i], path[i+1])]
                   for i in range(len(path) - 1))

    def _find_path_for_els(self, sd):
        if self._dijkstra_els(sd):
            path = []; cur = sd[1]
            while cur != self.topo.sentinel:
                path.append(cur); cur = self.parent[cur]
            return path[::-1]
        return []

    def _dijkstra_els(self, sd):
        src, dst = sd
        self.parent = {n: self.topo.sentinel for n in self.topo.nodes}
        adj = {n: set() for n in self.topo.nodes}
        for n1 in self.topo.nodes:
            for n2 in self.topo.nodes:
                if self.y[(n1, n2)] < self.edgeSuccessfulEntangle(n1, n2):
                    adj[n1].add(n2)
        dist    = {n: math.inf for n in self.topo.nodes}
        visited = {n: False    for n in self.topo.nodes}
        pq = PriorityQueue()
        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            d, uid = pq.get(); u = self.topo.nodes[uid]
            if visited[u]: continue
            if u == dst: return True
            dist[u] = d; visited[u] = True
            for nxt in adj[u]:
                nd = dist[u] + self.weightOfNode[nxt]
                if dist[nxt] > nd:
                    dist[nxt] = nd; self.parent[nxt] = u; pq.put((nd, nxt.id))
        return False
