"""
SP (Shortest Path) — Greedy hop-count routing baseline.

For each pending request, greedily builds the shortest path by hop count,
assigning qubits to each link along the way. Path selection and entanglement
are handled per-path without any ILP or RL. No entanglement caching — all
entanglements are cleared at the end of each time slot.

This is the "SP" baseline in the paper's ablation study (Fig. 5).
"""

import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from random import sample


class SP(AlgorithmBase):
    """Shortest-path baseline: greedy hop-count routing, no ILP, no RL."""

    def __init__(self, topo, name='SP'):
        super().__init__(topo)
        self.name                   = name
        self.pathsSortedDynamically = []
        self.requests               = []
        self.totalTime              = 0
        self.totalUsedQubits        = 0
        self.totalNumOfReq          = 0

    def prepare(self):
        self.totalTime = 0
        self.requests.clear()
        self.updateNeighbors()   # ensure neighbor lists are fresh after deepcopy

    def p2(self):
        self.pathsSortedDynamically.clear()

        for (src, dst) in self.srcDstPairs:
            self.totalNumOfReq += 1
            self.requests.append((src, dst, self.timeSlot))

        if self.requests:
            self.result.numOfTimeslot += 1

        while True:
            found = False

            for (src, dst, req_time) in self.requests:
                # Build shortest path by greedy min-hop
                p = [src]
                while True:
                    last = p[-1]
                    if last == dst:
                        break

                    selected_neighbors = []
                    for neighbor in last.neighbors:
                        if (neighbor.remainingQubits > 2
                                or (neighbor == dst and neighbor.remainingQubits > 1)):
                            for link in neighbor.links:
                                if link.contains(last) and not link.assigned:
                                    selected_neighbors.append(neighbor)
                                    break

                    # Pick neighbor with fewest hops to dst
                    next_node   = self.topo.sentinel
                    min_hops    = sys.maxsize
                    for nb in selected_neighbors:
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
                self.pathsSortedDynamically.append((0.0, width, p, req_time))
                self.pathsSortedDynamically.sort(key=lambda x: x[1])

                # Assign qubits along path
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

        for req in self.requests:
            has_path = any(
                (path[0], path[-1], t) == req
                for _, _, path, t in self.pathsSortedDynamically
            )
            if not has_path:
                self.result.idleTime += 1

        print(f'[SP] p2 end — time slot {self.timeSlot}')

    def p4(self):
        success_req = 0
        for _, width, p, req_time in self.pathsSortedDynamically:
            src, dst = p[0], p[-1]
            old_count = len(self.topo.getEstablishedEntanglements(src, dst))

            for i in range(1, len(p) - 1):
                prev, curr, nxt = p[i - 1], p[i], p[i + 1]

                prev_links, next_links = [], []
                remaining = width
                for link in curr.links:
                    if (link.entangled
                            and (link.n1 == prev and not link.s2
                                 or link.n2 == prev and not link.s1)
                            and remaining > 0):
                        prev_links.append(link)
                        remaining -= 1

                remaining = width
                for link in curr.links:
                    if (link.entangled
                            and (link.n1 == nxt and not link.s2
                                 or link.n2 == nxt and not link.s1)
                            and remaining > 0):
                        next_links.append(link)
                        remaining -= 1

                for l1, l2 in zip(prev_links, next_links):
                    curr.attemptSwapping(l1, l2)

            succ = len(self.topo.getEstablishedEntanglements(src, dst)) - old_count
            if succ > 0 or len(p) == 2:
                key = (src, dst, req_time)
                if key in self.requests:
                    self.totalTime += self.timeSlot - req_time
                    self.requests.remove(key)
                    success_req += 1

        self.result.successfulRequestPerRound.append(success_req)
        self.result.successfulRequest        += success_req

        remain_time = sum(self.timeSlot - t for _, _, t in self.requests)
        self.topo.clearAllEntanglements()

        # Always append so list length matches successfulRequestPerRound in every slot
        self.result.remainRequestPerRound.append(
            len(self.requests) / self.totalNumOfReq if self.totalNumOfReq > 0 else 0)

        if self.totalNumOfReq > 0:
            self.result.waitingTime = (
                (self.totalTime + remain_time) / self.totalNumOfReq + 1)
            self.result.usedQubits = self.totalUsedQubits / self.totalNumOfReq

        print(f'[SP] p4 end — slot {self.timeSlot} | '
              f'success={success_req} | remaining={len(self.requests)}')
        return self.result
