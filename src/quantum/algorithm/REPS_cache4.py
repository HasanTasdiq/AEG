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


######################
##pre entanglement and pre swap of them##
######################

EPS = 1e-6
class REPSCACHE5(AlgorithmBase):
    def __init__(self, topo, param=None, name=''):
        super().__init__(topo , param=param)
        self.name = "REPSCACHE5"
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        # self.param = param

    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')



        
    
    def printResult(self):
        print('number of used links ' , len(self.topo.usedLinks))
        # print('first link ' , list(self.topo.usedLinks)[0].n1.id , list(self.topo.usedLinks)[0].n2.id)
        print('total number of links ' , len(self.topo.links))
        print('diff ' , len(set(self.topo.links).difference(self.topo.usedLinks)))
        print(self.name , 'alpha: ' , self.topo.alpha)

        # self.topo.clearAllEntanglements()
        self.topo.resetEntanglement(self.timeSlot)
        self.result.waitingTime = self.totalWaitingTime / self.totalRequest
        self.result.usedQubits = self.totalUsedQubits / self.totalRequest
        
        # self.result.remainRequestPerRound.append(len(self.requests) / self.totalRequest)
        self.result.remainRequestPerRound.append(len(self.requests))
        
        print(self.name , " total time:", self.result.waitingTime)
        print(self.name ," remain request:", len(self.requests))
        print(self.name ," current Timeslot:", self.timeSlot)



        print(self.name ,' idle time:', self.result.idleTime)
        print(self.name ,' remainRequestPerRound:', self.result.remainRequestPerRound)
        print(self.name ,' avg usedQubits:', self.result.usedQubits)



    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))

        self.srcDstPairs = []
        for request in self.requests:
            src = request[0]
            dst = request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        self.tryPreSwapp()
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.PFT() # compute (self.ti, self.fi)
        # print('[REPS-CACHE] p2 end')
    
    def p4(self):
        if len(self.srcDstPairs) > 0:
            self.EPS()
            self.ELS()
        # print('[REPS-CACHE] p4 end') 
        self.printResult()
        return self.result

    
                     


    def LP1(self):
        # print('[REPS-CACHE] LP1 start')
        # initialize fi(u, v) ans ti

        self.fi_LP = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)

        edgeIndices = []
        notEdge = []
        # for edge in self.topo.edges:
        #     edgeIndices.append((edge[0].id, edge[1].id))

        # print('*** ========+============================== edgeindices len by edge ' , len(edgeIndices))


        edgeIndices = set()
        for link in self.topo.links:
            if (link.n1.id, link.n2.id) not in edgeIndices and (link.n2.id, link.n1.id) not in edgeIndices:
                edgeIndices.add((link.n1.id, link.n2.id))
        # print('========+============================== edgeindices len ' , len(edgeIndices))
        self.topo.edgeIndices = edgeIndices
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))
        # LP

        m = gp.Model('REPS for PFT')
        m.setParam("OutputFlag", 0)
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "f")
        t = m.addVars(numOfSDpairs, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "t")
        x = m.addVars(numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "x")
        m.update()
        
        m.setObjective(gp.quicksum(t[i] for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            m.addConstr(gp.quicksum(f[i, s, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, s] for v in range(numOfNodes)) == t[i])

            d = self.srcDstPairs[i][1].id
            m.addConstr(gp.quicksum(f[i, d, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, d] for v in range(numOfNodes)) == -t[i])

            for u in range(numOfNodes):
                if u not in [s, d]:
                    m.addConstr(gp.quicksum(f[i, u, v] for v in range(numOfNodes)) - gp.quicksum(f[i, v, u] for v in range(numOfNodes)) == 0)

        
        for (u, v) in edgeIndices:
            dis = self.topo.distance(self.topo.nodes[u].loc, self.topo.nodes[v].loc)
            # probability = math.exp(-self.topo.alpha * dis)
            # print('++===+++== orig prob ' , self.timeSlot , probability)
            links = [link for link in self.topo.links if ((link.n1.id == u and link.n2.id == v) or (link.n1.id == v and link.n2.id == u))]
            # print('++++++ ' , len(links))
            
            prob = 0
            isVirtual = False
            for link in links:
                if link.isEntangled(self.timeSlot):
                    prob+=1
                else:
                    prob +=link.p()
                isVirtual = isVirtual or link.isVirtualLink
            probability = prob/len(links)
            # print('++===+++== later prob ' , self.timeSlot , probability)

            m.addConstr(gp.quicksum(f[i, u, v] + f[i, v, u] for i in range(numOfSDpairs)) <= probability * x[u, v])

            capacity = self.edgeCapacity(self.topo.nodes[u], self.topo.nodes[v])
            # if isVirtual:
            #     print('for the v link capacity: ', (u,v) , capacity , len(links), probability)
            m.addConstr(x[u, v] <= capacity)


        for (u, v) in notEdge:
            m.addConstr(x[u, v] == 0)               
            for i in range(numOfSDpairs):
                m.addConstr(f[i, u, v] == 0)

        for u in range(numOfNodes):
            edgeContainu = []
            for (n1, n2) in edgeIndices:
                if u in (n1, n2):
                    edgeContainu.append((n1, n2))
                    edgeContainu.append((n2, n1))
            m.addConstr(gp.quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

        m.optimize()
        if m.status != 2:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++FAILED++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            exit()
        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]
            # for edge in self.topo.edges:
            #     u = edge[0]
            #     v = edge[1]
            for (p , q) in edgeIndices:
                u = self.topo.nodes[p]
                v = self.topo.nodes[q]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            # for edge in self.topo.edges:
            #     u = edge[1]
            #     v = edge[0]
            for (p , q) in edgeIndices:
                v = self.topo.nodes[p]
                u = self.topo.nodes[q]
                varName = self.genNameByComma('f', [i, u.id, v.id])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for (u, v) in notEdge:
                u = self.topo.nodes[u]
                v = self.topo.nodes[v]
                self.fi_LP[SDpair][(u, v)] = 0
            
            
            varName = self.genNameByComma('t', [i])
            self.ti_LP[SDpair] = m.getVarByName(varName).x
        # print('[REPS-CACHE] LP1 end')
    def edgeCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        used = 0
        for SDpair in self.srcDstPairs:
            used += self.fi[SDpair][(u, v)]
            used += self.fi[SDpair][(v, u)]
        if used >= capacity:
            return 0
        return capacity - used

    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    def PFT(self):

        # initialize fi and ti
        self.fi = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti = {SDpair : 0 for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for u in self.topo.nodes:
                for v in self.topo.nodes:
                    self.fi[SDpair][(u, v)] = 0
        
        # PFT
        failedFindPath = False
        while not failedFindPath:
            self.LP1()
            failedFindPath = True
            Pi = {}
            paths = []
            for SDpair in self.srcDstPairs:
                Pi[SDpair] = self.findPathsForPFT(SDpair)

            for SDpair in self.srcDstPairs:
                K = len(Pi[SDpair])
                for k in range(K):
                    width = math.floor(Pi[SDpair][k][-1])
                    Pi[SDpair][k][-1] -= width
                    paths.append(Pi[SDpair][k])
                    pathLen = len(Pi[SDpair][k]) - 1
                    self.ti[SDpair] += width
                    if width == 0:
                        continue
                    failedFindPath = False
                    for nodeIndex in range(pathLen - 1):
                        node = Pi[SDpair][k][nodeIndex]
                        next = Pi[SDpair][k][nodeIndex + 1]
                        self.fi[SDpair][(node, next)] += width

            sorted(paths, key = self.widthForSort)
            # print('[PFT]###===###+==== path len for ' , SDpair[0].id , SDpair[1].id , ':' , len(paths))
            # for path in paths:
            #     print('[PFT]============ ' ,[n for n in path])
            for path in paths:
                pathLen = len(path) - 1
                width = path[-1]
                SDpair = (path[0], path[-2])
                isable = True
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    if self.edgeCapacity(node, next) < 1:
                        isable = False
                
                if not isable:
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                    continue
                
                failedFindPath = False
                self.ti[SDpair] += 1
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    self.fi[SDpair][(node, next)] += 1

        # print('[REPS-CACHE] PFT end')
        for SDpair in self.srcDstPairs:
            # print('!!!!!!!==================  ' , len(self.topo.edgeIndices))
            # for edge in self.topo.edges:
            #     u = edge[0]
            #     v = edge[1]
            for (p , q) in self.topo.edgeIndices:
                u = self.topo.nodes[p]
                v = self.topo.nodes[q]
                need = self.fi[SDpair][(u, v)] + self.fi[SDpair][(v, u)]
                if need:
                    assignCount = 0
                    for link in u.links:
                        if link.contains(v) and link.assignable():
                            # link(u, v) for u, v in edgeIndices)
                            link.assignQubits()
                            self.totalUsedQubits += 2
                            assignCount += 1
                            if assignCount == need:
                                break
            
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            # if link.contains(v):
                # print('link.isEntangled({self.timeSlot}): ',  link.isEntangled(self.timeSlot))
            if link.contains(v) and link.isEntangled(self.timeSlot):
                capacity += 1
        return capacity

    def LP2(self):
        # print('[REPS-CACHE] LP2 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow = [self.ti[self.srcDstPairs[i]] for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxK = max(numOfFlow)
        else:
            maxK = 0
        self.fki_LP = {SDpair : [{} for _ in range(maxK)] for SDpair in self.srcDstPairs}
        self.tki_LP = {SDpair : [0] * maxK for SDpair in self.srcDstPairs}
        
        edgeIndices = []
        notEdge = []
        # for edge in self.topo.edges:
        #     edgeIndices.append((edge[0].id, edge[1].id))

        edgeIndices = set()
        for link in self.topo.links:
            if (link.n1.id, link.n2.id) not in edgeIndices and (link.n2.id, link.n1.id) not in edgeIndices:
                edgeIndices.add((link.n1.id, link.n2.id))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))

        m = gp.Model('REPS for EPS')
        m.setParam("OutputFlag", 0)

        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxK
            for k in range(maxK):
                f[i][k] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][k][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if k < numOfFlow[i] and ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][k][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, k, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                if k < numOfFlow[i]:
                    t[i][k] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))
                else:
                    t[i][k] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))

        m.update()
        
        m.setObjective(gp.quicksum(t[i][k] for k in range(maxK) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            d = self.srcDstPairs[i][1].id
            
            for k in range(numOfFlow[i]):
                neighborOfS = []
                neighborOfD = []

                for edge in edgeIndices:
                    if edge[0] == s:
                        neighborOfS.append(edge[1])
                    elif edge[1] == s:
                        neighborOfS.append(edge[0])
                    if edge[0] == d:
                        neighborOfD.append(edge[1])
                    elif edge[1] == d:
                        neighborOfD.append(edge[0])
                    
                m.addConstr(gp.quicksum(f[i][k][s][v] for v in neighborOfS) - gp.quicksum(f[i][k][v][s] for v in neighborOfS) == t[i][k])
                m.addConstr(gp.quicksum(f[i][k][d][v] for v in neighborOfD) - gp.quicksum(f[i][k][v][d] for v in neighborOfD) == -t[i][k])

                for u in range(numOfNodes):
                    if u not in [s, d]:
                        edgeUV = []
                        for v in range(numOfNodes):
                            if v not in [s, d]:
                                edgeUV.append(v)
                        m.addConstr(gp.quicksum(f[i][k][u][v] for v in edgeUV) - gp.quicksum(f[i][k][v][u] for v in edgeUV) == 0)

        
        for (u, v) in edgeIndices:
            capacity = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(gp.quicksum((f[i][k][u][v] + f[i][k][v][u]) for k in range(maxK) for i in range(numOfSDpairs)) <= capacity)

        m.optimize()

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]

            for k in range(numOfFlow[i]):
                # for edge in self.topo.edges:
                #     u = edge[0]
                #     v = edge[1]
                for (p , q) in edgeIndices:
                    u = self.topo.nodes[p]
                    v = self.topo.nodes[q]
                    varName = self.genNameByBbracket('f', [i, k, u.id, v.id])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                # for edge in self.topo.edges:
                #     u = edge[1]
                #     v = edge[0]
                for (p , q) in edgeIndices:
                    v = self.topo.nodes[p]
                    u = self.topo.nodes[q]
                    varName = self.genNameByBbracket('f', [i, k, u.id, v.id])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                for (u, v) in notEdge:
                    u = self.topo.nodes[u]
                    v = self.topo.nodes[v]
                    self.fki_LP[SDpair][k][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, k])
                self.tki_LP[SDpair][k] = m.getVarByName(varName).x
        # print('[REPS-CACHE] LP2 end')

    def EPS(self):
        self.LP2()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : self.ti[SDpair] for SDpair in self.srcDstPairs}
        self.fki = {SDpair : [{} for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [0 for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.pathForELS = {SDpair : [] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                self.tki[SDpair][k] = self.tki_LP[SDpair][k] >= random.random()

                if not self.tki[SDpair][k]:
                    continue
                paths = self.findPathsForEPS(SDpair, k)


                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0
                    
                for path in paths:
                    width = path[-1]
                    select = (width / self.tki_LP[SDpair][k]) >= random.random()
                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[SDpair].append(path)
                    pathLen = len(path)
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        self.fki[SDpair][k][(node, next)] = 1
                
        # print('[REPS-CACHE] EPS end')

    def ELS(self):
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes}
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        Pi = {SDpair : [] for SDpair in self.srcDstPairs}
        T = [SDpair for SDpair in self.srcDstPairs]

        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
                if len(Ci[SDpair]) == 0 and SDpair in T:
                    T.remove(SDpair)
            
            if len(T) == 0:
                break

            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) < minLength:
                        minLength = len(path)
                        i = SDpair
            
            src = i[0]
            dst = i[1]

            minR = math.inf
            for path in Ci[i]:
                r = 0
                for node in path:
                    r += self.weightOfNode[node]
                if minR > r:
                    targetPath = path
                    minR = r
            
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []

            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 2):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                targetLink1 = None
                targetLink2 = None
                for link in node.links:
                    if link.contains(next) and link.isEntangled(self.timeSlot) and link.notSwapped():
                        targetLink1 = link
                    
                    if link.contains(prev) and link.isEntangled(self.timeSlot) and link.notSwapped():
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1

                nextLink[node].append(targetLink1)
                if targetLink1 is not None and targetLink2 is not None:
                    needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))

            T.remove(i)

        T = [SDpair for SDpair in self.srcDstPairs]
        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
            
            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) - 1 < minLength:
                        minLength = len(path) - 1
                        i = SDpair
                if len(Ci[SDpair]) == 0 and i == -1:
                    i = SDpair
            
            src = i[0]
            dst = i[1]

            targetPath = self.findPathForELS(i)
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []
            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 1):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                targetLink1 = None
                targetLink2 = None
                for link in node.links:
                    if link.contains(next) and link.isEntangled(self.timeSlot):
                        targetLink1 = link
                    
                    if link.contains(prev) and link.isEntangled(self.timeSlot):
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1
                nextLink[node].append(targetLink1)
                if targetLink1 is not None and targetLink2 is not None:
                    needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))
            T.remove(i)
        
        # print('[REPS-CACHE] ELS end')
        # print('[REPS-CACHE]' + [(src.id, dst.id) for (src, dst) in self.srcDstPairs])
        totalEntanglement = 0
        for SDpair in self.srcDstPairs:
            src = SDpair[0]
            dst = SDpair[1]

            # print('##########path ' , src.id , dst.id , len(Pi[SDpair]))

            if len(Pi[SDpair]):
                self.result.idleTime -= 1

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                # print('[REPS-CACHE] attempt:', [node.id for node in path])
                # print('[REPS-CACHE] (node, link1, link2) :', [(x[0].id , x[1].n1.id , x[1].n2.id , x[2].n1.id , x[2].n2.id) for x in needLink[(SDpair, pathIndex)]])
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    swapped = node.attemptSwapping2(link1, link2, timeSlot = self.timeSlot)
                    # swapped = node.attemptSwapping(link1, link2)
                    key = (node , link1.theOtherEndOf(node) , link2.theOtherEndOf(node))
                    if not key in self.topo.needLinksDict:
                        self.topo.needLinksDict[key] = [self.timeSlot]
                    else:
                        self.topo.needLinksDict[key].append( self.timeSlot)

                        

                # successPath = self.topo.getEstablishedEntanglements(src, dst , self.timeSlot)


                successPath = self.topo.getEstablishedEntanglementsWithLinks(src, dst , self.timeSlot)
                usedLinksCount = 0
                for path in successPath:
                    for node , link in path:
                        if link is not None:
                            self.topo.usedLinks.add(link)
                            usedLinksCount += 1
                            # if link.isVirtualLink:
                            #     print('++++========++++++ virtual link found +++++==========+++++')
                # for x in successPath:
                    # print('[REPS-CACHE] success:', [z.id for z in x])
                # print('[REPS-CACHE] success path :', len(successPath))

                if len(successPath):
                    for request in self.requests:
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS-CACHE] finish time:', self.timeSlot - request[2])
                            self.requests.remove(request)
                            break
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    if link1 in self.topo.usedLinks:
                        link1.clearPhase4Swap()
                    else:
                        link1.keepPhase4Swap()
                    if link2 in self.topo.usedLinks:
                        link2.clearPhase4Swap()
                    else:
                        link2.keepPhase4Swap()
                totalEntanglement += len(successPath)
        self.result.entanglementPerRound.append(totalEntanglement)
        
        print(self.name , '######+++++++========= total ent: ' , totalEntanglement , 'at time:' , self.timeSlot)

        # for link in self.topo.links:
        #     if link.isVirtualLink:
        #         self.topo.links.remove(link)
        # for node in self.topo.nodes:
        #     for link in node.links:
        #         if link.isVirtualLink:
        #             node.links.remove(link)


    def findPathsForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForPFT(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForPFT(path, SDpair)
            
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fi_LP[SDpair][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node in self.topo.nodes:
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                adjcentList[node].add(neighbor)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fi_LP[SDpair][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForPFT(self, path, SDpair):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fi_LP[SDpair][(currentNode, nextNode)])

        return width
    
    def findPathsForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPS(SDpair, k):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForEPS(path, SDpair, k)
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][k][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(node1, node2) > 0:
                    adjcentList[node1].add(node2)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fki_LP[SDpair][k][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], next.id))

        return False

    def widthForEPS(self, path, SDpair, k):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fki_LP[SDpair][k][(currentNode, nextNode)])

        return width
    
    def findPathForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForELS(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.y[(node1, node2)] < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((self.weightOfNode[src], src.id))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = distance[u] + self.weightOfNode[next]
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], next.id))

        return False
if __name__ == '__main__':
    
    topo = Topo.generate(50, 0.9, 5, 0.0002, 6)
    s = REPSCACHE5(topo)
    result = AlgorithmResult()
    samplesPerTime = 40
    ttime = 50
    rtime = 50
    requests = {i : [] for i in range(ttime)}

    for i in range(ttime):
        if i < rtime:

            # ids = [(63, 93), (89, 13), (82, 77), (96, 71), (99, 40)]
            # for (p,q) in ids:
            #     source = None
            #     dest = None
            #     for node in topo.nodes:

            #         if node.id == p:
            #             source = node
            #         if node.id == q:
            #             dest = node
            #     requests[i].append((source , dest))

            a = sample(topo.nodes, samplesPerTime)
            for n in range(0,samplesPerTime,2):
                requests[i].append((a[n], a[n+1]))
        print('[REPS-CACHE4] S/D:' , i , [(a[0].id , a[1].id) for a in requests[i]])

    for i in range(ttime):
        result = s.work(requests[i], i)
    

    # print(result.waitingTime, result.numOfTimeslot)