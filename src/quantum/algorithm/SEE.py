import sys
import math
import random
import gurobipy as gp
from queue import PriorityQueue , Queue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from numpy import log as ln
from random import sample
import networkx as nx
import time


EPS = 1e-6
class SEE(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "SEE"
        self.requests = []
        self.totalRequest = 0
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        self.numOfFlowPerRequest = 10

    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')
    
    def printResult(self):
        self.topo.clearAllEntanglements()
        self.result.waitingTime = self.totalWaitingTime / self.totalRequest
        self.result.usedQubits = self.totalUsedQubits / self.totalRequest
        
        # self.result.remainRequestPerRound.append(len(self.requests) / self.totalRequest)
        self.result.remainRequestPerRound.append(len(self.requests))
        
        print("[SEE] total time:", self.result.waitingTime)
        print("[SEE] remain request:", len(self.requests))
        print("[SEE] current Timeslot:", self.timeSlot)



        print('[SEE] idle time:', self.result.idleTime)
        print('[SEE] remainRequestPerRound:', self.result.remainRequestPerRound)
        print('[SEE] avg usedQubits:', self.result.usedQubits)

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
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.EPI()
            self.ESC()

        # print('[REPS] p2 end')
    
    def p4(self):
        if len(self.srcDstPairs) > 0:
            self.ECE()
        # print('[REPS] p4 end') 
        self.printResult()
        return self.result




    
    
    

    
   
    def edgeCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        used = 0
        for SDpair in self.srcDstPairs:
            used += self.fi[SDpair][(u, v)]
            used += self.fi[SDpair][(v, u)]
        return capacity - used
    def edgeFullCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        return capacity
    def segmentCapacity(self, u, v):
        min_capacity = 1000
        for path , l in self.topo.k_shortest_paths(u , v , 5):
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i+1]
                capacity = 0
                for link in self.topo.nodes[p1].links:
                    if link.contains(self.topo.nodes[p2]):
                        capacity += 1
                if capacity < min_capacity:
                    min_capacity = capacity
        return min_capacity


    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    
            
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled:
                capacity += 1
        return capacity

    def entanglementProb(self, u , v , k):
        paths_with_len = self.topo.k_shortest_paths(u , v , 5)
        path , len = paths_with_len[k]
        # print('math.exp(-self.topo.alpha * len) ' , len , math.exp(-self.topo.alpha * len))

        return math.exp(-self.topo.alpha * len)

    def LP1(self):
        # print('[REPS] LP1 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        # numOfFlow = [self.ti[self.srcDstPairs[i]] for i in range(numOfSDpairs)]
        numOfFlow = [self.numOfFlowPerRequest for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxN = max(numOfFlow)
        else:
            maxN = 0
        # maxK = 2
        print('maxN ' , maxN)
        self.fki_LP = {SDpair : [{} for _ in range(maxN)] for SDpair in self.srcDstPairs}
        self.tki_LP = {SDpair : [0] * maxN for SDpair in self.srcDstPairs}
        
        edgeIndices = []
        notEdge = []
        self.segments = []
        K_u_v = {}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                K_u_v[(node1 , node2)] = 0
                


        for edge in self.topo.segments:
            edgeIndices.append((edge.n1.id, edge.n2.id))
            self.segments.append((edge.n1 , edge.n2))
            # print('self.topo.distance_by_node(node1.id , node2.id) ' , self.topo.distance_by_node(edge[0].id , edge[1].id))

        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))

        # for node1 in self.topo.nodes:
        #     for node2 in self.topo.nodes:
        #         if node1 == node2:
        #             notEdge.append((node1.id , node2.id))
        #         elif  self.topo.distance_by_node(node1.id , node2.id) <=self.topo.optimal_distance and ((node1.id , node2.id) not in edgeIndices) and ((node2.id , node1.id) not in edgeIndices):
        #             edgeIndices.append((node1.id , node2.id))
        #             self.segments.append((node1 , node2))
        #         else:
        #             notEdge.append((node1.id , node2.id))
        self.x_LP = {(u.id,v.id,k):0 for (u,v) in self.segments for k in range(len(self.topo.k_shortest_paths(u.id,v.id)))}

        print('len(edgeIndices)' , len(edgeIndices))
        print('len(self.topo.edges)' , len(self.topo.edges))
        print('len(self.segments)' , len(self.segments))



        m = gp.Model('SEE for EPI')
        m.setParam("OutputFlag", 0)

        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxN
            for n in range(maxN):
                f[i][n] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][n][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if n < numOfFlow[i] and ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][n][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, n, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxN
            for n in range(maxN):
                if n < numOfFlow[i]:
                    t[i][n] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, n))
                else:
                    t[i][n] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, n))
        x = [0] * numOfNodes
        for u in range(numOfNodes):
            x[u] = [0] * numOfNodes
            for v in range(numOfNodes):
                # if ((u, v) in edgeIndices or (v, u) in edgeIndices):
                    x[u][v] = [0] * len(self.topo.k_shortest_paths(u , v , 5))
                    for k in range(len(self.topo.k_shortest_paths(u , v , 5))): #later we'll find this
                        if ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            # x[u][v][k] = m.addVar(lb = 0 , vtype = gp.GRB.CONTINUOUS , name = "x[%d][%d][%d]"%(u,v,k))
                            x[u][v][k] = m.addVar(lb = 0 , vtype = gp.GRB.INTEGER , name = "x[%d][%d][%d]"%(u,v,k))

        m.update()
        
        m.setObjective(gp.quicksum(t[i][n] for n in range(maxN) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        for i in range(numOfSDpairs):
            s = self.srcDstPairs[i][0].id
            d = self.srcDstPairs[i][1].id
            
            for n in range(numOfFlow[i]):
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

                # neighborOfS = set()
                # neighborOfD = set()

                # for edge in edgeIndices:
                #     if edge[0] == s:
                #         neighborOfS.add(edge[1])
                #     elif edge[1] == s:
                #         neighborOfS.add(edge[0])
                #     if edge[0] == d:
                #         neighborOfD.add(edge[1])
                #     elif edge[1] == d:
                #         neighborOfD.add(edge[0])


                # --------1a----------------
                m.addConstr(gp.quicksum(f[i][n][s][v] for v in neighborOfS) - gp.quicksum(f[i][n][v][s] for v in neighborOfS) == t[i][n])
                # --------1b---------------------
                m.addConstr(gp.quicksum(f[i][n][d][v] for v in neighborOfD) - gp.quicksum(f[i][n][v][d] for v in neighborOfD) == -t[i][n])


                # --------1c------------------
                for u in range(numOfNodes):
                    if u not in [s,d]:    
                        # neighbourOfU = set([segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments])
                        # m.addConstr(gp.quicksum(f[i][n][u][v] for v in neighbourOfU) - gp.quicksum(f[i][n][v][u] for v in neighbourOfU) == 0)


                        m.addConstr(gp.quicksum(f[i][n][u][v] for v in [segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments]) - gp.quicksum(f[i][n][v][u] for v in [segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments]) == 0)


        
        # ---------------------1d-----------------
        # for (u,v) in edgeIndices: 
        #     m.addConstr(gp.quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= gp.quicksum((self.entanglementProb(u , v , k) * (x[u][v][k] + x[v][u][k])* math.sqrt(self.topo.nodes[u].q*self.topo.nodes[v].q)) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                m.addConstr(gp.quicksum((f[i][n][u][v] + f[i][n][v][u]) for n in range(maxN) for i in range(numOfSDpairs)) <= gp.quicksum((self.entanglementProb(u , v , k) * x[u][v][k] * math.sqrt(self.topo.nodes[u].q*self.topo.nodes[v].q)) for k in range(len(self.topo.k_shortest_paths(u , v , 5)))))
        
        
        # ----------------1e-----------------
        for edge in self.topo.edges:
            e1 = edge[0].id
            e2 = edge[1].id
            segContainingEdge = set()
            segContainingEdgeList = [] 
            for (u,v) in edgeIndices:
                paths_with_len = self.topo.k_shortest_paths(u , v , 5)
                j=0
                for path, l in paths_with_len:
                    for i in range(len(path) - 1):
                        p1 = path[i]
                        p2 = path[i+1]
                        if (e1,e2) == (p1 , p2) or (e1,e2) == (p2,p1):
                            # print('in 2nd constr ' , (e1,e2), (p1 , p2) )
                            segContainingEdge.add((u,v , j))
                            segContainingEdgeList.append((u,v , j))
                           
                        
                    j+=1
            capacity = self.edgeFullCapacity(edge[0] , edge[1])
            # print('capacity ' , capacity , 'len(segContainingEdge)' , len(segContainingEdge))
            m.addConstr(gp.quicksum(x[n1][n2][k] for (n1,n2 ,k) in segContainingEdge) <= capacity)






        #----------- 1f------------
        for u in range(numOfNodes):
            neighbourOfU = set([segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments])
            # print('len(neighbourOfU)111::::' , len(neighbourOfU) , 'self.topo.nodes[u].remainingQubits' , self.topo.nodes[u].remainingQubits)
            # neighbourOfU = set()
            # for (n1,n2) in edgeIndices:
            #     if n1 == u:
            #         neighbourOfU.add(n2)
            #     if n2 == u:
            #         neighbourOfU.add(n1)
            # print('len(neighbourOfU)222::::' , len(neighbourOfU) , 'self.topo.nodes[u].remainingQubits' , self.topo.nodes[u].remainingQubits)

            # m.addConstr(gp.quicksum(x[u][v][k] + x[v][u][k] for v in neighbourOfU for k in range(len(self.topo.k_shortest_paths(u , v , 5))) ) <= self.topo.nodes[u].remainingQubits )
            # print('list:: ' , [segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments])
            # print('set:: ' , set([segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments]))
            # print('-------------------------------')
            m.addConstr(gp.quicksum(x[u][v][k] for v in [segment.theOtherEndOf(self.topo.nodes[u]).id for segment in self.topo.nodes[u].segments] for k in range(len(self.topo.k_shortest_paths(u , v , 5))) ) <= self.topo.nodes[u].remainingQubits )

        # time.sleep(20)

        # --------------------------------------
        # for i in range(numOfSDpairs):
        #     for n in range(numOfFlow[i] - 1):
        #         m.addConstr(t[i][n] >= t[i][n+1])
        # -------------------------------------

        print('optimize start....')
        m.optimize()
        if m.status != 2:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++FAILED++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            exit()
        print('model.status' , m.status)

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]

            for n in range(numOfFlow[i]):
                for (u,v) in edgeIndices:
                    varName = self.genNameByBbracket('f', [i, n, u, v])
                    self.fki_LP[SDpair][n][(u, v)] = m.getVarByName(varName).x
                    # if not self.fki_LP[SDpair][n][(u, v)] == 0:
                    #     print('# ',SDpair[0].id ,SDpair[1].id, '  ' , u.id , v.id , '  '  , self.fki_LP[SDpair][n][(u, v)])

                for (v,u) in edgeIndices:

                    varName = self.genNameByBbracket('f', [i, n, u, v])
                    self.fki_LP[SDpair][n][(u, v)] = m.getVarByName(varName).x
                    # if not self.fki_LP[SDpair][n][(u, v)] == 0:
                    #     print('# ',SDpair[0].id ,SDpair[1].id, '  ' , u.id , v.id , '  '  , self.fki_LP[SDpair][n][(u, v)])
                

                # for (u, v) in notEdge:
                #     u = self.topo.nodes[u]
                #     v = self.topo.nodes[v]
                #     self.fki_LP[SDpair][n][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, n])
                self.tki_LP[SDpair][n] = m.getVarByName(varName).x
                print('* ' , self.tki_LP[SDpair][n])
        

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]
            for (u,v) in edgeIndices:
                sum = 0

                for n in range(numOfFlow[i]):
                    sum += self.fki_LP[SDpair][n][(u, v)]
                # print('########################################self.fki_LP[%d][n][(%d, %d)]'%(i , u.id , v.id) , sum)

        time.sleep(5)
        
        for (u,v) in edgeIndices:
            # u = segment[0].id    
            # v = segment[1].id 
            for k in range(len(self.topo.k_shortest_paths(u, v))):
                    varName = self.genNameByBbracket('x' , [u,v,k])
                    self.x_LP[(u,v,k)] = m.getVarByName(varName).x

                    varName = self.genNameByBbracket('x' , [v,u,k])
                    self.x_LP[(v,u,k)] = m.getVarByName(varName).x
                    # if self.x_LP[(v,u,k)] > 0 :
                    #     print('self.x_LP[(v,u,k)]' , ((v,u,k)) , self.x_LP[(v,u,k)] )
                    # if self.x_LP[(u,v,k)] > 0:
                    #     print('self.x_LP[(u,v,k)]' , ((u,v,k)) , self.x_LP[(u,v,k)] )
        
        time.sleep(5)
        print('[SEE] LP1 end')

    def EPI(self):
        self.LP1()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : self.numOfFlowPerRequest for SDpair in self.srcDstPairs}
        self.fki = {SDpair : [{} for n in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [0 for n in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.pathForELS = {SDpair : [] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for n in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][n][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for n in range(numOfFlow[SDpair]):
                self.tki[SDpair][n] = self.tki_LP[SDpair][n] >= random.random()
                # print('for SD ' , SDpair[0].id , SDpair[1].id , self.tki_LP[SDpair][n])

                if not self.tki[SDpair][n]:
                    continue
                paths = self.findPathsForEPI(SDpair, n)

                print('for SD ' , SDpair[0].id , SDpair[1].id  , len(paths))

                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][n][(u, v)] = 0
                    
                for path in paths:

                    width = path[-1]
                    select = (width / self.tki_LP[SDpair][n]) >= random.random()
                    print('path ', select , [p.id for p in path[0:-1]] , width)

                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[SDpair].append(path)

                    pathLen = len(path)
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        self.fki[SDpair][n][(node, next)] = 1
            print('number of path for ESC ' , len(self.pathForELS[SDpair]))
                
        print('[REPS] EPI end')

    def ESC(self):
        T=[]
        for SDPair in self.srcDstPairs:
            paths = self.pathForELS[SDPair]
            T.extend(paths)
            # for path in paths:
            #     print('+++++++++ ===== path for ' , SDPair[0].id , SDPair[1].id , [p.id for p in path])
        T = sorted(T, key=lambda x: len(x))
        # print('T ' , T)
        self.x = {(u, v , k) : 0 for u in self.topo.nodes for v in self.topo.nodes for k in range(len(self.topo.k_shortest_paths(u.id , v.id , 5)))}
        self.D = []
        
        for path in T:
            self.D.append(path)
            availableSegments = []
            availableResource = True

            for i in range(len(path) -1):
                n1 = path[i]
                n2 = path[i+1]
                # print('len(n1 seg)',len(n1.segments))
                # print('len(n2 seg)',len(n2.segments))
                availableSegmentsn1n2 = []
                assignedSeg = [0] * len(self.topo.k_shortest_paths(n1.id , n2.id))
                for k in range(len(self.topo.k_shortest_paths(n1.id , n2.id))):
                    assignedSeg[k] = 0
                for seg in n1.segments:
                    
                    if seg.contains(n2):
                        if seg.assignable():
                            print('assignedSeg[seg.k]', assignedSeg[seg.k], 'self.x_LP[(%d , %d , %d)]'%(n1.id , n2.id , seg.k) , self.x_LP[(n1.id , n2.id , seg.k)] , self.x_LP[(n2.id , n1.id , seg.k)])
                            if assignedSeg[seg.k] < self.x_LP[(n1.id , n2.id , seg.k)] or assignedSeg[seg.k] < self.x_LP[(n2.id , n1.id , seg.k)]:
                                availableSegmentsn1n2.append(seg)
                                assignedSeg[seg.k] += 1
                if not len(availableSegmentsn1n2):
                    # print('not available ' , n1.id , n2.id)
                    availableResource = False
                    break
                else:
                    availableSegments.extend(availableSegmentsn1n2)
                    # print('available ' , n1.id , n2.id)

            print('availableResource ' , availableResource)
            if not availableResource:
                self.D.remove(path)
            else:
                for seg in availableSegments:
                    if self.getAssignedResources(seg.n1 , seg.n2) <= self.maxAssignment(seg.n1.id , seg.n2.id):
                        seg.assignQubits()
                        print('assigning to ' , seg.n1.id , seg.n2.id , seg.k)
                        self.x[seg.n1 , seg.n2 , seg.k] +=1
            


        print('=========len of D ' , len(self.D))


    def maxAssignment(self , uid , vid):
        assignment = 0
        for k in range(len(self.topo.k_shortest_paths(uid , vid ))):
            assignment += self.entanglementProb(uid , vid , k) * self.x_LP[(uid , vid , k)]
        return assignment
    
    def getAssignedResources(self , u , v):
        assigned = 0
        for path in self.D:
            for i in range(len(path) - 1):
                n1 = path[i]
                n2 = path[i+1]
                if (u , v) == (n1,n2):
                    for segment in  u.segments:
                        if segment.contains(v):
                            if segment.assigned:
                                assigned += 1
        return assigned


                    
    def ECE(self):
        if len(self.D) <=0:
            return
        e = {}
        for segment in self.topo.segments:
            e[(segment.n1,segment.n2)] = 0
            e[(segment.n2,segment.n1)] = 0

        Pi = {SDpair : [] for SDpair in self.srcDstPairs}
        needLink = {}
        selected = {node: [] for node in self.topo.nodes}

        for segment in  self.topo.segments:
            if segment.entangled:
                e[(segment.n1,segment.n2)] += 1
                e[(segment.n2,segment.n1)] += 1


        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        print('+++++++++++++')
        for segment in  self.topo.segments:
            print('e: ' , (segment.n1.id,segment.n2.id), e[(segment.n1,segment.n2)] , 'hop:' , self.topo.hopsAway(segment.n1, segment.n2, 'Hop') ,'dist:' , segment.l , 'prob:',segment.p)
            print('remaining qubit ' , segment.n1.remainingQubits , segment.n2.remainingQubits)
        output = []
        for path in self.D:
            successfulEntangledPath = True
            for i in range(len(path) - 1):
                n1 = path[i]
                n2 = path[i+1]
                # print('e[(n1,n2)] hop', n1.id , n2.id , self.topo.hopsAway(n1, n2, 'Hop')    )

                # print('e[(n1,n2)]', n1.id , n2.id , e[(n1,n2)])
                # print('distance between nodes ' , self.topo.distance_by_node(n1.id , n2.id))
                if e[(n1,n2)] < 1:
                    successfulEntangledPath = False
                    break
            if successfulEntangledPath:
                for i in range(len(path) - 1):
                    n1 = path[i]
                    n2 = path[i+1]
                    
                    e[(n1,n2)] -= 1
                    e[(n2,n1)] -= 1
                output.append(path)

                # print('(path[0] , path[-1]) ' , (path[0].id , path[-1].id))
                pathIndex = len(Pi[(path[0] , path[-1])])
                Pi[(path[0] , path[-1])].append(path)
                needLink[((path[0] , path[-1]), pathIndex)] = []
                for nodeIndex in range(1, len(path) - 1):
                    prev = path[nodeIndex - 1]
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    targetLink1 = None
                    targetLink2 = None
                    for segment in node.segments:
                      
                        if segment.contains(next) and segment.entangled and segment.notSwapped() and (segment not in selected[next]) :
                            targetLink1 = segment
                            selected[next].append(segment)

                        
                        if segment.contains(prev) and segment.entangled and segment.notSwapped() and (segment not in selected[prev]):
                            targetLink2 = segment
                            selected[prev].append(segment)


                    
                    if targetLink1 is not None and targetLink2 is not None:
                        needLink[((path[0] , path[-1]), pathIndex)].append((node, targetLink1, targetLink2))
                    else:
                        print('???????????????++++++++?????????????? 1111111111 :' , node.id , [p.id for p in path])
                        for segment in node.segments:
                            if segment.contains(next) and segment.entangled and segment.notSwapped():
                                print('segment.contains(next) ' , 'segment.entangled' , segment.entangled  , 'segment.notSwapped()' , segment.notSwapped())


        print('len(output) before graph' , len(output))

        G2 = nx.Graph()
        for node in self.topo.nodes:
            G2.add_node(node.id , weight = node.q)
        for (n1,n2) in e.keys():
            G2.add_edge(n1.id , n2.id)
        moreEntangledConnection = True
        while moreEntangledConnection:
            moreEntangledConnection = False
            for SDPair in self.srcDstPairs:
                s = SDPair[0]
                d =SDPair[1]
                if len(Pi[SDPair]) < self.numOfFlowPerRequest:  #Ni
                    for u,v,da in G2.edges(data=True):
                        if e[(self.topo.nodes[u] , self.topo.nodes[v])] >=1:
                           da['weight']=math.exp(-5)
                        else:
                           da['weight']=math.exp(9)
                    print('--- ' , s.id ,d.id)
                    length , p = nx.single_source_dijkstra(G2 , s.id , d.id , weight='weight')
                    if len(p) > 0 and length < math.exp(9):
                        print('len of p ' , length)
                        for i in range(len(p) - 1):
                            n1 = p[i]
                            n2 = p[i+1]
                            e[(self.topo.nodes[u] , self.topo.nodes[v])] -= 1
                            e[(self.topo.nodes[v] , self.topo.nodes[u])] -= 1
                        path = [self.topo.nodes[nid] for nid in p ]
                        output.append(path)
                        
                        pathIndex = len(Pi[(path[0] , path[-1])])
                        Pi[(path[0] , path[-1])].append(path)
                        needLink[((path[0] , path[-1]), pathIndex)] = []
                        for nodeIndex in range(1, len(path) - 1):
                            prev = path[nodeIndex - 1]
                            node = path[nodeIndex]
                            next = path[nodeIndex + 1]
                            targetLink1 = None
                            targetLink2 = None
                            for segment in node.segments:
                                if segment.contains(next) and segment.entangled and segment.notSwapped() and (segment not in selected[next]) :
                                    targetLink1 = segment
                                    selected[next].append(segment)

                                
                                if segment.contains(prev) and segment.entangled and segment.notSwapped() and (segment not in selected[prev]):
                                    targetLink2 = segment
                                    selected[prev].append(segment)
                            if targetLink1 is not None and targetLink2 is not None:
                                needLink[((path[0] , path[-1]), pathIndex)].append((node, targetLink1, targetLink2))
                            else:
                                print('???????????????++++++++?????????????? 22222')


                        moreEntangledConnection = True
        print('len(output) after graph' , len(output))
        
        for path in output:
            print('()()()() ' , [p.id for p in path])
            # print(needLink)
            

        for SDpair in self.srcDstPairs:
            src = SDpair[0]
            dst = SDpair[1]
            print('len(Pi[SDpair])' , len(Pi[SDpair]))

            if len(Pi[SDpair]):
                self.result.idleTime -= 1

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                # print('[REPS] attempt:', [node.id for node in path])
                # print('len needLink[(SDpair, pathIndex)]' , len(needLink[(SDpair, pathIndex)]) , pathIndex)
                for (node, segment1, segment2) in needLink[(SDpair, pathIndex)]:
                    swapped = node.attemptSegmentSwapping(segment1, segment2)
                    print('swapped ' , node.id , (segment1.n1.id , segment1.n2.id) , (segment2.n1.id , segment2.n2.id) , swapped)
                successPath = self.topo.getEstablishedEntanglementsWithSegments(src, dst , needLink = needLink[(SDpair, pathIndex)])
                print('++++**************=============len(successPath)' , len(successPath))
                # for x in successPath:
                #     print('[REPS] success:', [z.id for z in x])

                if len(successPath) > 0 or len(path) == 2:
                    for request in self.requests:
                        if (src, dst) == (request[0], request[1]):
                            # print('[REPS] finish time:', self.timeSlot - request[2])
                            self.requests.remove(request)
                            break
                for (node, segment1, segment2) in needLink[(SDpair, pathIndex)]:
                    segment1.clearPhase4Swap()
                    segment2.clearPhase4Swap()


        



        

   
    
    def findPathsForEPI(self, SDpair, n):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPI(SDpair, n):
            print('found dijkstra =====================')
            path = []
            currentNode = dst
            print(len(self.parent))
            # for key in self.parent.keys():
            #     print(key.id , self.parent[key].id)
            while currentNode != self.topo.sentinel:
                # print('in while 2 currentNode ' , currentNode.id)

                path.append(currentNode)
                currentNode = self.parent[currentNode].pop()
                # time.sleep(.1)
            # print('after while loop 2')

            path = path[::-1]
            # width = self.widthForEPS(path, SDpair, n)
            width = 0
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                if width < self.fki_LP[SDpair][n][(node.id, next.id)]:
                    width = self.fki_LP[SDpair][n][(node.id, next.id)]

            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][n][(node.id, next.id)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPI(self, SDpair, n):
        # print('-----------')
        src = SDpair[0]
        dst = SDpair[1]
        # print(src.id , dst.id , n)
        self.parent = {node : [self.topo.sentinel] for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        # for node1 in self.topo.nodes:
        #     for node2 in self.topo.nodes:
        #         if self.edgeSuccessfulEntangle(node1, node2) > 0:
        #             adjcentList[node1].add(node2)
        for node1 , node2 in self.segments:
            adjcentList[node1].add(node2)
            adjcentList[node2].add(node1)
        # print('adjacent list for ' , src.id , [x.id for x in adjcentList[src]])
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = Queue()

        pq.put( src.id)
        while not pq.empty():
            uid = pq.get()
            # print('in dj uid ' , uid)
            u = self.topo.nodes[uid]
            # print('visited[u]' , u.id , visited[u])
            if visited[u]:
                self.parent[u].pop()
                continue

            if u == dst:
                return True
            # distance[u] = -dist
            visited[u] = True
            # print('in while loop adjacent list for ' , u.id , [x.id for x in adjcentList[u]])
            
            for next in adjcentList[u]:
                # print('inside loop1 ' , distance[u] , self.fki_LP[SDpair][n][(u, next)] )
                # newDistance = min(distance[u], self.fki_LP[SDpair][n][(u, next)])
                # # print('inside loop1 ' , distance[next] , newDistance )

                # if distance[next] < newDistance:
                #     distance[next] = newDistance
                #     self.parent[next] = u
                #     # print('parent of ' , next.id , u.id)
                #     pq.put((-distance[next], next.id))
                
                if self.fki_LP[SDpair][n][(u.id, next.id)] > 0:
                    self.parent[next].append(u)
                    # pq.put((self.fki_LP[SDpair][n][(u, next)], next.id))
                    pq.put(next.id)
                    # print('in dj next ' , next.id)
            # print('---+---')




        return False


    
    
if __name__ == '__main__':
    
    topo = Topo.generate(20, 0.9, 5, 0.0002, 6)
    s = SEE(topo)
    result = AlgorithmResult()
    samplesPerTime = 10
    ttime = 10
    rtime = 10
    requests = {i : [] for i in range(ttime)}
    Ni = 5

    for i in range(ttime):
        if i < rtime:

            # ids = [(1, 2)]
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
        print('[REPS] S/D:' , i , [(a[0].id , a[1].id) for a in requests[i]])

    for i in range(ttime):
        result = s.work(requests[i], i)
    

    # print(result.waitingTime, result.numOfTimeslot)