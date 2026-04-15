import sys
import random
import heapq
import math
from queue import PriorityQueue
import networkx as nx
from .Node import Node
from .Link import Link
from .Segment import Segment
from dataclasses import dataclass
from itertools import islice
from random import sample
import itertools
import time
import matplotlib.pyplot as plt
from .helper import entanglement_lifetimeslot
from random import sample
import pickle






@dataclass
class Edge(Node):

    def __init__(self, n1: Node, n2: Node):
        self.n1 = n1
        self.n2 = n2
        self.p_first = []
        self.p_second = []

        p = [self.p_first, self.p_second]

    def toList(self):
        List = [self.n1, self.n2]
        return List

    def otherthan(self, n: Node):
        if n == self.n1:
            return self.n2
        elif n == self.n2:
            return self.n1
        else:
            print('Neither\n')

    def contains(self, n: Node):
        if self.n1 in n or self.n2 in n:
            return True

    def hashCode(self):
        return self.n1.id ^ self.n2.id

    def equals(self, other):
        return other is Edge and (other.n1 == self.n1 and other.n2 == self.n2 or other.n1 == self.n2 and other.n2 == self.n1)

class TopoConnectionChecker:
    def setTopo(self, topo):
        self.topo = topo

    def checkConnected(self):
        self.visited = {node : False for node in self.topo.nodes}
        self.DFS(self.topo.nodes[0])
        for node in self.topo.nodes:
            if not self.visited[node]:
                return False
        return True

    def DFS(self, currentNode):
        self.visited[currentNode] = True
        for link in currentNode.links:
            nextNode = link.theOtherEndOf(currentNode)
            if not self.visited[nextNode]:
                self.DFS(nextNode)

class Topo:

    def __init__(self, G, q, k, a, degree , name = 'waxman'):
        _nodes, _edges, _positions = G.nodes(), list(G.edges()), nx.get_node_attributes(G, 'pos')

        self.nodes = []
        self.links = []
        self.segments = []
        self.edges = [] # (Node, Node)
        self.q = q
        self.alpha = a
        self.k = k
        self.sentinel = Node(-1, (-1.0, -1.0), -1, self)
        self.cacheTable = {}
        self.t_val = 2
        self.usedLinks = set()
        self.G = G
        self.optimal_distance = 130
        self.k_shortest_paths_dict = {}
        self.k_alternate_paths_dict = {}
        self.pair_edge_dict = {}
        self.needLinks = set()
        self.needLinksDict = {}
        self.preSwapFraction = 1/2
        self.tmpcount = 0
        self.entanglementLifetime = 10
        self.requestTimeout = 1
        self.reward = {}
        self.reward_ent = {}
        self.name = name
        self.link_capacity = {}
        self.preswap_capacity = 0.5

        self.positive_reward = 25
        self.negative_reward = -5


        # for pos in _positions:
        #     print(_positions[pos])

        # for _node in _nodes:
        #     for _node2 in _nodes:
        #         if self.distance(_positions[_node], _positions[_node2]) <= 10:
        #             print('node 1:', _node, 'node 2:', _node2, self.distance(_positions[_node], _positions[_node2]))

        # _edges = [] # reset edge

        # Construct neighbor table by int type
        _neighbors = {_node: [] for _node in _nodes}
        x_scale = int(2000 * len(_nodes)/100)
        y_scale = int(2000 * len(_nodes)/100)
        if self.name == 'surfnet':
            x_scale = 200
            y_scale = 200
        for _node in _nodes:
            # print(eval(_positions[_node])[0])
            # return
            if len(_positions[_node]) > 2:
                (p1, p2) = eval(_positions[_node])
            else:
                (p1, p2) = _positions[_node]

            # _positions[_node] = (p1 * x_scale, p2 * y_scale)
            # _positions[_node] = (p1 * 500,  p2 * 500)
            _positions[_node] = (p1 * 10000,  p2 * 10000)
            _neighbors[_node] = list(nx.neighbors(G,_node))
            # print('neighbors of node ' , len(_neighbors[_node]))
          
        # Construct Node 
        #---------
        for _node in _nodes:
            self.nodes.append(Node(_node, _positions[_node], random.random()*5+10 , self))  # 10~14
            # self.nodes.append(Node(_node, _positions[_node], random.random()*11+15 , self))  # 20-40
            # self.nodes.append(Node(_node, _positions[_node], 24 , self))  # 10~14
            usedNode = []
            usedNode.append(_node) 
            
            # Make the number of neighbors approach degree  
            if len(_neighbors[_node]) < degree - 1:  
                for _ in range(0, degree - 1 - len(_neighbors[_node])):
                    curNode = -1
                    curLen = sys.maxsize
                    for _node2 in _nodes:
                        # print(_positions[_node], _positions[_node2])
                        dis = self.distance(_positions[_node], _positions[_node2])
                        if _node2 not in usedNode and _node2 not in _neighbors[_node] and dis < curLen: # no duplicate
                            # print(_positions[_node], _positions[_node2], self.distance(_positions[_node], _positions[_node2]))
                            curNode = _node2
                            curLen = dis
                    # print(curNode)
                    if curNode >= 0:
                        _neighbors[_node].append(curNode)
                        _neighbors[curNode].append(_node)
                        _edges.append((_node, curNode))
                        usedNode.append(curNode)
        # ----------
        self.virtualLinkCount = {(node1 , node2): 0 for node1 in self.nodes for node2 in self.nodes}

                        #print(usedNode)

        # Construct node's neighbor list in Node struct
        for node in self.nodes:
            # print('in node ' , node.id , len(self.nodes))
            for neighbor in _neighbors[node.id]:
                # print('in node ' , neighbor )
                # print('in node 2 ' ,  self.nodes[neighbor].id)

                node.neighbors.append(self.nodes[neighbor])
        
        for edge in _edges:
            node1, node2 = edge
            self.G.add_edge(node1 , node2)
            # print('in cal len ' , node1 , node2)
            self.G.edges[node1, node2]['length'] = self.distance(_positions[node1] , _positions[node2])

        # for _node in _nodes:
        #     print(_node, _neighbors[_node])

        #     for _node2 in _neighbors[_node]:
        #         print(self.distance(_positions[_node], _positions[_node2]))

        # print('average neighbors:', len(_edges)*2/len(_nodes))

        # Construct Link
        linkId = 0
        for _edge in _edges:
            self.edges.append((self.nodes[_edge[0]], self.nodes[_edge[1]]))
            rand = int(random.random()*5+3) # 3~7
            # rand = int(random.random()*6+3) # 3-10
            rand = 3
            self.link_capacity[(_edge[0], _edge[1])] = rand
            self.link_capacity[(_edge[1], _edge[0])] = rand

            for _ in range(0, rand):
                link = Link(self, self.nodes[_edge[0]], self.nodes[_edge[1]], False, False, linkId, self.distance(_positions[_edge[0]], _positions[_edge[1]])) 
                self.links.append(link)
                self.nodes[_edge[0]].links.append(link)
                self.nodes[_edge[1]].links.append(link)

                # self.nodes[_edge[0]].remainingQubits += 1
                # self.nodes[_edge[1]].remainingQubits += 1
                linkId += 1
        self.lastLinkId = linkId
        segmentId = 0
        # for edge in self.edges:
        #     for path,l in self.k_shortest_paths(edge[0].id , edge[1].id):

        #         segment = Segment(self, self.nodes[edge[0]], self.nodes[edge[1]], False, False, segmentId, l , path)
        #         self.segments.append(segment)
        #         self.nodes[edge[0]].segments.append(segment)
        #         self.nodes[edge[1]].segments.append(segment)
        #         linkId += 1
        edgenum = 0
        pairs = list(itertools.combinations(self.nodes, r=2))
        # for node1 in self.nodes:
        #     for node2 in self.nodes:
        for node1 ,node2 in  pairs:
                if  self.distance_by_node(node1.id , node2.id) <=self.optimal_distance or (((node1 , node2) in self.edges) or ((node2 , node1)  in self.edges)):
                # if  (((node1 , node2) in self.edges)) or (((node2 , node1) in self.edges)):
                    k = 0
                    for path,l in self.k_shortest_paths(node1.id , node2.id):
                        edgenum += 1

                        for i in range(self.segmentCapacity(path)):
                            segment = Segment(self, node1, node2, False, False, segmentId, l , path , k)
                            self.segments.append(segment)
                            node1.segments.append(segment)
                            node2.segments.append(segment)
                            segmentId += 1
                        k+=1
                # else:
                #     print('distance of non edge node pair ' , self.distance_by_node(node1.id , node2.id))
                #     time.sleep(.01)
        for node in self.nodes:
            node.maxMem = node.remainingQubits
        
        self.requests = []
        reqFileName = 'request' + str(len(self.nodes)) + '.txt'
        try:
            # print('in try req')
            f = open(reqFileName, 'r')
            for x in f.readlines():
                x = x.replace('\n' , '').replace('(' , '').replace(')' , '').split(',')
                x = [int(i) for i in x]
                x = tuple(x)
                self.requests.append(x)
            f.close()
            # print(self.requests)
        except Exception as e:
            # print('in except req' , e)
            x_cut = int(1000 * (len(self.nodes) / 100))
            y_cut = int(3000 * (len(self.nodes) / 100))
            source_nodes = [node.id for node in self.nodes if node.loc[1] < x_cut]
            dest_nodes = [node.id for node in self.nodes if node.loc[1] > y_cut]
            print(len(source_nodes) , len(dest_nodes))

            for _ in range(200):
                req = (source_nodes[int(random.random()*(len(source_nodes) - 1))] , dest_nodes[int(random.random()*(len(dest_nodes) - 1))])
                if req not in self.requests:
                    self.requests.append(req)
            
            # Using "with open" syntax to automatically close the file
            with open(reqFileName, 'w') as file:
                for req in self.requests:
                    file.write(str(req) + '\n')
        self.get_k_shortest_path_edge_dict(k=5)
        
        print('****** len seg' , len(self.segments))
        print('****** len links' , len(self.links))
        print('****** len edge' , len(self.edges))
        print('****** edgenum' , edgenum)
        print('****** self.req len' , len(self.requests))



        # print p and width for test
        # p = self.shortestPath(self.nodes[3], self.nodes[99], 'Hop')[1]
        # print('Hop path:', [x.id for x in p])
        # print('width:', self.widthPhase2(p))
    def updatedG(self , timeSlot = 0):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.id)
        for link in self.links:
            # print('===========v link ***  in updatedG=========' , (link.n1.id , link.n2.id))
            
            if (link.n1.id , link.n2.id) not in G.edges and (link.n2.id , link.n1.id) not in G.edges:
                # if link.isVirtualLink:
                #     print('===========v link found in updatedG=========' , (link.n1.id , link.n2.id))
                G.add_edge(link.n1.id , link.n2.id)
                G.edges[link.n1.id , link.n2.id]['length'] = self.distance(link.n1.loc , link.n2.loc)
                if link.isEntangled(timeSlot):
                    G.edges[link.n1.id , link.n2.id]['length'] = 0
        return G
    def removeLink(self, link  , skipNodes = []):

        link.n1.links.remove(link)
        link.n2.links.remove(link)

        self.links.remove(link)

        if link.isVirtualLink:
            if self.virtualLinkCount[(link.n1 , link.n2)] > 0: 
                self.virtualLinkCount[(link.n1 , link.n2)] -= 1

            # if link.n1 not in skipNodes:
            #     link.n1.remainingQubits += 1
            # if link.n2 not in skipNodes:
            #     link.n2.remainingQubits += 1


    def printNodeMem(self):
        for node in self.nodes:
            print(node.id , node.remainingQubits , len([link for link in node.links if link.isVirtualLink]) , node.maxMem)
            if node.maxMem  < len([link for link in node.links if link.isVirtualLink]):
                exit()
    
    def addLink(self,link , skipNodes = []):
        self.lastLinkId += 1

        link.n1.links.append(link)
        link.n2.links.append(link)
        self.links.append(link)

        if link.isVirtualLink:
            # if link.n1 not in skipNodes:
            #     link.n1.remainingQubits -= 1
            # if link.n2 not in skipNodes: 
            #     link.n2.remainingQubits -= 1
            
            self.virtualLinkCount[(link.n1 , link.n2)] += 1


    def segmentCapacity(self, path):
        min_capacity = 1000
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            capacity = 0
            for link in self.nodes[p1].links:
                if link.contains(self.nodes[p2]):
                    capacity += 1
            if capacity < min_capacity:
                min_capacity = capacity
        # print('min_capacity',min_capacity , len(path))
        return min_capacity
    def distance(self, pos1: tuple, pos2: tuple): # para1 type: tuple, para2 type: tuple
        d = 0
        for a, b in zip(pos1, pos2):
            d += (a-b) ** 2
        # print('link len ' , d** 0.5)
        return d ** 0.5
    def distance_by_node(self , node1 , node2):
        # print(node1 , node2)
        try:
            (path_cost, sp) = nx.single_source_dijkstra(G=self.G, source=node1, target=node2 , weight='length')
            # print('+_+_+_+_+_+_+_ ' , path_cost)

            return path_cost
        except:
            return math.inf
    def get_k_shortest_path_edge_dict(self , k = 5):
        
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1.id < n2.id:
                    self.pair_edge_dict[(n1.id , n2.id)] = self.k_shortest_paths(n1.id , n2.id , k)
        


    def k_shortest_paths(self, source, target, k = 1):
        # if (source,target) in self.k_shortest_paths_dict:
        #     return self.k_shortest_paths_dict[(source,target)]
        # k=1
        paths_with_len=[]

        paths =  list(
            islice(nx.shortest_simple_paths(self.G, source, target, weight='length'), k)
        )
        # print('in k_shortest_paths ' , len(paths))

        for path in paths:
            dist = 0
            select = True
            for i in range(len(path)-1):
                n1 = path[i]
                n2 = path[i+1]
                edge_dist = self.distance(self.nodes[n1].loc , self.nodes[n2].loc)
                dist += edge_dist

            # if len(path) <= 2 or dist < self.optimal_distance:
            paths_with_len.append((path , dist))
            

        self.k_shortest_paths_dict[(source,target)] = paths_with_len
        return paths_with_len
    
    # def k_alternate_paths(self, source, target , k = 5 , timeSlot = 0):
    #     # if (source,target) in self.k_alternate_paths_dict:
    #     #     return self.k_alternate_paths_dict[(source,target)]
    #     k=k
    #     paths =  list(
    #         islice(nx.shortest_simple_paths(self.updatedG(timeSlot),  source, target), k)
    #     )
    #     self.k_alternate_paths_dict[(source,target)] = paths
    #     return paths
    def k_alternate_paths(self, source, target , k = 5 , timeSlot = 0):
        # if (source,target) in self.k_alternate_paths_dict:
        #     return self.k_alternate_paths_dict[(source,target)]
        k=k
        paths =  list(
            islice(nx.shortest_simple_paths(self.G,  source, target), k)
        )
        self.k_alternate_paths_dict[(source,target)] = paths
        return paths
    def generate( n, q, k, a, degree):
        # dist = lambda x, y: distance(x, y)
        # dist = lambda x, y: sum((a-b)**2 for a, b in zip(x, y))**0.5
        
        checker = TopoConnectionChecker()
        graphFileName = 'graph' + str(n) +'.pickle'
        file = 'SurfnetCore.gml'
        name = 'waxman'
        while True:
            try:
                G = G = pickle.load(open(graphFileName, 'rb'))
            except:
                G = nx.waxman_graph(n, beta=0.9, alpha=0.01, domain=(0, 0, 1, 2))
                pickle.dump(G, open(graphFileName, 'wb'))
            

            G = nx.waxman_graph(n, beta=0.9, alpha=0.001, domain=(0, 0, 1, 1))

            # name = 'surfnet'
            # G = nx.read_gml(file)

            # G = Topo.create_custom_graph()
            print('leeeen ' , len(G.edges))
            # Topo.draw_graph(G)

            topo = Topo(G, q, k, a, degree , name)
            checker.setTopo(topo)
            if checker.checkConnected():
                break
            else:
                print("topo is not connected", file = sys.stderr)
        return topo
    def create_custom_graph():

       
        G = nx.Graph()
        G.add_node(0, pos=[.5, 0.3])
        G.add_node(1, pos=[0, 0])
        G.add_node(2, pos=[.3, 0])
        G.add_node(3, pos=[.7, 0])
        G.add_node(4, pos=[.9, 0])
        G.add_node(5, pos=[0, .3])
        G.add_node(6, pos=[.3, .3])
        G.add_node(7, pos=[.7, .3])
        G.add_node(8, pos=[.9, .3])
        G.add_node(9, pos=[0, .7])
        G.add_node(10, pos=[.4, .7])
        G.add_node(11, pos=[.9, .7])
        G.add_node(12, pos=[.2, .8])
        G.add_node(13, pos=[.7, .8])
        G.add_node(14, pos=[.9, .9])
        G.add_node(15, pos=[.6, .9])
        G.add_node(16, pos=[.3, .9])
        G.add_node(17, pos=[0, .9])

        G.add_edge(0, 1)
        G.add_edge(0, 13)

        G.add_edge(1, 2)
        G.add_edge(1, 5)
        G.add_edge(1, 6)
        G.add_edge(2, 3)
        G.add_edge(2, 7)
        G.add_edge(3, 4)
        G.add_edge(4, 8)
        G.add_edge(5, 9)
        G.add_edge(5, 12)
        G.add_edge(6, 10)
        G.add_edge(7, 11)
        G.add_edge(7, 13)
        G.add_edge(8, 11)
        G.add_edge(9, 17)
        G.add_edge(10, 12)
        G.add_edge(10, 13)
        G.add_edge(11, 14)
        G.add_edge(12, 16)
        G.add_edge(12, 15)
        G.add_edge(13, 14)
        G.add_edge(14, 15)
        G.add_edge(15, 16)
        G.add_edge(16, 17)



        for node in G.nodes():
            G.nodes[node]['xcoord'] = G.nodes[node]['pos'][0]
            G.nodes[node]['ycoord'] = G.nodes[node]['pos'][1]
        # Topo.draw_graph(G)
        return G
    
    def draw_graph(G):
        pos = nx.get_node_attributes(G, 'pos')
        repeater_nodes = []
        end_nodes = []
        for node in G.nodes():
            end_nodes.append(node)
        fig, ax = plt.subplots(figsize=(7, 7))
        end_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=end_nodes, node_shape='s', node_size=150,
                                        node_color=[[1.0, 120 / 255, 0.]], label="End Node", linewidths=3)
        end_nodes.set_edgecolor('k')
        rep_nodes = nx.draw_networkx_nodes(G=G, pos=pos, nodelist=repeater_nodes, node_size=150,
                                        node_color=[[1, 1, 1]], label="Repeater Node")
        rep_nodes.set_edgecolor('k')
        end_node_labels = {}
        repeater_node_labels = {}
        for node, nodedata in G.nodes.items():
            end_node_labels[node] = node

        nx.draw_networkx_labels(G=G, pos=pos, labels=end_node_labels, font_size=7, font_weight="bold", font_color="w",
                                font_family='serif')
        nx.draw_networkx_labels(G=G, pos=pos, labels=repeater_node_labels, font_size=5, font_weight="bold")
        nx.draw_networkx_edges(G=G, pos=pos, width=1)
        plt.axis('off')
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1. - margin, 1. - margin)
        ax.axis('equal')
        fig.tight_layout()
        plt.show()

    def widthPhase2(self, path):
        curMinWidth = min(path[0].remainingQubits, path[-1].remainingQubits)

        # Check min qubits in path
        for i in range(1, len(path) - 1):
            if path[i].remainingQubits / 2 < curMinWidth:
                curMinWidth = path[i].remainingQubits // 2
        # print('curMinWidth: ' , curMinWidth) 

        # Check min links in path
        for i in range(0, len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            t = 0
            for link in n1.links:
                if link.contains(n2) and not link.assigned:
                    t += 1

            if t < curMinWidth:
                curMinWidth = t

        # print('curMinWidth_: ' , curMinWidth) 

        return curMinWidth
    
    def widthByProbPhase2(self , path , timeslot = 0):
        curMaxWidth = 9999

        for i in range(0, len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            t = 0
            for link in n1.links:
                if link.contains(n2) and link.assigned:
                    if link.isEntangled(timeslot):
                        t += 1
                    else:
                        # print('link.p ' , link.p)
                        t += link.p()
            # print('t ' , t)
            if t < curMaxWidth:
                curMaxWidth = t
        return curMaxWidth
    
    def shortestPath(self, src, dst, greedyType, edges = None):
        return self.shortestPathForPreswap( src, dst, greedyType, edges )

    def shortestPathForPreswap(self, src, dst, greedyType, edges = None):
        temp_edges = set()
        for link in self.links:
            n1 = link.n1
            n2 = link.n2
            if (n1,n2) not in temp_edges and (n2,n1) not in temp_edges:
                temp_edges.add((n1,n2))
        fStateMetric = {}   # {edge: fstate}
        fStateMetric.clear()
        if edges != None:
            fStateMetric = {edge : self.distance(edge[0].loc, edge[1].loc) for edge in edges} 
        elif greedyType == 'Hop' and edges == None:
            fStateMetric = {edge : 1 for edge in temp_edges}
        else: 
            fStateMetric = {edge : self.distance(edge[0].loc, edge[1].loc) for edge in temp_edges}

        # Construct neightor & weight table for nodes
        neighborsOf = {node: {} for node in self.nodes}    # {Node: {Node: weight, ...}, ...}
        if edges == None:
            for edge in temp_edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]
        else:
            for edge in edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]

        D = {node.id : sys.float_info.max for node in self.nodes} # {int: [int, int, ...], ...}
        q = [] # [(weight, curr, prev)]

        D[src.id] = 0.0
        prevFromSrc = {}   # {cur: prev}

        q.append((D[src.id], src, self.sentinel))
        sorted(q, key=lambda q: q[0])

        # Dijkstra 
        while len(q) != 0:
            contain = q.pop(0)
            w, prev = contain[1], contain[2]
            if w in prevFromSrc.keys():
                continue
            prevFromSrc[w] = prev

            # If find the dst return D & path 
            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinel:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]          
                return (D[dst.id], path)
            
            # Update neighbors of w  
            for neighbor in neighborsOf[w]:
                weight = neighborsOf[w][neighbor]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]

                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.append((D[neighbor.id], neighbor, w))
                    sorted(q, key=lambda q: q[0])

        return (sys.float_info.max, [])


    def shortestPath2(self, src, dst, greedyType, edges = None):
        # Construct state metric (weight) table for edges
        fStateMetric = {}   # {edge: fstate}
        fStateMetric.clear()
        if edges != None:
            fStateMetric = {edge : self.distance(edge[0].loc, edge[1].loc) for edge in edges} 
        elif greedyType == 'Hop' and edges == None:
            fStateMetric = {edge : 1 for edge in self.edges}
        else: 
            fStateMetric = {edge : self.distance(edge[0].loc, edge[1].loc) for edge in self.edges}

        # Construct neightor & weight table for nodes
        neighborsOf = {node: {} for node in self.nodes}    # {Node: {Node: weight, ...}, ...}
        if edges == None:
            for edge in self.edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]
        else:
            for edge in edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]

        D = {node.id : sys.float_info.max for node in self.nodes} # {int: [int, int, ...], ...}
        q = [] # [(weight, curr, prev)]

        D[src.id] = 0.0
        prevFromSrc = {}   # {cur: prev}

        q.append((D[src.id], src, self.sentinel))
        sorted(q, key=lambda q: q[0])

        # Dijkstra 
        while len(q) != 0:
            contain = q.pop(0)
            w, prev = contain[1], contain[2]
            if w in prevFromSrc.keys():
                continue
            prevFromSrc[w] = prev

            # If find the dst return D & path 
            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinel:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]          
                return (D[dst.id], path)
            
            # Update neighbors of w  
            for neighbor in neighborsOf[w]:
                weight = neighborsOf[w][neighbor]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]

                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.append((D[neighbor.id], neighbor, w))
                    sorted(q, key=lambda q: q[0])

        return (sys.float_info.max, [])
        

    def hopsAway(self, src, dst, greedyType):
        # print('enter hopsAway')
        path = self.shortestPath(src, dst, greedyType)
        return len(path[1]) - 1


    def hopsAway2(self, src, dst, greedyType):
        # print('enter hopsAway')
        path = self.shortestPath2(src, dst, greedyType)
        # print([p.id for p in path[1]])
        return len(path[1]) - 1

    def e(self, path: list, width: int, oldP: list):
        s = len(path) - 1
        P = [0.0 for _ in range(0,width+1)]
        p = [0 for _ in range(0, s+1)]  # Entanglement percentage
        
        for i in range(0, s):
            l = self.distance(path[i].loc, path[i+1].loc)
            p[i+1] = math.exp(-self.alpha * l)

        start = s
        if sum(oldP) == 0:
            for m in range(0, width+1):
                oldP[m] = math.comb(width, m) * math.pow(p[1], m) * math.pow(1-p[1], width-m)
                start = 2
        
        for k in range(start, s+1):
            for i in range(0, width+1):
                exactlyM = math.comb(width, i) *  math.pow(p[k], i) * math.pow(1-p[k], width-i)
                atLeastM = exactlyM

                for j in range(i+1, width+1):
                    atLeastM += (math.comb(width, j) * math.pow(p[k], j) * math.pow(1-p[k], width-j))

                acc = 0
                for j in range(i+1, width+1):
                    acc += oldP[j]
                
                P[i] = oldP[i] * atLeastM + exactlyM * acc
            
            for i in range(0, width+1):
                oldP[i] = P[i]
        
        acc = 0
        for m in range(1, width+1):
            acc += m * oldP[m]
        
        return acc * math.pow(self.q, s-1)



    def e2(self, path: list, width: int, oldP: list):
        s = len(path) - 1
        P = [0.0 for _ in range(0,width+1)]
        p = [0 for _ in range(0, s+1)]  # Entanglement percentage
        
        for i in range(0, s):
            # if len(self.getEstablishedEntanglements(path[i] , path[i+1])) > 0:
            #     p[i+1] = 1
            #     # print('+++++=====++++ ent prob ' ,p[i+1] , len(self.getEstablishedEntanglements(path[i] , path[i+1])))

            # else:
            #     l = self.distance(path[i].loc, path[i+1].loc)
            #     p[i+1] = math.exp(-self.alpha * l)
            l = self.distance(path[i].loc, path[i+1].loc)
            p[i+1] = math.exp(-self.alpha * l)

        start = s
        if sum(oldP) == 0:
            for m in range(0, width+1):
                oldP[m] = math.comb(width, m) * math.pow(p[1], m) * math.pow(1-p[1], width-m)
                start = 2
        
        for k in range(start, s+1):
            for i in range(0, width+1):
                exactlyM = math.comb(width, i) *  math.pow(p[k], i) * math.pow(1-p[k], width-i)
                atLeastM = exactlyM

                for j in range(i+1, width+1):
                    atLeastM += (math.comb(width, j) * math.pow(p[k], j) * math.pow(1-p[k], width-j))

                acc = 0
                for j in range(i+1, width+1):
                    acc += oldP[j]
                
                P[i] = oldP[i] * atLeastM + exactlyM * acc
            
            for i in range(0, width+1):
                oldP[i] = P[i]
        
        acc = 0
        for m in range(1, width+1):
            acc += m * oldP[m]
        
        return acc * math.pow(self.q, s-1)
    

    def getEstablishedEntanglements(self, n1: Node, n2: Node , timeSlot = 0):
        stack = []
        stack.append((None, n1)) #Pair[Link, Node]
        result = []

        while stack:
            (incoming, current) = stack.pop()
            # if incoming != None:
            #     print(incoming.n1.id, incoming.n2.id, current.id)

            if current == n2:
                path = []
                path.append(n2)
                inc = incoming
                while inc.n1 != n1 and inc.n2 != n1:
                    if inc.n1 == path[-1]:
                        prev = inc.n2
                    elif inc.n2 == path[-1]:
                        prev = inc.n1
                        
                    #inc = prev.internalLinks.first { it.contains(inc) }.otherThan(inc)
                    for internalLinks in prev.internalLinks:
                        # if inc in internalLinks:
                        #     for links in internalLinks:
                        #         if inc != links:
                        #             inc = links
                        #             break
                        #         else:
                        #             continue
                        #     break
                        # else:
                        #     continue
                        (l1, l2) = internalLinks
                        if l1 == inc:
                            inc = l2
                            break
                        elif l2 == inc:
                            inc = l1
                            break

                    path.append(prev)

                path.append(n1)
                path.reverse()
                result.append(path)
                continue

            outgoingLinks = []
            if incoming is None:
                for links in current.links:
                    if links.isEntangled(timeSlot) and not links.swappedAt(current):
                        outgoingLinks.append(links)
            else:
                for internalLinks in current.internalLinks:
                    # for links in internalLinks:
                    #     if incoming != links:
                    #         outgoingLinks.append(links)
                    (l1, l2) = internalLinks
                    if l1 == incoming:
                        outgoingLinks.append(l2)
                    elif l2 == incoming:
                        outgoingLinks.append(l1)
                    
            
            for l in outgoingLinks:
                if l.n1 == current:
                    stack.append((l, l.n2))
                elif l.n2 == current:
                    stack.append((l, l.n1))

        return result

    def getEstablishedEntanglements2(self, n1: Node, n2: Node , timeSlot = 0):
        stack = []
        stack.append((None, n1)) #Pair[Link, Node]
        result = []

        while stack:
            (incoming, current) = stack.pop()
            # if incoming != None:
            #     print(incoming.n1.id, incoming.n2.id, current.id)

            if current == n2:
                path = []
                path.append(n2)
                inc = incoming
                while inc.n1 != n1 and inc.n2 != n1:
                    if inc.n1 == path[-1]:
                        prev = inc.n2
                    elif inc.n2 == path[-1]:
                        prev = inc.n1
                        
                    #inc = prev.internalLinks.first { it.contains(inc) }.otherThan(inc)
                    for internalLinks in prev.internalLinks:
                        # if inc in internalLinks:
                        #     for links in internalLinks:
                        #         if inc != links:
                        #             inc = links
                        #             break
                        #         else:
                        #             continue
                        #     break
                        # else:
                        #     continue
                        (l1, l2) = internalLinks
                        if l1 == inc:
                            inc = l2
                            break
                        elif l2 == inc:
                            inc = l1
                            break

                    path.append(prev)

                path.append(n1)
                path.reverse()
                result.append(path)
                continue

            outgoingLinks = []
            if incoming is None:
                for links in current.links:
                    if links.isEntangled(timeSlot) and not links.swappedAt(current):
                        outgoingLinks.append(links)
            else:
                for internalLinks in current.internalLinks:
                    # for links in internalLinks:
                    #     if incoming != links:
                    #         outgoingLinks.append(links)
                    (l1, l2) = internalLinks
                    if l1 == incoming:
                        outgoingLinks.append(l2)
                    elif l2 == incoming:
                        outgoingLinks.append(l1)
                    
            
            for l in outgoingLinks:
                if l.n1 == current:
                    stack.append((l, l.n2))
                elif l.n2 == current:
                    stack.append((l, l.n1))      # else:
                #     print('distance of non edge node pair ' , self.distance_by_node(node1.id , node2.id))
                #     time.sleep(.01)

        return result


    def getEstablishedEntanglementsWithLinks(self, n1: Node, n2: Node , timeSlot = 0):
        stack = []
        stack.append((None, n1)) #Pair[Link, Node]
        result = []
        # print('----------------------' , n1.id , n2.id)
        while stack:
            (incoming, current) = stack.pop()
            # if incoming != None:
            #     print(incoming.n1.id, incoming.n2.id, current.id)

            if current == n2:
                path = []
                path.append((n2, None))
                inc = incoming
                # prev = None
                while inc.n1 != n1 and inc.n2 != n1:
                    if inc.n1 == path[-1][0]:
                        prev = inc.n2
                    elif inc.n2 == path[-1][0]:
                        prev = inc.n1
                    path.append((prev , inc))
                        
                    #inc = prev.internalLinks.first { it.contains(inc) }.otherThan(inc)
                    for internalLinks in prev.internalLinks:
                        # if inc in internalLinks:
                        #     for links in internalLinks:
                        #         if inc != links:
                        #             inc = links
                        #             break
                        #         else:
                        #             continue
                        #     break
                        # else:
                        #     continue
                        (l1, l2) = internalLinks
                        if l1 == inc:
                            inc = l2
                            break
                        elif l2 == inc:
                            inc = l1
                            break


                path.append((n1 , inc))
                path.reverse()
                result.append(path)
                continue

            outgoingLinks = []
            if incoming is None:
                for links in current.links:
                    if links.isEntangled(timeSlot) and not links.swappedAt(current):
                        outgoingLinks.append(links)
            else:
                for internalLinks in current.internalLinks:
                    # for links in internalLinks:
                    #     if incoming != links:
                    #         outgoingLinks.append(links)
                    (l1, l2) = internalLinks
                    if l1 == incoming:
                        outgoingLinks.append(l2)
                    elif l2 == incoming:
                        outgoingLinks.append(l1)
                    
            
            for l in outgoingLinks:
                if l.n1 == current:
                    stack.append((l, l.n2))
                elif l.n2 == current:
                    stack.append((l, l.n1))
        # print('-----------end-----------' )

        return result


    def getEstablishedEntanglementsWithSegments(self, n1: Node, n2: Node , timeSlot = 0 , needLink = None):
        stack = []
        stack.append((None, n1)) #Pair[Link, Node]
        result = []
        # print('n1 n2 ' , n1.id , n2.id)

        while stack:
            (incoming, current) = stack.pop()
            # print('current ' , current.id)
            # print('len stack ' , len(stack))

            # if incoming != None:
            #     print(incoming.n1.id, incoming.n2.id, current.id)

            if current == n2:
                path = []
                path.append((n2, None))
                inc = incoming
                # prev = None
                while inc.n1 != n1 and inc.n2 != n1:
                    if inc.n1 == path[-1][0]:
                        prev = inc.n2
                    elif inc.n2 == path[-1][0]:
                        prev = inc.n1
                    # print('prev ' , prev.id)
                    time.sleep(.1)
                    #inc = prev.internalLinks.first { it.contains(inc) }.otherThan(inc)
                    for internalLinks in prev.internalSegments:

                        (l1, l2) = internalLinks
                        if l1 == inc:
                            inc = l2
                            break
                        elif l2 == inc:
                            inc = l1
                            break

                    path.append((prev , inc))

                path.append((n1 , inc))
                path.reverse()
                result.append(path)
                continue

            outgoingLinks = []
            if incoming is None:
                # print('len current.segments when incoming is none ' , len(current.segments))
                for links in current.segments:
                    # print('links.entangled ' , links.entangled , 'links.swappedAt(current) ' , links.swappedAt(current))
                    if links.entangled and not links.swappedAt(current):
                        outgoingLinks.append(links)
                    # if links.entangled and links.swappedAt(current):
                    #     print('!!!!!!!!!!!!!!!!!   links.entangled and links.swappedAt(current)')
                # if len(outgoingLinks) <= 0:
                #     node , link1 , link2 = needLink[0]
                #     print('*** *** *** ' , node.id , link1.n1.id , link1.n2.id , link2.n1.id , link2.n2.id)
                # print('len outgoing link for first node ' , len(outgoingLinks))
            else:
                # print('len internalSegments  for  ', current.id , len(current.internalSegments))
                for internalLinks in current.internalSegments:
                    # for links in internalLinks:
                    #     if incoming != links:
                    #         outgoingLinks.append(links)
                    (l1, l2) = internalLinks
                    if l1 == incoming:
                        outgoingLinks.append(l2)
                    elif l2 == incoming:
                        outgoingLinks.append(l1)
                #     print('+++++++++=== l1' , l1.n1.id , l1.n2.id ,'l2' ,  l2.n1.id , l2.n2.id )
                # print('++++++++++++++++===+++++++++++for outgoing ' , current.id , [[((seg[0].n1.id , seg[0].n2.id) , (seg[1].n1.id , seg[1].n2.id)) for seg in current.internalSegments]])
                
                # print('len outgoing link  ' , len(outgoingLinks))
                
            
            for l in outgoingLinks:
                if l.n1 == current:
                    stack.append((l, l.n2))
                elif l.n2 == current:
                    stack.append((l, l.n1))

        return result

        
    def clearAllEntanglements(self):
        for link in self.links:

            link.clearEntanglement()
    def restoreOriginalLinks(self , vLink , expired = False):
        for link in vLink.subLinks:
            if expired:
                edge = self.linktoEdgeSorted(link)
                self.reward_ent[edge] = self.negative_reward
            link.clearEntanglement()
            self.addLink(link)
        vLink.subLinks.clear()
    def generateRequest(self , numOfRequestPerRound):
        # ids = []
        ret = []
        # source_nodes = [node.id for node in self.nodes if node.loc[1] < 1000]
        # dest_nodes = [node.id for node in self.nodes if node.loc[1] > 3000]
        # print(len(source_nodes) , len(dest_nodes))
        # for _ in range(100):
        #     ids.append((source_nodes[int(random.random()*(len(source_nodes) - 1))] , dest_nodes[int(random.random()*(len(dest_nodes) - 1))]))
        ret = sample(self.requests , numOfRequestPerRound)
        # for _ in range(numOfRequestPerRound):
        #     ret.append(self.requests[int(random.random()*30) + 50])
        # print('reqs ' , ret)
        return ret

    def linktoEdgeSorted(self , link):
        edge = (link.n1 , link.n2)
        if link.n2.id < link.n1.id:
            edge = (link.n2 , link.n1)

        return edge


    def resetEntanglement(self , timeslot = 0):
        for link in self.usedLinks:
            link.clearEntanglement(timeslot = timeslot)
            # self.reward_ent[link] = 50 - (timeslot - link.entangledTimeSlot)
            # if link.isVirtualLink:
            #     for link_ in link.subLinks:
            #         self.addLink(link_) 
            #     self.removeLink(link)

        for link in set(self.links).difference(self.usedLinks):
            if timeslot - link.entangledTimeSlot >=  self.entanglementLifetime:
                link.clearEntanglement(expired = True , timeslot = timeslot)
            else:
                if link.isVirtualLink:
                    self.restoreOriginalLinks(link)
                link.keepEntanglementOnly()
            
        self.usedLinks.clear()

        # for link in self.links:
        #     if link.isVirtualLink:
        #         if not link.isEntangled(timeslot):
        #             for link_ in link.subLinks:
        #                 link_.clearEntanglement()
        #                 self.addLink(link_)                    
        #             self.removeLink(link)


        

    def preEntanglement(self):
        for sd in self.cacheTable:
            if self.cacheTable[sd] > self.t_val:
                # print('[Cache Ent]')
                links = []
                n1, n2 = sd[0] ,sd[1]

                for link in n1.links:
                    if link.contains(n2) and not link.assigned:
                        links.append(link)
                for link in links:
                    link.assigned = True
                    for _ in range(20):
                        if link.tryEntanglement():
                            break
                    # if not link.entangled:

                    #     print('***!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Ent not established :() ')

                    link.assigned = False
                    # print('ent prob ' , link.p , link.entangled)
                    

    def updateLinks(self):
        for link in self.links:
            l = self.distance(link.n1.loc, link.n2.loc)
            link.alpha = self.alpha
            # link.p = math.exp(-self.alpha * l)
    
    def updateNodes(self):
        for node in self.nodes:
            node.q = self.q

    def setAlpha(self, alpha):
        self.alpha = alpha
        self.updateLinks()
        self.updateNodes()

    def setQ(self, q):
        self.q = q
        self.updateLinks()
        self.updateNodes()