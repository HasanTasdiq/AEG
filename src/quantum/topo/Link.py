
from .Node import Node
import random
import math
from .helper import entanglement_lifetimeslot

class Link:
    
    def __init__(self, topo, n1: Node, n2: Node, s1: bool, s2: bool, id: int, l: float , isVirtualLink = False, vtimeslot = 0):
        self.id, self.n1, self.n2, self.s1, self.s2, self.alpha = id, n1, n2, s1, s2, topo.alpha
        self.assigned = False
        self.entangled = False
        self.entangledTimeSlot = 0
        # self.p = math.exp(-self.alpha * l)
        # print('rnt prob ' , self.p, self.alpha , l)
        self.l = l
        self.isVirtualLink = isVirtualLink
        self.vtimeslot = vtimeslot
        self.subLinks = []
        self.topo = topo
        self.used = False
        # print(self.n1.id, self.n2.id, self.p)
    def p(self):
        # print('rnt prob ' , self.alpha , self.l , math.exp(-self.alpha * self.l))
         
        return math.exp(-self.alpha * self.l)
    def theOtherEndOf(self, n: Node): 
        if (self.n1 == n): 
            tmp = self.n2
        elif (self.n2 == n): 
            tmp = self.n1 
        return tmp
    def contains(self, n: Node):  
        return self.n1 == n or self.n2 == n
    def swappedAt(self, n: Node): 
        return ((self.n1 == n and self.s1) or (self.n2 == n and self.s2))
    def swappedAtTheOtherEndOf(self, n: Node):  
        return ((self.n1 == n and self.s2) or (self.n2 == n and self.s1))
    def swapped(self):  
        return self.s1 or self.s2
    def notSwapped(self):  
        return not self.swapped()
    def isEntangled(self , timeSlot):
        return self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime



    def assignQubits(self):
        # prevState = self.assigned
        self.assigned = True
        if not self.isVirtualLink:
            self.n1.remainingQubits -= 1
            self.n2.remainingQubits -= 1
  
    def clearEntanglement(self , expired = False , timeslot = 0):
        preState = self.assigned
        self.s1 = False
        self.s2 = False
        self.assigned = False
        self.entangled = False
        self.used = False


        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)
        if self.isVirtualLink and self in self.topo.links:
            
            self.topo.restoreOriginalLinks(self , expired)                 
            self.topo.removeLink(self)

        if preState and not self.isVirtualLink:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
        
        reward2 = 10 - (timeslot - self.vtimeslot)
        
        if self.isVirtualLink and expired:
            if (self.n1.id , self.n2.id , self.vtimeslot ) in self.topo.reward:
                reward = self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )]
                self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = reward - reward2
            else:     
                 self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = -reward2
        elif self.isVirtualLink:
            if (self.n1.id , self.n2.id , self.vtimeslot ) in self.topo.reward:
                reward = self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )]
                self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = reward + reward2
            else:     
                 self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = reward2
        elif expired:
             self.topo.reward_ent[self] = self.topo.negative_reward

        # if self.isVirtualLink and expired:
        #     if (self.n1.id , self.n2.id , self.vtimeslot ) in self.topo.reward:
        #         reward = self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )]
        #         self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = reward + (-10)
        #     else:     
        #          self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = -10 
        # elif self.isVirtualLink:
        #      self.topo.reward[(self.n1.id , self.n2.id , self.vtimeslot )] = 10 - (timeslot - self.vtimeslot)



    def keepEntanglementOnly(self):
        preState = self.assigned
        self.s1 = False
        self.s2 = False
        self.assigned = False
        # self.entangled = False

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)

        if preState and not self.isVirtualLink:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
    
    def clearPhase4Swap(self):
        self.s1 = False
        self.s2 = False
        self.entangled = False

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)
                # if self.isVirtualLink and self in self.n2.links:
                #     self.n2.links.remove(self)

        for internalLink in self.n2.internalLinks:
            if self in internalLink:
                self.n2.internalLinks.remove(internalLink)
                # if self.isVirtualLink and self in self.n1.links:
                #     self.n1.links.remove(self)

        for internalSegment in self.n1.internalSegments:
            if self in internalSegment:
                self.n1.internalSegments.remove(internalSegment)

        for internalSegment in self.n2.internalSegments:
                    if self in internalSegment:
                        self.n2.internalSegments.remove(internalSegment)
        # if self.isVirtualLink:

                        
    def keepPhase4Swap(self):
        self.s1 = False
        self.s2 = False
        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)
                # self.n1.prevInternalLinks.append((internalLink[0] , internalLink[1] , self.entangledTimeSlot))
                self.n1.prevInternalLinks.append((internalLink[0] , internalLink[1] ))

        for internalLink in self.n2.internalLinks:
                    if self in internalLink:
                        self.n2.internalLinks.remove(internalLink)
                        # self.n2.prevInternalLinks.append((internalLink[0] , internalLink[1] , self.entangledTimeSlot))
                        self.n2.prevInternalLinks.append((internalLink[0] , internalLink[1] ))
        
         
    # def tryEntanglement(self , timeSlot = 0):
    #     # print('ent prob', self.p)

    #     b = self.assigned and self.p >= random.random()
    #     if b:
    #         self.entangledTimeSlot = timeSlot
    #         self.entangled = b
            
    #     if self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime:
    #         return True
    #     return b

    def tryEntanglement3(self , timeSlot = 0):

        b = self.assigned and self.p() >= random.random()
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
            
        # if self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime:
        #     # if not b:
        #     #     print('ent not prob', self.p , b)

        #     return True
        # if not b:
        #     print('ent prob', self.p , b)
        
        return b


    def tryEntanglement2(self , timeSlot = 0):
        # print('ent prob', self.p)
        # print('+++++++++++++++++++++++ every')
        b = self.assigned and self.p() >= random.random()
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
            
        if self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime:
            # print('tryEntanglement2 ' , self.n1.id , self.n2.id , timeSlot , True)
            return True

        # print('tryEntanglement2 ', self.n1.id , self.n2.id , timeSlot , b)
        return b

    def tryEntanglement1(self , timeSlot = 0):
        b =self.assigned and self.p() >= random.random()
        # b = self.assigned and self.p >= random.random()
 
        if self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime:
            # if not b:
            #     print('######cached entanglement found########================' , self.topo.entanglementLifetime)
            return True
        if b:
            self.entangledTimeSlot = timeSlot
            self.entangled = b
        # print('ent prob', self.p , b)
        
        return b
    
    def tryEntanglement(self , timeSlot = 0, param=None):
        if param == 'every':
            if self.assigned:
                return self.tryEntanglement2(timeSlot)  
            else:
                return self.tryEntanglementForUnassigned(timeSlot) 
            return self.tryEntanglement2(timeSlot)
        elif param == 'ten' or param == 'everya':
            if self.assigned:
                return self.tryEntanglement1(timeSlot)  
            else:
                return self.tryEntanglementForUnassigned(timeSlot)
        elif param == 'reps_ten' or param == 'everya':
            if self.assigned:
                return self.tryEntanglement3(timeSlot)  
            else:
                return self.tryEntanglementForUnassigned(timeSlot)
        else:
            return self.tryEntanglement3(timeSlot)
    

    def tryEntanglementForUnassigned(self , timeSlot = 0):
        if not self.assigned:
            if self.assignable():
                b = self.p() >= random.random()

                if self.isEntangled(timeSlot):
                    #  if not b:
                    #     print('######cached entanglement found for unassigned########================' , timeSlot - self.entangledTimeSlot)

                     return True
                # if self.entangled and (timeSlot - self.entangledTimeSlot) < self.topo.entanglementLifetime:
                #     if not b:
                #         print('######cached entanglement found########================' , timeSlot - self.entangledTimeSlot)
                #     return True
                if b:
                    self.entangledTimeSlot = timeSlot
                    self.entangled = b
                    # print('++++++++++++++++++++++++++++++++++++++++====entangled unassigned ==================++++++++++++++++++++++++')
                    # self.assignQubits()
                # print('ent prob', self.p , b)

                return b

    
  
    def assignable(self): 
        return not self.assigned and self.n1.remainingQubits > 0 and self.n2.remainingQubits > 0
