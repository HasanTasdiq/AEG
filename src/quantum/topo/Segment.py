from .Node import Node
import random
import math
from .Link import Link

class Segment(Link):
    def __init__(self, topo, n1: Node, n2: Node, s1: bool, s2: bool, id: int, l: float , path , k):
        super().__init__(topo ,  n1, n2, s1, s2, id, l)
        self.path = path
        self.k = k