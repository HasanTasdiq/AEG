import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append("../../..")

from RoutingEnv import RoutingEnv      #for ubuntu
# from .RoutingEnv import RoutingEnv   #for mac
from itertools import combinations
import random
import pickle 



#Hyperparameters
NUM_EPISODES = 2500
LEARNING_RATE = 0.3
DISCOUNT = 0.95

GAMMA = 0.99

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 0.5  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 2000
EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


class Agent():
    def __init__(self , algo , pid):
        #Initializations
        self.env = RoutingEnv(algo)
        # env = gym.make('Breakout-v0')
        # env.reset()
        self.nA = 2
        self.dim = 2 #ball position x*width + ball pos y and player position
        # Init weight
        self.w = np.random.rand(self.dim, self.nA)

        # Keep stats for final print of graph
        self.episode_rewards = []
        # self.qFileName = 'q_table' + str(len(algo.topo.nodes)) + '_' + str(pid) + '.pkl'
        # try:
        #     print('in agent try' , pid)
        #     with open(self.qFileName, 'rb') as f:
        #         self.q_table = pickle.load(f)
        # except:
        #     print('in agent except')

        #     self.q_table = {x:[random.random(), random.random()] for x in list(combinations([n.id for n in  self.env.algo.topo.nodes], 2)) }
        #     with open(self.qFileName, 'wb') as f:
        #         pickle.dump(self.q_table, f)
        self.q_table = {x:[random.random(), random.random()] for x in list(combinations([n.id for n in  self.env.algo.topo.nodes], 2)) }

        self.last_action_table = {}
        # print(self.q_table)


   
    def learn_and_predict(self):
        global EPSILON_
        state = self.env.reset()
        timeSlot = self.env.algo.timeSlot
        for pair in state:
            if (pair[0].id, pair[1].id) in self.q_table:
                if np.random.random() > EPSILON_:
                    # Get action from Q table
                    action = np.argmax(self.q_table[(pair[0].id, pair[1].id)])
                    # print('==Action from Q table==')
                else:
                    # Get random action
                    # print('##from random action##')

                    action = np.random.randint(0, 2)
                if not (pair[0].id, pair[1].id) in self.last_action_table:
                    self.last_action_table[(pair[0].id, pair[1].id)] = [(action , timeSlot)]
                else:
                    self.last_action_table[(pair[0].id, pair[1].id)].append((action , timeSlot))

            else:
                if np.random.random() > EPSILON_:
                    # Get action from Q table
                    action = np.argmax(self.q_table[(pair[1].id, pair[0].id)])
                else:
                    # Get random action
                    action = np.random.randint(0, 2)
                if not (pair[1].id, pair[0].id) in self.last_action_table:
                    self.last_action_table[(pair[1].id, pair[0].id)] = [(action , timeSlot)]
                else:
                    self.last_action_table[(pair[1].id, pair[0].id)].append((action , timeSlot))

            # print('llllll ', action)
            self.env.step(pair , action , timeSlot)
        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE

            # print(len(self.env.algo.topo.needLinksDict) , action)
    def update_reward(self):
        # print('update reward: ' , self.last_action_table )
        for pair in self.q_table:
            if not pair in self.last_action_table:
                continue
            reward = 0
            for i in range(len(self.last_action_table[pair])):

                n1 = self.env.algo.topo.nodes[pair[0]]
                n2 = self.env.algo.topo.nodes[pair[1]]
                (action , timeSlot) = self.last_action_table[pair][i]

                usedCount = 0
                used = False
                # if (n1,n2) in  self.env.algo.topo.needLinksDict:
                #     usedCount = self.env.algo.topo.needLinksDict[(n1, n2)].count(timeSlot)
                #     used = True
                # elif (n2, n1) in  self.env.algo.topo.needLinksDict:
                #     usedCount = self.env.algo.topo.needLinksDict[(n2, n1)].count(timeSlot)
                #     used = True
                
                # if used:
                #     if usedCount > 0:
                #         reward += 10 - (self.env.algo.timeSlot - timeSlot)
                #     else:
                #         reward += -(10 - (self.env.algo.timeSlot - timeSlot))

                reward = self.env.find_reward(pair , action , timeSlot)
                if not reward:
                    continue

                max_future_q = np.max(self.q_table[pair])
                current_q = self.q_table[pair][action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                # print('new q:', new_q , 'current_q:', current_q , 'max_future_q:', max_future_q , 'reward:' , reward)


                self.q_table[pair][action] = new_q

        for pair in self.last_action_table:
            # print(self.last_action_table[pair])
            self.last_action_table[pair] = list(filter(lambda x: self.env.algo.timeSlot -  x[1] < ENTANGLEMENT_LIFETIME , self.last_action_table[pair]))
        
        # if (self.env.algo.timeSlot + 1 ) % 100 == 0:
        #     with open(self.qFileName, 'wb') as f:
        #         pickle.dump(self.q_table, f)
        #         print('-------------::::::: q table saved :::::::-------------')

