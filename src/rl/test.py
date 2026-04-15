import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


from keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import random

max_ = 0
for i in range(10000):
    rand = int(random.random()*5+3) 
    print(rand)
    max_ = max(max_ , rand)
print(max_)