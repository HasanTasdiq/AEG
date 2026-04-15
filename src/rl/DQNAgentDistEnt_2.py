import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras import Input
from collections import deque
import time
import random
import os
from RoutingEnv import RoutingEnv      #for ubuntu
# from .RoutingEnv import RoutingEnv   #for mac
import multiprocessing
import math
NUM_EPISODES = 2500
LEARNING_RATE = 0.1


GAMMA = 0.99

ENTANGLEMENT_LIFETIME = 10
# Exploration settings

EPSILON_ = 0.5  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 100
EPSILON_DECAY_VALUE = EPSILON_/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 5000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 50  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20



#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class DQNAgentDistEnt_2:
    def __init__(self ,algo , pid):
        print('++++++++++initiating DQN agent for:' , algo.name)
        self.env = RoutingEnv(algo)
        self.OBSERVATION_SPACE_VALUES = (self.env.SIZE *2 +2,self.env.SIZE,)  
        self.model_name = algo.name+'_'+str(len(algo.topo.nodes)) +'_'+str(algo.topo.alpha) +'_'+str(algo.topo.q) +'_'+'DQNAgentDistEnt_2.keras'

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.last_action_table = {}
        self.link_qs = {}



    def create_model(self):
        try:
            model = load_model(self.model_name)
            print('=====================================================model loaded from ' , self.model_name , ' =====================================')
            # time.sleep(10)
            return model
        except:
            print('=====================no model found========================')
            # time.sleep(10)

        
        model = Sequential()



        model.add(Flatten(input_shape = self.OBSERVATION_SPACE_VALUES))  
        model.add(Dense(72 , activation='relu'))
        model.add(Dense(48 , activation='relu'))
        model.add(Dense(24 , activation='relu'))

        model.add(Dense(2, activation='linear')) 
        print(model.summary)

        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(), metrics=['accuracy'])
        # model._make_predict_function()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        # print('----------len(self.replay_memory)----------------')
        # print(len(self.replay_memory))
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        
        print([(m[1] , m[2]) for m in minibatch])

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states , verbose=0, batch_size=64)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states , verbose=0, batch_size=64)

        X = []
        y = []
        # print(len(minibatch))

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # print('---action-- ' , action)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # print('=============train start===========')
        t1 = time.time()
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, )
        print('=============train done===========' , time.time() - t1)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            print('------------------self.model.get_weights()-------------------')
            print(self.model.get_weights())
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)

    def get_qs_batch(self, states):
        
        sts = list()
        c = 0
        for link , state in states:
            # sts.append(np.array(state).reshape(-1, *state.shape))
            sts.append(state)
            # c+= 1
            # if c > 100:
            #     break
        # print(sts)
        
        # print('---------- len sts --------------' , len(sts))
        return self.model.predict(np.array(sts), verbose=0, batch_size=50)
    
    def get_link_qs_batch(self , links , timeSlot):
        # print('in get p q')
        if not len(links):
            return
        states = []
        for link in links:
            current_state = self.env.ent_state(link , timeSlot)
            # print(current_state)
            states.append((link , current_state))

        # print('getting qs')
        # print(states)
        qs = self.get_qs_batch(states)
        # print('getting qs done! '  , len(qs))
        for i in range(len(states)):
            link , current_state = states[i]
            self.link_qs[link] = (current_state , qs[i])
    
    # def update_last_action_table(self , action , timeSlot , current_state , next_state):


    # def learn_and_predict(self):
    #     global EPSILON_
    #     t1 = time.time()
    #     edges = self.env.algo.topo.edges
    #     timeSlot = self.env.algo.timeSlot
    #     link_action_q = []


    #     # -----sequntial prediction----

       

    #     # for pair in state:
    #     for link in edges:
    #         self.get_link_qs_batch([link] , timeSlot)
    #         current_state , qs = self.link_qs[link]
    #         if np.random.random() > EPSILON_:
    #             # Get action from Q net
    #             # print('------self.get_qs(current_state)-----')
    #             # print(self.get_qs(current_state))
    #             t2 = time.time()

    #             action = np.argmax(qs)
    #             q = qs[action]
    #             # print('--- get_qs-- ' , time.time() - t2 , 'seconds')

    #         else:
    #             # Get random action
    #             action = np.random.randint(0, 2)
    #             q = qs[action]

    #         next_state  = self.env.assignQubitEdge(link , action , timeSlot)
    #         if next_state is None:
    #             next_state = current_state
            
    #         if not link in self.last_action_table:
    #             self.last_action_table[link] = [(action , timeSlot , current_state , next_state)]
    #         else:
    #             self.last_action_table[link].append((action , timeSlot , current_state , next_state))
        

    #     if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
    #         EPSILON_ -= EPSILON_DECAY_VALUE

    #     self.link_qs = {}
    #     print('**learn_and_predict multicore ent dqrl done in ' , time.time() - t1 , 'seconds')

    def learn_and_predict(self):
        assignable = True
        while assignable:
            assignable = self.learn_and_predict2()
            if 'no_repeat' in self.env.algo.name:
                break
    def learn_and_predict2(self):
        global EPSILON_
        t1 = time.time()
        # links = self.env.algo.topo.links
        links = self.env.algo.topo.edges
        timeSlot = self.env.algo.timeSlot
        link_action_q = []
        links_len = len(links)
        assignable = False




        self.get_link_qs_batch(links , timeSlot)

        # for pair in state:
        for link in self.link_qs:
            current_state , qs = self.link_qs[link]
            if np.random.random() > EPSILON_:
                # Get action from Q net
                # print('------self.get_qs(current_state)-----')
                # print(self.get_qs(current_state))
                t2 = time.time()

                action = np.argmax(qs)
                q = qs[action]
                # print('--- get_qs-- ' , time.time() - t2 , 'seconds')

            else:
                # Get random action
                action = np.random.randint(0, 2)
                q = qs[action]

            link_action_q.append((link , action , q , current_state))
        
        # if np.random.random() > EPSILON_:
        #     link_action_q.sort(key=lambda x: x[2], reverse=True)
        link_action_q.sort(key=lambda x: x[2], reverse=True)
     

        for (link ,action , q , current_state) in link_action_q:

            # print('---action in learn rand -- ' , action)
            # next_state = self.env.assignQubit(link , action , timeSlot)
            next_state, assignable2 = self.env.assignQubitEdge_2(link , action , timeSlot)
            if next_state is None:
                next_state = current_state
            if assignable2:
                assignable = True
            if not link in self.last_action_table:
                self.last_action_table[link] = [(action , timeSlot , current_state , next_state)]
            else:
                self.last_action_table[link].append((action , timeSlot , current_state , next_state))

        if END_EPSILON_DECAYING >= timeSlot >= START_EPSILON_DECAYING:
            EPSILON_ -= EPSILON_DECAY_VALUE

        self.link_qs = {}
        print('**learn_and_predict multicore ent dqrl done in ' , time.time() - t1 , 'seconds')
        return assignable
    def update_reward(self):
        print('update reward:::::::::::::::::::::::: ' , len(self.last_action_table) )
        t1 = time.time()
        for link in self.last_action_table:

            reward = 0
            for i in range(len(self.last_action_table[link])):

                (action , timeSlot , current_state , next_state) = self.last_action_table[link][i]
                reward = self.env.find_reward_ent(link  , timeSlot , action)
                if not reward:
                    continue

                self.update_replay_memory((current_state, action, reward, next_state, False))
        self.env.algo.topo.reward_ent = {}
        self.train(False , self.env.algo.timeSlot)



        for pair in self.last_action_table:
            # print(self.last_action_table[pair])
            self.last_action_table[pair] = list(filter(lambda x: self.env.algo.timeSlot -  x[1] < ENTANGLEMENT_LIFETIME , self.last_action_table[pair]))
        
        print('update_reward done in ' , time.time() - t1 , 'seconds\n')
    
    def save_model(self):
        self.model.save((self.model_name))
