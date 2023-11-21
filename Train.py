"""
@author: Elijah French
"""

#Import necessary packages and classes

import numpy as np
import copy
import torch
import random
import time
import datetime
from datetime import datetime

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import OPM
from OPM import Environment

class Train():

    ###Initialization

    def __init__(self, env: Environment):
        """Initialize the training class which involves an iteration method and a learning method
        """
        #Set parameters to be used in run
        self.env = env
        self.time_steps = self.env.time_steps
        self.max_pressure = self.env.max_pressure

    ###Iterating methods
    
    def iterate(self, num_cores, id, n_iter = 100):
        """Run through OPM taking random actions and add states to memory 
        """

        for i in range(n_iter+1):

            #Reset environment (reset file to be used by flow)
            self.env.reset()

            #Define initial state
            state = torch.stack((torch.ones((60,60))*3500/self.max_pressure, torch.zeros((60,60))))

            #Start running through an episode
            for step in range(0, self.env.time_steps+1):

                #Take a random action
                action = torch.rand(2*num_cores)

                #Step in the encironment and store the observed tuple
                state_ = self.env.step(action[2*id:2*id+2], step)
                print("Time: " + datetime.now().strftime("%H:%M:%S") + " Core: " + str(id) + " Iteration: " + str(i) + " Step: " + str(step) + " Action: " + str(np.array(action[2*id:2*id+2])) + " State: " + str(torch.mean(state_[1,:,:]).item()))

                state = state_.clone()




    