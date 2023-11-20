"""
@author: Elijah French
"""

#Import necessary packages and classes

import numpy as np
import copy
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import wandb
import random
import time

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import OPM
import Memory 

from OPM import Environment
from Memory import Buffer

class TrainMP():

    ###Initialization

    def __init__(self, env: Environment, transitionmemory: Buffer):
        """Initialize the training class which involves an iteration method and a learning method
        """
        
        #Set parameters to be used in run
        self.env = env
        self.time_steps = self.env.time_steps
        self.max_pressure = self.env.max_pressure

        #Start wandb and keep track of changes in hypers
        #run = wandb.init(project='OPM_Generator')

        #Initialize the memory
        self.transitionmemory = transitionmemory

    ###Iterating methods
    
    def iterate(self, n_iter = 100, n_save = 10, n_plot  = 50, n_break = 50):
        """Run through OPM taking random actions and add states to memory 
        """
        #Track the network using wandb
        #wandb.watch((self.simulator), log = 'all', log_freq = 100)

        for i in tqdm(range(n_iter+1)):

            #Reset environment (reset file to be used by flow)
            self.env.reset()
            #Define initial state
            state = torch.stack((torch.ones((60,60))*3500/self.max_pressure, torch.zeros((60,60))))

            #Start running through an episode
            for step in range(0, self.env.time_steps+1):

                #Take a random action
                action = torch.rand(2)

                #Step in the environment and store the observed tuple
                state_ = self.env.step(action, state, step)
                self.transitionmemory.store_transition(state, action, state_)

                state = state_.clone()

        return self.transitionmemory.state_memory, self.transitionmemory.action_memory, self.transitionmemory.new_state_memory
           


    