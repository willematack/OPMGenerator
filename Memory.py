"""
@author: Elijah French/Willem Atack
"""

import torch


class Buffer():

    def __init__(self, iteration_size, time_steps):

        self.mem_size = int(iteration_size*time_steps)
        self.mem_cntr = 0

        self.state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.new_state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.action_memory = torch.zeros(size = (self.mem_size, 2))

        self.mem_cntr_initial = self.mem_cntr

    def store_transition(self, state, action, new_state):
        '''Store a state, action, and new state to the buffer
        '''

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.detach()
        self.action_memory[index] = action.detach()
        self.new_state_memory[index] = new_state.detach()
        

        self.mem_cntr += 1
