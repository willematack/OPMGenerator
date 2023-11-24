"""
@author: Elijah French
"""

import torch


class Buffer():

    def __init__(self, TRANSITIONDIRECTORY: str, max_size = 500, save_frequency = 100):

        self.mem_size = int(max_size)
        self.file_path = TRANSITIONDIRECTORY
        self.save_frequency  = save_frequency

        self.mem_cntr = 0
        self.state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.new_state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.action_memory = torch.zeros(size = (self.mem_size, 2))

        self.mem_cntr_initial = self.mem_cntr

    def initiate_storage(self):
        '''Reset the transition tuples (necessary at the beginning of generation)
        '''
        
        torch.save(torch.zeros(size = (1,2,60,60)), self.file_path+'/state.pt')
        torch.save(torch.zeros(size = (1,2,60,60)), self.file_path+'/state_.pt')
        torch.save(torch.zeros(size = (1,2)), self.file_path+'/state.pt')

    def store_transition(self, state, action, new_state):
        '''Store a state, action, and new state to the buffer
        '''

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.detach()
        self.new_state_memory[index] = new_state.detach()
        self.action_memory[index] = action.detach()

        self.mem_cntr += 1