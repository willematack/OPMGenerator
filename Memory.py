"""
@author: Elijah French
"""

import torch

class Buffer():

    def __init__(self, TRANSITIONDIRECTORY: str, max_size = 10_000, input_shape = 3, save_frequency = 500, reset_buffer = False):
        
        self.mem_size = int(max_size)
        self.file_path = TRANSITIONDIRECTORY
        self.save_frequency  = save_frequency

        self.mem_cntr = 0
        self.state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.new_state_memory = torch.zeros(size = (self.mem_size, 2, 60, 60))
        self.action_memory = torch.zeros(size = (self.mem_size, 2))
        torch.save(torch.tensor(self.mem_cntr), self.file_path + '/mem_cntr.pt')
        torch.save(self.state_memory.detach(), self.file_path + '/state.pt')
        torch.save(self.new_state_memory.detach(), self.file_path + '/state_.pt')
        torch.save(self.action_memory.detach(), self.file_path + '/action.pt')

        self.mem_cntr_initial = self.mem_cntr

    def store_transition(self, state, action, new_state):
        '''Store a state, action, and new state to the buffer
        '''

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.detach()
        self.new_state_memory[index] = new_state.detach()
        self.action_memory[index] = action.detach()

        self.mem_cntr += 1
    
    def save_buffer(self):
        '''Save the current buffer 
        '''

        torch.save(torch.tensor(self.mem_cntr), self.file_path + '/mem_cntr.pt')
        torch.save(self.state_memory.detach(), self.file_path + '/state.pt')
        torch.save(self.new_state_memory.detach(), self.file_path + '/state_.pt')
        torch.save(self.action_memory.detach(), self.file_path + '/action.pt')
        