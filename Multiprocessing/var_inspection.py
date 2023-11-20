import torch
import numpy as np

TRANSITIONDIRECTORY = '/mnt/c/Users/wille/Documents/Research/CarbonStorage/Generator/Transitions'

actions = torch.load(TRANSITIONDIRECTORY + '/actionMP.pt')
mem_c = torch.load(TRANSITIONDIRECTORY + '/mem_cntr.pt')
states = torch.load(TRANSITIONDIRECTORY + '/stateMP.pt')
states_ = torch.load(TRANSITIONDIRECTORY + '/state_MP.pt')

print(actions.shape)
