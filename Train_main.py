"""
@author: Elijah French
"""

#Import necessary packages and classes 

import os
import torch
import datetime
from datetime import datetime
import concurrent.futures
import multiprocessing as mp

import Train
import OPM
import Memory

from Memory import Buffer
from OPM import Environment
from Train import Train

#Define the folder in which the base OPM file is stored in and the name of the file

TEMPDIRECTORY = os.getenv('SLURM_TMPDIR') 
OPMFILENAME = 'RESTART'

num_cores = 32
n_iters = [500 for id in range(num_cores)]

#Initialize the environments

E = [Environment(OPMFILEDIRECTORY = TEMPDIRECTORY , OPMFILENAME = OPMFILENAME, process_id = id, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps = 12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year') for id in range(num_cores)]

#Define the folder where transitions are stored

TRANSITIONDIRECTORY = TEMPDIRECTORY + '/Dec_13'

#Create the folder to save transitions to

os.mkdir(TRANSITIONDIRECTORY)

#Initialize a transition buffer

D = [Buffer(n_iters[id], E[id].time_steps) for id in range(num_cores)]

#Function to iterate generators

def run_training(id, n_iter):
    """Run iterator
    """
    try:
        generator = Train(env = E[id], transitionmemory = D[id])
        return generator.iterate(num_cores, id, n_iter)
    
    except Exception as e:
        print(f"Exception in run_training: {e}")
        raise  

if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = list(range(num_cores))
        futures = [executor.submit(run_training, ids[id], n_iters[id]) for id in range(num_cores)]
        s = torch.zeros(1, 2, 60, 60)
        a = torch.zeros(1, 2)
        s_ = torch.zeros(1, 2, 60, 60)
        for future in concurrent.futures.as_completed(futures):
            states, actions, states_ = future.result()
            s = torch.cat((s, states), 0)
            a = torch.cat((a, actions), 0)
            s_ = torch.cat((s_, states_), 0)
    
    torch.save(s[1:,:,:,:], TRANSITIONDIRECTORY + '/state.pt')
    torch.save(a[1:,:], TRANSITIONDIRECTORY + '/action.pt')
    torch.save(s_[1:,:,:,:], TRANSITIONDIRECTORY + '/state_.pt')

    print(s[1:,:,:,:].size())
