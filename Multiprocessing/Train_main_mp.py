"""
@author: Elijah French
"""

#Import necessary packages and classes 

import datetime
from datetime import datetime
import concurrent.futures

import Memory
import Train
import OPM
import torch

from Memory import Buffer
from OPM import Environment
from Train import Train
from Train_MP import TrainMP

#Define the folder in which the base OPM file is stored in and the name of the file

OPMFILEDIRECTORY = '/mnt/c/Users/wille/Documents/Research/CarbonStorage/Generator/Restart'
OPMFILENAME = 'RESTART'

#Initialize the environment
num_cores = 2

E = [Environment(OPMFILEDIRECTORY = OPMFILEDIRECTORY , OPMFILENAME = OPMFILENAME, process_id = id, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps =12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year') for id in range(num_cores)]

#Define file path with which to save transitions in (note a .pt file must already be created in the folder) 

TRANSITIONDIRECTORY = '/mnt/c/Users/wille/Documents/Research/CarbonStorage/Generator/Transitions'

#Initialize a transition buffer

D = [Buffer(TRANSITIONDIRECTORY = TRANSITIONDIRECTORY, reset_buffer  = False, save_frequency = 1) for _ in range(num_cores)]

#Initialize the generator
def run_training(id, n_iter):
    try:
        generator = TrainMP(env=E[id], transitionmemory=D[id])
        return generator.iterate(n_iter, n_save=2)
    except Exception as e:
        print(f"Exception in run_training: {e}")
        raise  # Rethrow the exception to ensure it's caught by the main process
   

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ids = list(range(num_cores))
        n_iters = [2, 2]
        futures = [executor.submit(run_training, ids[i], n_iters[i]) for i in range(num_cores)]
        s = []
        a = []
        s_ = []
        for future in concurrent.futures.as_completed(futures):
            result1, result2, result3 = future.result()
            s.append(result1)
            a.append(result2)
            s_.append(result3)


torch.save(torch.stack(s), TRANSITIONDIRECTORY+'/stateMP.pt')
torch.save(torch.stack(a), TRANSITIONDIRECTORY+'/actionMP.pt')
torch.save(torch.stack(s_), TRANSITIONDIRECTORY+'/state_MP.pt')

