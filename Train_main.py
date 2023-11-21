"""
@author: Elijah French
"""

#Import necessary packages and classes 

import os
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

OPMFILEDIRECTORY = os.getenv('SLURM_TMPDIR') 
OPMFILENAME = 'RESTART'

num_cores = 1
n_iters = [0 for id in range(num_cores)]

#Initialize the environments

E = [Environment(OPMFILEDIRECTORY = OPMFILEDIRECTORY , OPMFILENAME = OPMFILENAME, process_id = id, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps = 12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year') for id in range(num_cores)]

TRANSITIONDIRECTORY = os.getenv('SLURM_TMPDIR')

#Initialize a transition buffer

#D = [Buffer(TRANSITIONDIRECTORY = TRANSITIONDIRECTORY, reset_buffer  = False, save_frequency = 1) for _ in range(num_cores)]

#Function to iterate generators

def run_training(id, n_iter):
    """Run iterator
    """
    try:
        generator = Train(env = E[id])
        generator.iterate(num_cores, id, n_iter)
    except Exception as e:
        print(f"Exception in run_training: {e}")
        raise  

#Iterate

if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = list(range(num_cores))
        futures = [executor.submit(run_training, ids[i], n_iters[i]) for i in range(num_cores)]

        for future in concurrent.futures.as_completed(futures):
            future.result()




