"""
@author: Elijah French
"""

#Import necessary packages and classes 

import os
import datetime
from datetime import datetime
import concurrent.futures

import Train
import OPM

from OPM import Environment
from Train import Train

#Define the folder in which the base OPM file is stored in and the name of the file

OPMFILEDIRECTORY = os.getenv('SLURM_TMPDIR') 
OPMFILENAME = 'RESTART'

num_cores = 2

#Initialize the environments

E = [Environment(OPMFILEDIRECTORY = OPMFILEDIRECTORY , OPMFILENAME = OPMFILENAME, process_id = id, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps = 12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year') for id in range(num_cores)]

#Initialize the generators
def run_training(id, n_iter):
    try:
        generator = Train(env=E[id])
        generator.iterate(id, n_iter)
    except Exception as e:
        print(f"Exception in run_training: {e}")
        raise  # Rethrow the exception to ensure it's caught by the main process

#Iterate

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ids = list(range(num_cores))
        n_iters = [2, 2]
        futures = [executor.submit(run_training, ids[i], n_iters[i]) for i in range(num_cores)]




