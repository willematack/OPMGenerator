"""
@author: Elijah French
"""

#Import necessary packages and classes 

import datetime
from datetime import datetime

import Memory
import Train
import OPM

from Memory import Buffer
from OPM import Environment
from Train import Train

#Define the folder in which the base OPM file is stored in and the name of the file

#OPMFILEDIRECTORY = '/home/elijahf/computecan/Generator/'
OPMFILEDIRECTORY = '/mnt/c/elijah/Carbon_Storage_RL/Generator/RESTART'
OPMFILENAME = 'RESTART'

#Initialize the environment

E = Environment(OPMFILEDIRECTORY = OPMFILEDIRECTORY , OPMFILENAME = OPMFILENAME, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps = 12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year')

#Define file path with which to save transitions in (note a .pt file must already be created in the folder) 

#TRANSITIONDIRECTORY = '/home/elijahf/computecan/Generator/'
TRANSITIONDIRECTORY = '/mnt/c/elijah/Carbon_Storage_RL/Generator/Transitions'

#Initialize a transition buffer

D = Buffer(TRANSITIONDIRECTORY = TRANSITIONDIRECTORY, reset_buffer  = False, save_frequency = 50)

#Initialize the generator

generator = Train(env = E, transitionmemory = D)

#Create and save transitions

generator.iterate(n_iter = 1000, n_save = 50)





