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
OPMFILEDIRECTORY = '/mnt/c/elijah/OPMGenerator/Generator/RESTART'
OPMFILENAME = 'RESTART'

#Initialize the environment

E = Environment(OPMFILEDIRECTORY = OPMFILEDIRECTORY , OPMFILENAME = OPMFILENAME, OPMtextoutput = False, startdate = datetime(2023, 1, 1), 
            time_steps = 12, min_injection = 10, max_injection = 30_000, min_production = 10, max_production =  6000, 
            max_pressure_OPM = 10_000, dt = '2year')

#Initialize the generator

generator = Train(env = E)

generator.iterate(n_iter = 2)





