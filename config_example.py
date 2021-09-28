# Configuration file for Semeter and Kamalabadi 2005 inversion code
# Written by Riley Troyer Fall 2021

# Write quantities to send to program in this dictionary
# isr_data_path is where the isr data files are stored
# isr_lat, isr_lon are geographic latitude and longitude of isr
# isr_min_alt is the minimum altitude to use isr data, below data is usually bad
# max_msis_alt is the altitude in meters to approximate infinity when calculating mass distance
# max_iterations defines the max number of iterations of the maximum entropy method
# test_flux is the number flux to use when constructing the A matrix
# altitude_bins is the altitude values in meters to calculate inversion for
# energy_bins is the energy values in electron-volts to calculate inversion for

import numpy as np
import os

d = {

    'isr_data_dir' : '../source-energy-pa/data/pfisr-data/mswinds/',
    'isr_lat' : 65.117,
    'isr_lon' : 212.540,
    'isr_min_alt' : 50,
    'max_msis_alt' : 1001e3,
    'max_iterations' : 1000,
    'test_flux' : 1e11,	
    'altitude_bins' : np.linspace(50e3, 140e3, 100),
    'energy_bins' : np.logspace(np.log10(300), np.log10(300000), 50),
    'start_file_i' : 12,
    'end_file_i' : 13

}

run_info = {'config_info' : d,
            'run_files' : sorted([f for f in 
                    os.listdir(d['isr_data_dir']) 
                    if f.endswith('.h5')])[d['start_file_i']:d['end_file_i']]
}
