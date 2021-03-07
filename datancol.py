##
# data analysis of BFSCAN 
##
from __future__ import (division, print_function, absolute_import,unicode_literals)
import os,sys,glob
import happi
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
import scipy.constants as sc
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import lpa2 as l 


## get the list of config folder 

rootpath = '/ccc/scratch/cont003/smilei/cassouke/BF-TEST/'

files = list(filter(os.path.isdir, glob.glob(rootpath + "/*/")))
files.sort(key=lambda x: os.path.getmtime(x))

number_files = len(files)

print("Number of configuration : \t", number_files)

# for test 

number_files = 10

# initialization of arrays
Config          = np.zeros([number_files])
n_e_1           = np.zeros([number_files])
r               = np.zeros([number_files])
l_1             = np.zeros([number_files])
x_foc           = np.zeros([number_files])
c_N2            = np.zeros([number_files])
x_foc_vac       = np.zeros([number_files])
a0_max          = np.zeros([number_files])
x_a0_max        = np.zeros([number_files])
injection_flag  = np.zeros([number_files])
ti              = np.zeros([number_files])
xi              = np.zeros([number_files])
E_peak          = np.zeros([number_files])
E_fwhm          = np.zeros([number_files])
q_end           = np.zeros([number_files])
emittance_y     = np.zeros([number_files])
emittance_z     = np.zeros([number_files])
divergence_rms  = np.zeros([number_files])



## scanning the configuration
for f in range(number_files):
    # loading data 
    print("")
    print("--------------------------------------------")
    print("")
    print("loading data ...\n",
    str(files[f]),"\n")

    tmp = l.loadData(str(files[f]))
    # read configuration
    Config[f] = tmp.namelist.config_external['Config']
    n_e_1[f] = tmp.namelist.config_external['n_e_1']
    r[f]= tmp.namelist.config_external['r']
    l_1[f] = tmp.namelist.config_external['l_1']
    x_foc[f] = tmp.namelist.config_external['x_foc']
    c_N2[f] = tmp.namelist.config_external['c_N2']
    x_foc_vac[f] = tmp.namelist.xfocus
    # value and position of the max of a0 
    x,a = l.getMaxinMovingWindow(tmp)
    a0_max[f] = a.max()
    x_a0_max[f] = x[a.argmax()]

    # timesteps vector
    ts = l.getPartAvailableSteps(tmp)
    # injection timestep and position (m)
    ti[f],xi[f] = l.getInjectionTime(tmp,ts)
    if ti[f] == np.nan :
        injection_flag[f] = False
        print("#####################################\n",
        '#\t no injection \n',
        "#####################################")
        ti[f] = np.nan
        xi[f] = np.nan
        E_fwhm[f] = np.nan
        E_peak[f] = np.nan
        emittance_y[f] = np.nan
        emittance_z[f] = np.nan
        divergence_rms[f] = np.nan
        q_end[f] = np.nan
    else :
        injection_flag[f] = True
    
        # get electron spectrum at last timestep
        energy_axis, spectrum, E_peak[f], E_fwhm[f]  = l.getSpectrum(tmp,ts[-1],print_flag=True)
        
        # beam parameter filter around 
        if (E_peak[f] != np.nan) and (E_fwhm[f] != np.nan) :
            param_list = l.getBeamParam(tmp,ts[-1], E_min=(E_peak[f]-2*E_fwhm[f])/0.512, E_max = (E_peak[f]+3*E_fwhm[f])/0.512,print_flag=True)
            emittance_y[f] = param_list[5]
            emittance_z[f] = param_list[6]
            divergence_rms[f] = param_list[10]
            q_end[f] = param_list[5]
        elif (E_peak[f] != np.nan) and (E_fwhm[f] == np.nan) :
            param_list = l.getBeamParam(tmp,ts[-1], E_min=50, E_max = (2*E_peak[f])/0.512,print_flag=True)
            emittance_y[f] = param_list[5]
            emittance_z[f] = param_list[6]
            divergence_rms[f] = param_list[10]
            q_end[f] = param_list[5]
        else :
            param_list = l.getBeamParam(tmp,ts[-1], E_min=50, E_max = 500,print_flag=True)
            emittance_y[f] = param_list[5]
            emittance_z[f] = param_list[6]
            divergence_rms[f] = param_list[10]
            q_end[f] = param_list[5]


dict_data = {'Config':Config,'n_e_1':n_e_1, 'r':r, 'l_1':l_1,'x_foc':x_foc,'c_N2':c_N2,'x_foc_vac':x_foc_vac, 'a0_max':a0_max,'x_a0_max':x_a0_max,
'injection':injection_flag,'t_i': ti,'x_i':xi,'E_peak':E_peak,'E_fwhm':E_fwhm,
'q_end':q_end,'emit_y':emittance_y,'emit_z':emittance_z,'div_rms':divergence_rms}

df = pd.DataFrame(dict_data)

df = df[['Config','n_e_1', 'r','l_1','x_foc','c_N2','x_foc_vac', 'a0_max','x_a0_max',
'injection','t_i','x_i','E_peak','E_fwhm',
'q_end','emit_y','emit_z','div_rms']]

# saving dataframe to csv
df.to_csv('dataframe_bfscan.csv')

print('')
