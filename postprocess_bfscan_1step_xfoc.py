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
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpa2 as l 

# Parameters to get electron spectrum at last timestep
Emin = 50.0   #   np.max((50),(E_peak[f]-2*E_fwhm[f])/0.512))     # me c^2 unit
Emax = 1000.0  #  (E_peak[f]+2*E_fwhm[f])/0.512@                  # me c^2 unit
nbins = 200 # number of bins for the histogram default value is 200. 

## get the list of config folder 

rootpath = '/ccc/scratch/cont003/smilei/cassouke/BF-TEST/'

files = list(filter(os.path.isdir, glob.glob(rootpath + "/*/")))
files.sort(key=lambda x: os.path.getmtime(x))

number_files = len(files)

print("Number of configuration : \t", number_files)

timeStep = -1 

# for test, comment for postprocessing

number_files = 10

# initialization of arrays 400 is the binning in the histogram energy 

Config          = np.zeros([number_files])
a_0             = np.zeros([number_files])
x_foc           = np.zeros([number_files])
c_N2            = np.zeros([number_files])
x_foc_vac       = np.zeros([number_files])
x_p             = np.zeros([number_files,1001]) #
n_e_p           = np.zeros([number_files,1001]) # may change , scanning size. 
injection_flag  = np.zeros([number_files])
indi            = np.zeros([number_files])
ti              = np.zeros([number_files])
xi              = np.zeros([number_files])
zeros_vector    = np.zeros([number_files]) 

a0_max          = np.zeros([number_files])
x_a0_max        = np.zeros([number_files])
E_peak          = np.zeros([number_files])
dQdE_max        = np.zeros([number_files])
E_fwhm          = np.zeros([number_files])
E_mean          = np.zeros([number_files])
E_med           = np.zeros([number_files])
E_std           = np.zeros([number_files])
E_wstd          = np.zeros([number_files])
E_mad           = np.zeros([number_files])
spectrum        = np.zeros([number_files,nbins])
energy_axis     = np.zeros([number_files,nbins])
q_end           = np.zeros([number_files])
emittance_y     = np.zeros([number_files])
emittance_z     = np.zeros([number_files])
divergence_rms  = np.zeros([number_files])


print("")
print("--------------------------------------------")
print("")
print("Post-processing timeStep : ",timeStep)
print("")
print("--------------------------------------------")
print("")


## scanning the configuration
for f in range(number_files):
    # loading data 
    print("")
    print("--------------------------------------------")
    print("")
    print("loading data ...\n",str(files[f]),"\n")

    tmp = l.loadData(str(files[f]))

    # read configuration
    Config[f] = tmp.namelist.config_external['Config']
    a_0[f] = tmp.namelist.config_external['a_0']
    x_foc[f] = tmp.namelist.config_external['x_foc']
    c_N2[f] = tmp.namelist.config_external['c_N2']
    x_foc_vac[f] = tmp.namelist.xfocus
    
    # plasma profile
    x_p[f] = tmp.namelist.x_h_points
    n_e_p[f] = tmp.namelist.x_h_values

    # timesteps vector
    ts = l.getPartAvailableSteps(tmp)
    # injection timestep and position (m)
    ind,ti[f],xi[f] = l.getInjectionTime(tmp,ts)
    indi[f] = int(ind)

    if (indi[f] == len(ts)-1) | np.isnan(ti[f]):
        injection_flag[f] = False
        print(" ###################################################\n",
        '#\t no injection \n',
        "###################################################")
        ti[f] = np.nan
        xi[f] = np.nan
        E_mean[f] = np.nan
        E_std[f] = np.nan
        E_fwhm[f] = np.nan
        E_peak[f] = np.nan
        E_wstd[f] = np.nan
        E_med[f] = np.nan
        E_mad[f]  = np.nan
        dQdE_max[f]   = np.nan
        emittance_y[f] = np.nan
        emittance_z[f] = np.nan
        spectrum[f]    = np.nan
        divergence_rms[f] = np.nan
        q_end[f] = np.nan


    else :
        injection_flag[f] = True
        print(" ###################################################\n",
        '#\t  injection occured at:\t',ti[f],' \n',
        "###################################################")

        #  only the given the timestep value 
        # laser self-focusing
        x,a = l.getMaxinMovingWindow(tmp)
        a0_max[f] = a.max()
        x_a0_max[f] = x[a.argmax()]

        # energy distribution 
        energy_axis[f], spectrum[f], E_peak[f], dQdE_max[f], E_fwhm[f]  = l.getSpectrum(tmp,ts[timeStep], E_min=Emin, E_max = Emax, print_flag=False)

        # beam parameter filter around 
        if (E_peak[f] == 0) or (E_fwhm[f] == 0) :
            param_list = l.getBeamParam(tmp,ts[-1], E_min=Emin, E_max = Emax,print_flag=False)
            E_mean[f] = param_list['energy_wmean']
            E_med[f] = param_list['energy_wmedian']
            E_wstd[f] = param_list['energy_wrms']
            E_std[f] = param_list['energy_rms']
            E_mad[f] = param_list['energy_wmad']
            E_peak[f] = np.nan
            E_fwhm[f] = np.nan
            dQdE_max[f] = spectrum.max()
            emittance_y[f] = param_list['emittancey']
            emittance_z[f] = param_list['emittancez']
            divergence_rms[f] = param_list['divergence_rms']
            q_end[f] = param_list['charge']

        else :

            param_list = l.getBeamParam(tmp,ts[-1], E_min=Emin, E_max = Emax ,print_flag=False)
            E_mean[f] = param_list['energy_wmean']
            E_med[f] = param_list['energy_wmedian']
            E_wstd[f] = param_list['energy_wrms']
            E_std[f] = param_list['energy_rms']
            E_mad[f] = param_list['energy_wmad']
            emittance_y[f] = param_list['emittancey']
            emittance_z[f] = param_list['emittancez']
            divergence_rms[f] = param_list['divergence_rms']
            q_end[f] = param_list['charge']


# saving dataframe to changing 2D ndarray to list of array to avoid trouble opening the dataframe 

dict_data = {'Config':Config,'x_foc':x_foc,'c_N2':c_N2,'a_0':a_0, 'x_foc_vac':x_foc_vac,
'a0_max':a0_max,'x_a0_max':x_a0_max,'injection':injection_flag,'t_i': ti,'x_i':xi,'E_mean':E_mean,'E_med':E_med,'E_std':E_std,'E_wstd':E_wstd, 'E_mad':E_mad,
'E_peak':E_peak,'E_fwhm':E_fwhm,'dQdE_max':dQdE_max,'q_end':q_end,'emit_y':emittance_y,'emit_z':emittance_z,'div_rms':divergence_rms,
'ener_axis':zeros_vector,'spec':zeros_vector,'x_p':zeros_vector,'n_e_p':zeros_vector}

df = pd.DataFrame(dict_data)

df = df[['Config','n_e_1', 'r','l_1','x_foc','c_N2','x_p','n_e_p','x_foc_vac', 'a0_max','x_a0_max',
'injection','t_i','x_i','E_mean','E_med','E_std','E_mad','E_peak','E_fwhm','dQdE_max',
'q_end','emit_y','emit_z','div_rms','ener_axis','spec']]
tmp_e = []
tmp_s = []
tmp_x = []
tmp_ne = []

for f in range(number_files):
    tmp_e.append(energy_axis[f])
    tmp_s.append(spectrum[f])
    tmp_x.append(x_p[f])
    tmp_ne.append(n_e_p[f])

df['ener_axis'] = tmp_e
df['spec'] = tmp_s
df['x_p'] = tmp_x
df['n_e_p'] = tmp_ne


# saving dataframe to pickle
df.to_pickle('dataframe_bfscan.pickle')

print('Post processing Ended')