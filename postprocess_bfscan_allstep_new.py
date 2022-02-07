##
## data analysis of BFSCAN 
##

from __future__ import (division, print_function, absolute_import,unicode_literals)
import os,sys,glob
sys.path.append('/users/flc/cassou/src/LPAbrew/')
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
import scipy.constants as sc
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import lpa2 as l 
from tqdm import tqdm

## parameters for postprocessing

nbins = 200     # bining of energy spectrum 
Emin = 25       # np.max((50),(E_peak[f]-2*E_fwhm[f])/0.512))     # me c^2 unit
Emax = 1000     # (E_peak[f]+2*E_fwhm[f])/0.512@                  # me c^2 unit 

## get the list of config folder 

#rootpath = '/ccc/scratch/cont003/smilei/drobniap/BF_TEST_CN2-05_a0-1.15/'
rootpath = '/silver/PALLAS/simulations/smilei/BF-TEST-CN2-03/'

files = list(filter(os.path.isdir, glob.glob(rootpath + "/*/")))
files.sort(key=lambda x: os.path.getmtime(x))

number_files = len(files)

print("Number of configuration : \t", number_files)

# for test, comment for postprocessing

start_file = 0

number_files = 10

# initialization of arrays 400 is the binning in the histogram energy 

Config          = np.zeros([number_files])
a_0             = np.zeros([number_files])
x_foc           = np.zeros([number_files])
c_N2            = np.zeros([number_files])
x_foc_vac       = np.zeros([number_files])
x_p             = np.zeros([number_files,1001]) #
n_e_p           = np.zeros([number_files,1001]) # may change , scanning size.
r               = np.zeros([number_files])
n_e_1           = np.zeros([number_files]) 
l_1             = np.zeros([number_files])    
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
q               = np.zeros([number_files])
emittance_y     = np.zeros([number_files])
emittance_z     = np.zeros([number_files])
divergence_rms  = np.zeros([number_files])

print("")
print("----------------------------------------------")
print("")
print("Post-processing :  all timeStep from injection")
print("")
print("----------------------------------------------")
print("")
a0              = []
x_a             = []
E_peak          = []
dQdE_max        = []
E_fwhm          = []
E_mean          = []
E_med           = []
E_mad           = []
E_std           = []
E_wstd          = []
spectrum        = []
q               = []
size_x          = []
emittance_y     = []
emittance_z     = []
divergence_rms  = []


## scanning the configuration
for f in range(number_files):
    # loading data 
    fnumber =  start_file + f
    print("")
    print("--------------------------------------------")
    print("")
    print("loading data ...\n",str(files[fnumber]),"\n")

    tmp = l.loadData(str(files[fnumber]))
    # read configuration
    Config[f] = tmp.namelist.config_external['Config']
    x_foc[f] = tmp.namelist.config_external['x_foc']
    c_N2[f] = tmp.namelist.config_external['c_N2']
    r[f] = tmp.namelist.config_external['r']
    l_1[f] = tmp.namelist.config_external['n_e_1']
    n_e_1[f] = tmp.namelist.config_external['n_e_1']
    x_foc_vac[f] = tmp.namelist.xfocus
    
    # plasma profile
    x_p[f] = tmp.namelist.x_h_points
    n_e_p[f] = tmp.namelist.x_h_values

    # timesteps vector
    ts = l.getPartAvailableSteps(tmp)

    # injection timestep and position (m)
    ind,ti[f],xi[f] = l.getInjectionTime(tmp,ts)
    indi[f] = int(ind)

    # laser properties in plasma 
    vec_a0              = np.zeros([vec_len])
    vec_x_a             = np.zeros([vec_len])

    for t in range(len(ts)):
            # value and position of the max of a0 
            vec_a0[t] = l.getLasera0(tmp,ts[t])
            vec_x_a[t] = ts[t]*tmp.namelist.onel
    
    a0.append(vec_a0)
    x_a.append(vec_x_a)
    a0_max[f] = np.max(vec_a0)
    x_a0_max[f] = vec_x_a[vec_a0.argmax()]

   
    #try: # SLK: continue postprocessing even in case of errors:
    #@@@@@@@@@@@@@@@@@@@@@ injection @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if ti[f] != None:
        injection_flag[f] = True

        # timestep from ionization
        tsi = ts[int(indi[f]+2):-1]
        vec_len = tsi.shape[0]

        print("#####################################\n",
        '#\t  injection occured at:\t',ti[f],' \n',
        "#####################################")
        print(' DEUBG shape :', vec_len)

        vec_E_peak          = np.zeros([vec_len])
        vec_dQdE_max        = np.zeros([vec_len])
        vec_E_fwhm          = np.zeros([vec_len])
        vec_E_mean          = np.zeros([vec_len])
        vec_E_med           = np.zeros([vec_len])
        vec_E_std           = np.zeros([vec_len])
        vec_E_mad           = np.zeros([vec_len])
        vec_spectrum        = np.zeros([vec_len,nbins])
        vec_q               = np.zeros([vec_len])
        vec_size_x          = np.zeros([vec_len])
        vec_emittance_y     = np.zeros([vec_len])
        vec_emittance_z     = np.zeros([vec_len])
        vec_divergence_rms  = np.zeros([vec_len])

        for t in range(len(tsi)):
            print('file:\t',f,' \t timestep:\t',t)
            
            # energy distribution characteristics 
            energy_axis[f], vec_spectrum[t], vec_E_peak[t], vec_dQdE_max[t], vec_E_fwhm[t]  = l.getSpectrum(tmp,tsi[t], E_min=Emin, E_max = Emax, print_flag=False)

            # beam parameter filter around energy peak 
            if (vec_E_peak[t] == 0) or (vec_E_fwhm[t] == 0) :
                try:
                    param_dict = l.getBeamParam(tmp,tsi[t], E_min=Emin, E_max = Emax,print_flag=False)
                    #print('DEBUG :\n',param_list)
                    vec_E_mean[t] = param_dict['energy_wmean']
                    vec_E_med[t] = param_list['energy_wmedian']
                    vec_E_std[t] = param_list['energy_rms']
                    vec_E_mad[t] = param_list['energy_wmad']
                    vec_E_peak[t] = np.nan
                    vec_E_fwhm[t] = np.nan
                    vec_dQdE_max[t] = vec_spectrum.max()
                    vec_emittance_y[t] = param_list['emittance_y']
                    vec_emittance_z[t] = param_list['emittance_z']
                    vec_size_x[t] =  param_list['size_x_rms']
                    vec_divergence_rms[t] = param_list['divergence_rms']
                    vec_q[t] = param_list['charge']
                except TypeError :
                    vec_E_mean[t] = np.nan
                    vec_E_med[t] = np.nan
                    vec_E_std[t] = np.nan
                    vec_E_mad[t] = np.nan
                    vec_E_peak[t] = np.nan
                    vec_E_fwhm[t] = np.nan
                    vec_E_std[t] = np.nan
                    vec_dQdE_max[t] = np.nan
                    vec_size_x[t] =  np.nan
                    vec_emittance_y[t] = np.nan
                    vec_emittance_z[t] = np.nan
                    vec_divergence_rms[t] = np.nan
                    vec_q[t] = np.nan
            else :
                #Emin =  np.max((50),(E_peak[f]-2*E_fwhm[f])/0.512))     # me c^2 unit
                #Emax = (E_peak[f]+2*E_fwhm[f])/0.512                  # me c^2 unit
                param_list = l.getBeamParam(tmp,tsi[t], E_min=Emin, E_max = Emax ,print_flag=False)
                #print('DEBUG :\n',param_list)
                vec_E_mean[t] = param_dict['energy_wmean']
                vec_E_med[t] = param_list['energy_wmedian']
                vec_E_std[t] = param_list['energy_rms']
                vec_E_mad[t] = param_list['energy_wmad']
                vec_dQdE_max[t] = vec_spectrum.max()
                vec_emittance_y[t] = param_list['emittance_y']
                vec_emittance_z[t] = param_list['emittance_z']
                vec_size_x[t] =  param_list['size_x_rms']
                vec_divergence_rms[t] = param_list['divergence_rms']
                vec_q[t] = param_list['charge']

        
        E_peak.append(vec_dQdE_max)
        dQdE_max.append(vec_dQdE_max)
        E_fwhm.append(vec_E_fwhm)
        E_mean.append(vec_E_mean)
        E_med.append(vec_E_mean)
        E_std.append(vec_E_std)
        E_mad.append(vec_E_mad)
        spectrum.append(vec_spectrum)
        q.append(vec_q)
        emittance_y.append(vec_emittance_y)
        emittance_z.append(vec_emittance_z)
        divergence_rms.append(vec_divergence_rms)
    
        #print('################# DEBUG ####################')
        #print("\t q[",f,"]=",q[f][0:10],"\n")
        print("len(q)",len(q))

    #@@@@@@@@@@@@@@@@@@@@@ no injection @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    else :
        injection_flag[f] = False
        print("#####################################\n",
        '#\t no injection \n',
        "#####################################")
        vec_len = ts.shape[0]
        print(' DEUBG shape :', vec_len)

        # all other valuees are filled with NaN
        ti[f] = np.nan
        xi[f] = np.nan
        energy_axis[f] = np.nan
        spectrum.append(np.nan)
        E_peak.append(np.nan)
        dQdE_max.append(np.nan)
        E_mean.append(np.nan)
        E_med.append(np.nan)
        E_mad.append(np.nan)
        E_std.append(np.nan)
        E_fwhm.append(np.nan)
        emittance_y.append(np.nan)
        emittance_z.append(np.nan)
        divergence_rms.append(np.nan)
        q.append(np.nan)

    #except : # SLK:  in case of error fill values with nans and continue the postprocessing
    injection_flag[f] = False
    print("#####################################\n",
    '#\t bad configuration, error occured in postprocessing \n',
    "#####################################")
        
    #    pass





# saving dataframe to changing 2D ndarray to list of array to avoid trouble opening the dataframe 

dict_data = {'Config':Config,'n_e_1':n_e_1, 'r':r, 'l_1':l_1,'x_foc':x_foc,'c_N2':c_N2,'x_foc_vac':x_foc_vac,
'a0_max':a0_max,'x_a0_max':x_a0_max,'injection':injection_flag,'t_i': ti,'x_i':xi,'a0':a0,'x_a':x_a,'E_mean':zeros_vector,'E_std':zeros_vector,
'E_peak':zeros_vector,'E_fwhm':zeros_vector,'dQdE_max':zeros_vector,'q':zeros_vector,'emit_y':zeros_vector,'emit_z':zeros_vector,'div_rms':zeros_vector,
'ener_axis':zeros_vector,'spec':zeros_vector,'x_p':zeros_vector,'n_e_p':zeros_vector}

df = pd.DataFrame(dict_data)

df = df[['Config','n_e_1', 'r','l_1','x_foc','c_N2','x_foc_vac', 'a0_max','x_a0_max','injection','t_i','x_i','a0','x_a','x_p','n_e_p',
'E_mean','E_std','E_peak','E_fwhm','dQdE_max',
'q','emit_y','emit_z','div_rms','ener_axis','spec']]

print('################# DEBUG ####################')
print("\t size dataframe:",df.shape)

for f in range(number_files):
    df['ener_axis'].iloc[f]         = energy_axis[f].astype(object)
    df['x_p'].iloc[f]               = x_p[f].astype(object)
    df['n_e_p'].iloc[f]             = n_e_p[f].astype(object)
    df['spec'].iloc[f]              = spectrum[f].astype(object)
    df['a0'].iloc[f]                = a0[f].astype(object)
    df['x_a'].iloc[f]               = x_a[f].astype(object) 
    df['E_peak'].iloc[f]            = E_peak[f].astype(object)
    df['dQdE_max'].iloc[f]          = dQdE_max[f].astype(object)
    df['E_fwhm'].iloc[f]            = E_fwhm[f].astype(object)
    df['E_mean'].iloc[f]            = E_mean[f].astype(object)
    df['E_std'].iloc[f]             = E_std[f].astype(object)
    df['q'].iloc[f]                 = q[f].astype(object)
    df['emit_y'].iloc[f]            = emittance_y[f].astype(object)
    df['emit_z'].iloc[f]            = emittance_z[f].astype(object)
    df['div_rms'].iloc[f]           = divergence_rms[f].astype(object)
            
# saving dataframe to pickle
df.to_pickle('dataframe_bfscan.pickle')

print('Post processing Ended')