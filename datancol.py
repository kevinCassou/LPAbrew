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

every_step = True # give 

timeStep = -1 

# for test, comment for postprocessing

number_files = 10

# initialization of arrays 400 is the binning in the histogram energy 

Config          = np.zeros([number_files])
n_e_1           = np.zeros([number_files])
r               = np.zeros([number_files])
l_1             = np.zeros([number_files])
x_foc           = np.zeros([number_files])
c_N2            = np.zeros([number_files])
x_foc_vac       = np.zeros([number_files])
x_p             = np.zeros([number_files,12])
n_e_p           = np.zeros([number_files,12])
injection_flag  = np.zeros([number_files])
ti              = np.zeros([number_files])
xi              = np.zeros([number_files])
zeros_vector    = np.zeros([number_files]) 
energy_axis     = np.zeros([number_files,400])

if every_step == False: 
    #
    print("")
    print("--------------------------------------------")
    print("")
    print("Post-processing timeStep : ",timeStep)
    print("")
    print("--------------------------------------------")
    print("")
    a0_max          = np.zeros([number_files])
    x_a0_max        = np.zeros([number_files])
    E_peak          = np.zeros([number_files])
    dQdE_max        = np.zeros([number_files])
    E_fwhm          = np.zeros([number_files])
    E_mean          = np.zeros([number_files])
    E_std           = np.zeros([number_files])
    spectrum        = np.zeros([number_files,400])
    q_end           = np.zeros([number_files])
    emittance_y     = np.zeros([number_files])
    emittance_z     = np.zeros([number_files])
    divergence_rms  = np.zeros([number_files])
else :
    print("")
    print("--------------------------------------------")
    print("")
    print("Post-processing :  all timeStep from injection")
    print("")
    print("--------------------------------------------")
    print("")
    a0_max          = []
    x_a0_max        = []
    E_peak          = []
    dQdE_max        = []
    E_fwhm          = []
    E_mean          = []
    E_std           = []
    spectrum        = []
    q_end           = []
    emittance_y     = []
    emittance_z     = []
    divergence_rms  = []


## scanning the configuration
for f in range(number_files):
    try: # SLK: continue postprocessing even in case of errors:
        # loading data 
        print("")
        print("--------------------------------------------")
        print("")
        print("loading data ...\n",str(files[f]),"\n")

        tmp = l.loadData(str(files[f]))
        # read configuration
        Config[f] = tmp.namelist.config_external['Config']
        n_e_1[f] = tmp.namelist.config_external['n_e_1']
        r[f]= tmp.namelist.config_external['r']
        l_1[f] = tmp.namelist.config_external['l_1']
        x_foc[f] = tmp.namelist.config_external['x_foc']
        c_N2[f] = tmp.namelist.config_external['c_N2']
        x_foc_vac[f] = tmp.namelist.xfocus
       
        # plasma profile
        x_p[f] = tmp.namelist.xr
        n_e_p[f] = tmp.namelist.ner

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
            print("#####################################\n",
            '#\t  injection occured at:\t',ti[f],' \n',
            "#####################################")
            # get electron spectrum at last timestep
            Emin = 50       # me c^2 unit 
            Emax = 1000     # me c^2 unit
            print('everys_step:',every_step)

            if every_step == True:
                # timestep from ionization
                tsi = ts[ti:]
                vec_len = len(tsi)
                print(vec_len)
                vec_a0_max          = np.zeros([vec_len])
                vec_x_a0_max        = np.zeros([vec_len])
                vec_E_peak          = np.zeros([vec_len])
                vec_dQdE_max        = np.zeros([vec_len])
                vec_E_fwhm          = np.zeros([vec_len])
                vec_E_mean          = np.zeros([vec_len])
                vec_E_std           = np.zeros([vec_len])
                vec_spectrum        = np.zeros([vec_len,400])
                vec_q_end           = np.zeros([vec_len])
                vec_emittance_y     = np.zeros([vec_len])
                vec_emittance_z     = np.zeros([vec_len])
                vec_divergence_rms  = np.zeros([vec_len])

                for t in range(len(tsi)):
                    print('file:\t',f)
                    print('timestep:\t',t)
                     # value and position of the max of a0 
                    x,a = l.getMaxinMovingWindow(tmp)
                    vec_a0_max[t] = a.max()
                    vec_x_a0_max[t] = x[a.argmax()]
                    # energy distribution characteristics 
                    energy_axis[f], vec_spectrum[t], vec_E_peak[t], vec_dQdE_max[t], vec_E_fwhm[t]  = l.getSpectrum(tmp,tsi[t], E_min=Emin, E_max = Emax, print_flag=True)
                    # beam parameter filter around 
                    if (vec_E_peak[t] == 0) or (vec_E_fwhm[t] == 0) :
                        param_list = l.getBeamParam(tmp,ts[t], E_min=Emin, E_max = Emax,print_flag=True)
                        vec_E_std[t] = param_list[3]
                        vec_E_peak[t] = np.nan
                        vec_E_fwhm[t] = np.nan
                        vec_E_std[t] = param_list[3]
                        vec_dQdE_max[t] = spectrum.max()
                        vec_emittance_y[t] = param_list[5]
                        vec_emittance_z[t] = param_list[6]
                        vec_divergence_rms[t] = param_list[10]
                        vec_q_end[t] = param_list[4]
                    else :
                        Emin = 50   #   np.max((50),(E_peak[f]-2*E_fwhm[f])/0.512))     # me c^2 unit
                        Emax = 1000  #  (E_peak[f]+2*E_fwhm[f])/0.512@                  # me c^2 unit
                        param_list = l.getBeamParam(tmp,ts[t], E_min=Emin, E_max = Emax ,print_flag=True)
                        vec_E_mean[t] = param_list[2]
                        vec_E_std[t] = param_list[3]
                        vec_emittance_y[t] = param_list[5]
                        vec_emittance_z[t] = param_list[6]
                        vec_divergence_rms[t] = param_list[10]
                        vec_q_end[t] = param_list[4]

                a0_max.append(vec_a0_max)
                x_a0_max.append(vec_x_a0_max)
                E_peak.append(vec_dQdE_max)
                dQdE_max.append(vec_dQdE_max)
                E_fwhm.append(vec_E_fwhm)
                E_mean.append(vec_E_mean)
                E_std.append(vec_E_std)
                spectrum.append(vec_spectrum)
                q_end.append(vec_q_end)
                emittance_y.append(vec_emittance_y)
                emittance_z.append(vec_emittance_z)
                divergence_rms.append(vec_divergence_rms)
            else :
                #  only the given final step value timeStep
                energy_axis[f], spectrum[f], E_peak[f], dQdE_max[f], E_fwhm[f]  = l.getSpectrum(tmp,ts[-1], E_min=Emin, E_max = Emax, print_flag=True)
                # beam parameter filter around 
                if (E_peak[f] == 0) or (E_fwhm[f] == 0) :
                    param_list = l.getBeamParam(tmp,ts[-1], E_min=Emin, E_max = Emax,print_flag=True)
                    E_std[f] = param_list[3]
                    E_peak[f] = np.nan
                    E_fwhm[f] = np.nan
                    E_std[f] = param_list[3]
                    dQdE_max[f] = spectrum.max()
                    emittance_y[f] = param_list[5]
                    emittance_z[f] = param_list[6]
                    divergence_rms[f] = param_list[10]
                    q_end[f] = param_list[4]
                else :
                    Emin = 50   #   np.max((50),(E_peak[f]-2*E_fwhm[f])/0.512))     # me c^2 unit
                    Emax = 1000  #  (E_peak[f]+2*E_fwhm[f])/0.512@                  # me c^2 unit
                    param_list = l.getBeamParam(tmp,ts[-1], E_min=Emin, E_max = Emax ,print_flag=True)
                    E_mean[f] = param_list[2]
                    E_std[f] = param_list[3]
                    emittance_y[f] = param_list[5]
                    emittance_z[f] = param_list[6]
                    divergence_rms[f] = param_list[10]
                    q_end[f] = param_list[4]
    except: # SLK:  in case of error fill values with nans and continue the postprocessing
        ti[f] = np.nan
        xi[f] = np.nan
        energy_axis[f] = np.nan
        if every_step == True :
                spectrum.append(np.nan)
                E_peak.append(np.nan)
                dQdE_max.append(np.nan)
                E_mean.append(np.nan)
                E_std.append(np.nan)
                E_fwhm.append(np.nan)
                emittance_y.append(np.nan)
                emittance_z.append(np.nan)
                divergence_rms.append(np.nan)
                q_end.append(np.nan)
                print('################# DEBUG ####################')
                print("\t,spec,q,ener ",len(spectrum),len(q_end),len(energy_axis))
 
        else :
            spectrum[f] = np.nan 
            E_peak[f] = np.nan
            dQdE_max[f] =  np.nan
            E_mean[f] = np.nan
            E_std[f] = np.nan
            E_fwhm[f] = np.nan
            emittance_y[f] = np.nan
            emittance_z[f] = np.nan
            divergence_rms[f] = np.nan
            q_end[f] = np.nan

        pass





# saving dataframe to changing 2D ndarray to list of array to avoid trouble opening the dataframe 

if every_step == True:

    dict_data = {'Config':Config,'n_e_1':n_e_1, 'r':r, 'l_1':l_1,'x_foc':x_foc,'c_N2':c_N2,'x_foc_vac':x_foc_vac,
    'a0_max':zeros_vector,'x_a0_max':zeros_vector,'injection':injection_flag,'t_i': ti,'x_i':xi,'E_mean':zeros_vector,'E_std':zeros_vector,
    'E_peak':zeros_vector,'E_fwhm':zeros_vector,'dQdE_max':zeros_vector,'q_end':zeros_vector,'emit_y':zeros_vector,'emit_z':zeros_vector,'div_rms':zeros_vector,'ener_axis':zeros_vector,'spec':zeros_vector,'x_p':zeros_vector,'n_e_p':zeros_vector}

    df = pd.DataFrame(dict_data)

    df = df[['Config','n_e_1', 'r','l_1','x_foc','c_N2','x_p','n_e_p','x_foc_vac', 'a0_max','x_a0_max',
    'injection','t_i','x_i','E_mean','E_std','E_peak','E_fwhm','dQdE_max',
    'q_end','emit_y','emit_z','div_rms','ener_axis','spec']]
    
    tmp_e = []
    tmp_s = []
    tmp_x = []
    tmp_ne = []

    for f in range(number_files):
        tmp_e.append(energy_axis[f])
        tmp_x.append(x_p[f])
        tmp_ne.append(n_e_p[f])

        df['ener_axis'] = tmp_e
        df['x_p']       = tmp_x
        df['n_e_p']     = tmp_ne
        df['spec']      = spectrum[f]
        df['x_p']       = x_p
        df['n_e_p']     = n_e_p
        df['a0_max']    = a0_max[f]
        df['x_a0_max']  = x_a0_max[f]    
        df['E_peak']    = E_peak[f]  
        df['dQdE_max']  = dQdE_max[f]
        df['E_fwhm']    = E_fwhm[f]
        df['E_mean']    = E_mean[f]
        df['E_std']     = E_std[f]
        df['q_end']     = q_end[f]
        df['emittance_y']  = emittance_y[f]
        df['emitance_z'] = emittance_z[f]
        df['divergence_rms'] = divergence_rms[f]

else: 

    dict_data = {'Config':Config,'n_e_1':n_e_1, 'r':r, 'l_1':l_1,'x_foc':x_foc,'c_N2':c_N2,'x_foc_vac':x_foc_vac,
    'a0_max':a0_max,'x_a0_max':x_a0_max,'injection':injection_flag,'t_i': ti,'x_i':xi,'E_mean':E_mean,'E_std':E_std,
    'E_peak':E_peak,'E_fwhm':E_fwhm,'dQdE_max':dQdE_max,'q_end':q_end,'emit_y':emittance_y,'emit_z':emittance_z,'div_rms':divergence_rms,'ener_axis':zeros_vector,'spec':zeros_vector,'x_p':zeros_vector,'n_e_p':zeros_vector}

    df = pd.DataFrame(dict_data)

    df = df[['Config','n_e_1', 'r','l_1','x_foc','c_N2','x_p','n_e_p','x_foc_vac', 'a0_max','x_a0_max',
    'injection','t_i','x_i','E_mean','E_std','E_peak','E_fwhm','dQdE_max',
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