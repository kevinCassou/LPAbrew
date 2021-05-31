#!/usr/bin/python
# Author:       F Massimo, implemented as function by K Cassou
# Date:         2018-02-10, 2020-02-12, 2020-06-21 
# Purpose:      set of function for LPA simulation with SMILEI
# Source:       Python 3 (python2)
#####################################################################

### loading module
from __future__ import (division, print_function, absolute_import,unicode_literals)
import os,sys
import warnings
import statsmodels.api
import statsmodels.stats as stats

try :
    import happi
except ImportError :
    try:
        with open("somefile.py") as f:
            code = compile(f.read(), "Diagnostics.py", 'exec')
            exec(code)
    except IOError:
        print('copy the Diagnostics.py file from Smilei directory')
    pass

#sys.path.append('/Users/cassou/Simulations/Smilei/scripts/Diagnostics.py')
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import scipy.constants as sc
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

#################### Inputs ##############################################

species_name   = "electronfromion"
lambda0        = 0.8e-6 # Wavelength of the laser

default_directory = "/ccc/scratch/cont003/smilei/cassouke"
homedirectory     = "/ccc/work/cont003/smilei/cassouke"

# used to apply a filter in energy (m_e c^2 units, or Lorentz factor)
E_min          = 0.
E_max          = 500

chunk_size     = 100000000  #Chunck of particles treated simultaneously

horiz_axis_conversion_factor = 0.512 # to convert from Smilei units to MeV
hist_conversion_factor       = 1.    # if equal to 1, the charge is in pC

################# Fundamental Physical constants ########################
eps0   = sc.epsilon_0;  # Electric permittivity of vacuum, F/m
mu0    = sc.mu_0;       # Magnetic permittivity of vacuum, kg.m.A-2s-2
e      = sc.e;          # Elementary charge, C
EeV    = sc.eV;         # 1 eV = 1.6e-19 Joules
c      = sc.c;          # Lightspeed, m/s
me     = sc.m_e;        # Electron mass, kg
mp     = sc.m_p;        # Proton mass, kg
h      = sc.h;          # Planck's constant, J.s
hbar   = sc.hbar

########## Physical constants ##########################################
omega0 = 2*np.pi*c/lambda0          # 
onel   = lambda0/ (2*np.pi)         # 
ncrit  = eps0*me*omega0**2/e**2;    # critical density (m^-3, not cm^-3)


######### useful functions #################################### 
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),lin_interp(x, y, zero_crossings_i[1], half)]

def fwhm(x,y):
    hmx = half_max_x(x,y)
    return hmx[1]-hmx[0]

def gaussian(x, amp, xcenter, width):
    return amp * np.exp(-(x-xcenter)**2 / width**2)

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

def mad(data, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """
    # Median along given axis, but *keeping* the reduced axis so that
    # result can still broadcast against a.
    med = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.absolute(data - med), axis=axis)  # MAD along given axis

    return mad

def weighted_std(data, weights):
    """
    Compute *weighted_standard Deviation* of an array along given axis.
    """    
    d = stats.weightstats.DescrStatsW(data,weights)

    return d.std

def weighted_mean(data, weights):
    """
    Compute *weighted_mean* of an array along given axis.
    """    
    d = stats.weightstats.DescrStatsW(data,weights)

    return d.mean

def weighted_mad(data, weights):
    """
    Compute *weighted_Median Absolute Deviation* of an array along given axis.
    """
    wmed = weighted_median(data,weights)
    wmad = weighted_median(np.absolute(data - wmed),weights)  # MAD along given axis
    
    return wmad


########## load data with happi ##############################

def loadData(directory=default_directory):
    """loading data in the simulation directory and return an object pointing to the various 
    files, see smilei website"""
    S = happi.Open(directory, show = False,verbose = False, )
    return S

######### extract laser var ##############################

def getMaxinMovingWindow(S,var="Env_E_abs"):
    """ return the max of var on axis (r=0) for all timestep available
    S : is the simulation output object return by happi.Open()
    var : check namelist ["Env_E_abs]
    return a numpy array - var.max() and the timestep vector [0:iteration_max]
    """
    # read all timestep Available
    ts = S.Probe(0,var).getTimesteps()
    varmax = np.zeros((2,len(ts)))
    for t in range(len(ts)):
        temp = S.Probe(0,var,ts[t]).getData()[0]
        varmax[0,t] = ts[t]
        varmax[1,t] = np.max(temp)
    return varmax

def getLasera0(S,timeStep,var='Env_E_abs'):
    """ return the max of var on axis (r=0) for the timestep 
    S : is the simulation output object return by happi.Open()
    timeStep : timestep smilei unit
    var : check namelist ["Env_E_abs]
    return a numpy array - var.max() and the timestep vector [0:iteration_max]
    """
    return np.max(S.Probe(0,var,timeStep).getData()[0])
    
def getLaserWaist(S,timeStep,var='Env_E_abs'):
    """ return the laser waist of Env or field `var` at the iteration
    S : is the simulation output object return by happi.Open()
    timestep : simulation timestep
    var : check namelist ["Env_E_abs" or laser field] to be updated for AM geometry
    return the waist evaluated with Gaussian fit in code units (lamda_0/2pi)
    """
    temp = S.Probe(1,var,timeStep).getData()[0]
    x_max,y_max = np.unravel_index(np.argmax(temp),temp.shape)
    init_vals = [np.max(temp),y_max, 1.0]
    a_val = temp[x_max,:]
    y_val = np.arange(0,temp.shape[1],1)
    # gaussian fit 
    best_vals, covar = curve_fit(gaussian, y_val, a_val, p0=init_vals)
    return best_vals[2]

def getLaserPulselength(S,timeStep,var='Env_E_abs'):
    """ return the laser pulse length of Env or field `var` at the iteration 
    S : is the simulation output object return by happi.Open()
    timestep : simulation timestep
    var : check namelist ["Env_E_abs" or laser field] to be updated for AM geometry
    return the pulse length FWHM evaluated with Gaussian fit in code units (lamda_0/2pi)
    """
    temp = S.Probe(1,var,timeStep).getData()[0]
    x_max,y_max = np.unravel_index(np.argmax(temp),temp.shape)
    init_vals = [np.max(temp),x_max, 1.0]
    a_val = temp[:,y_max]
    x_val = np.arange(0,temp.shape[0],1)
    # gaussian fit slightly underestimate FWHM value
    best_vals, _ = curve_fit(gaussian, x_val, a_val, p0=init_vals)
    return best_vals[2]*2*np.sqrt(2*np.log(2))


######### extract plasma profile ############################

def plasmaProfile(S):
    """ return the electon plasma density  profile
    S : is the simulation output object return by happi.Open()
    return the numpy array - plasProfile (x,ne) e-/m^3
    """
    nc = S.namelist.ncrit
    ne = np.array(S.namelist.xh_values)*nc
    plasProfile = np.array((S.namelist.xh_points,ne))
    return plasProfile

def dopantProfile(S): 
    """ return the electon dopan density profile 
    S : is the simulation output object return by happi.Open()
    return the numpy array - plasProfile (x,nN2) N2/m^3
    """
    nc = S.namelist.ncrit
    nd = np.array(S.namelist.xd_values)*nc
    dopProfile = np.array((S.namelist.xd_points,nd))
    return dopProfile

######### extract beam parameter for one iteration ###########

def getBeamParam(S,iteration,species_name="electronfromion",sort = False, E_min=50,E_max=520,chunk_size=100000000,print_flag=True,save_flag=False):
    """return beams paramater for the species_name of the Smilei simulation data
    iteration : timestep
    S : is the simulation output object return by happi.Open()
    species_name :  [electronfromion], electron
    E_min :         [0] energy filter min 
    E_max :         [400] energy filter max
    printflag :     [True] print output on screen. 
    saveflag :      [False] True to save the data in an csv file
     """
    ########## Read data from Track Particles Diag ############
    track_part = S.TrackParticles(species = species_name, sort = sort, chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())
    dt_adim    = S.namelist.dt
    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        #if print_flag==True:
        #    print(particle_chunk.keys())
        px           = particle_chunk["px"]
        py           = particle_chunk["py"]
        pz           = particle_chunk["pz"]
        x            = particle_chunk["x"]
        y            = particle_chunk["y"]
        z            = particle_chunk["z"]
        w            = particle_chunk["w"]
        p            = np.sqrt((px**2+py**2+pz**2))                # momentum
        E            = np.sqrt((1.+p**2))
        Nparticles   = np.size(w)
        if print_flag == True:                                  # Number of particles read
            print("Read ",Nparticles," particles from the file")
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag == True:  
            print("Total charge before filter in energy= ",Q," pC")
        # Apply a filter on energy
        filter       = np.intersect1d( np.where( E > E_min )[0] ,  np.where( E < E_max )[0] )
        x            = x[filter]
        y            = y[filter]
        z            = z[filter]
        px           = px[filter]
        py           = py[filter]
        pz           = pz[filter]
        E            = E[filter]
        w            = w[filter]
        p            = p[filter]
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag == True:  
            print("Total charge after filter in Energy = ",Q," pC")
            print("Filter energy limits: ",E_min,", ",E_max," (m_e c^2)")
        if total_weight > 0:
            #Compute mean values
            x_moy    = (x    *w).sum() / total_weight
            y_moy    = (y    *w).sum() / total_weight
            z_moy    = (z    *w).sum() / total_weight
            #px_moy   = (px   *w).sum() / total_weight
            py_moy   = (py   *w).sum() / total_weight
            pz_moy   = (pz   *w).sum() / total_weight
            p_moy    = (p    *w).sum() / total_weight
            #Place center of mass at the center of the coordinates
            x  -= x_moy
            y  -= y_moy
            z  -= z_moy
            #px -= px_moy
            py -= py_moy
            pz -= pz_moy
            # Compute properties of the bunch
            x2_moy   = (x**2 *w).sum() / total_weight
            y2_moy   = (y**2 *w).sum() / total_weight
            z2_moy   = (z**2 *w).sum() / total_weight
            #px2_moy  = (px**2*w).sum() / total_weight
            py2_moy  = (py**2*w).sum() / total_weight
            pz2_moy  = (pz**2*w).sum() / total_weight
            ypy_moy  = (y*py *w).sum() / total_weight
            zpz_moy  = (z*pz *w).sum() / total_weight
            py2ovpx2 = (py**2/px**2*w).sum()/total_weight #divergence y squared
            pz2ovpx2 = (pz**2/px**2*w).sum()/total_weight

            # emittances
            emittancey = ( py2_moy*y2_moy - ypy_moy**2 )
            emittancez = ( pz2_moy*z2_moy - zpz_moy**2 )
            if emittancey > 0:
                emittancey = np.sqrt(emittancey) * onel * 1e6 # [mm mrad]
            else:
                emittancey = 0.
            if emittancez > 0:
                emittancez = np.sqrt(emittancez) * onel * 1e6 # [mm mrad]
            else:
                emittancez = 0.
                #emittance_transverse = np.sqrt(emittancey**2+emittancez**2) # [mm mrad]

            rmssize_longitudinal = 2*np.sqrt(x2_moy) * onel * 1e6 # [micron]
            rmssize_y =            2*np.sqrt(y2_moy) * onel * 1e6 # [micron]
            rmssize_z =            2*np.sqrt(z2_moy) * onel * 1e6 # [micron]
            divergence_rms = np.sqrt( py2ovpx2 + pz2ovpx2 )

            # width of the peak to be implemented

            # print beam parameter
            if print_flag == True:
                print("")
                print("--------------------------------------------")
                print("")
                print(" Read \t\t\t\t\t\t",np.size(E)," particles")
                print( "[0] Iteration = \t\t\t\t\t ",iteration)
                print( "[1] Simulation time = \t\t\t ",iteration*dt_adim*onel/c*1e15," fs")
                print( "[2] E_mean = \t\t\t\t\t ",weighted_mean(E,w)*0.512," MeV")
                print( "[3] E_med = \t\t\t\t\t ", weighted_median(E,w)*0.512, "MeV")
                print( "[4] DeltaE_rms / E_mean = \t\t\t", weighted_std(E,w)/weighted_mean(E,w)*100 , " %.")
                print( "[5] E_mad /E_med  = \t\t\t\t ",weighted_mad(E,w)/weighted_median(E,w)*100, " %.")
                print( "[6] Total charge = \t\t\t\t ", Q, " pC.")
                print( "[7] Emittance_y = \t\t\t\t",emittancey," mm-mrad")
                print( "[8] Emittance_z = \t\t\t\t",emittancez," mm-mrad")
                print( "[8] size_x = \t\t\t\t\t\t",rmssize_longitudinal,"um (RMS)")
                print( "[10] divergence_rms = \t\t\t\t",divergence_rms*1e3,"mrad")
                print( "")
                print( "--------------------------------------------")
                print( "")

            # beam paramater list for iteration timestep
           
            if Q > 0.0 :
                # [0] timestep
                # [1] time [fs]
                # [2] weighted mean energy   [MeV]
                # [3] weighted median value   [MeV]
                # [4] RMS energy spread   [%]
                # [5] MAD value [%]
                # [6] charge [pC]
                # [7] emittance [pi.mm.mrad]
                # [8] emittance [pi.mm.mrad]
                # [9] bunch RMS length [um]
                # [10] bunch RMS sigy [um]
                # [11] bunch RMS sigz [um]
                # [12] RMS divergence [mrad]
                beamparam_dict = {"iteration":iteration,
                "time": iteration*dt_adim*onel/c*1e15,
                "energy_wmean": weighted_mean(E,w)*0.512,
                "energy_wmedian": weighted_median(E,w)/weighted_median(E,w)*100,
                "energy_wrms": weighted_std(E,w)/weighted_mean(E,w)*100,
                "energy_wmad": weighted_mad(E,w)/weighted_median(E,w)*100,
                "charge": Q,
                "emittancey": emittancey,
                "emittancez": emittancez,
                "size_x_rms": rmssize_longitudinal,
                "size_y_rms" : rmssize_y,
                "size_z_rms" : rmssize_z,
                "divergence_rms": divergence_rms*1e3}
            else:
                print('no data in the filtered energy range')
                beamparam_dict = {"iteration":iteration,
                "time": iteration*dt_adim*onel/c*1e15,
                "energy_wmean": np.nan,
                "energy_wmedian": np.nan,
                "energy_wrms": np.nan,
                "energy_wmad": np.nan,
                "charge": np.nan,
                "emittancey": np.nan,
                "emittancez": np.nan,
                "size_x_rms": np.nan,
                "size_y_rms" : np.nan,
                "size_z_rms" : np.nan,
                "divergence_rms": np.nan}

            # save beam parameter in a file
            if save_flag == True:
                print( "data saved in npy file")
                filename = 'smilei-beamparam'+str(iteration)+'.npy'
                filepath = homedirectory+'/'+filename
                np.save(filepath,beamparam_dict)
            return beamparam_dict

def getPartAvailableSteps(S,species_name="electronfromion",sort = False, chunk_size=10000000):
    """return available timesteps for the trackParticles"""
    return S.TrackParticles(species = species_name, sort = False, chunksize=chunk_size).getAvailableTimesteps()

def getInjectionTime(S,ts,specie='Rho_electronfromion',threshold = 1e-4,print_flag = False):
    """ return the injection timestep and longitudinal coordinate of the injection.
    The injection is defined by a threshold on the `electron_from_ion` density
    S : is the simulation output object return by happi.Open()
    ts : timestep vector [numpy array]
    ti : injection timestep 
    xi : injection longitudinal position [m]
    """ 
    dls = S.namelist.lambda_0/(2*np.pi)
    
    for t in range(len(ts)):
        rhoei = S.Probe(0,specie,ts[t]).getData()[0]
        if np.abs(rhoei.min())> threshold:
            ti = ts[t]
            xi = ts[t]*dls
            #print('index:', t
            #print('injection time:',ti,'timestep')
            #print('injection x:',xi,'mm')
            break
        else :
            ti = None
            xi = None
    return t,ti,xi

def getSpectrum(S,iteration_to_plot,species_name= "electronfromion",horiz_axis_name= "E", E_min=50, E_max = 640,plot_flag = False, print_flag = False, nbins_horiz = 200, normalized = False):
    """ return spectrum plot or data for a given timesteps
    S : smilei output data
    iteration_to_plot : timestep 
    species_name : [electronfromion], electron
    horiz_axis_name : [E] can be px, p or E 
    E_min : [50] min value considered in histogram for the horiz axis, in code units 
    E_max : [640] max value considered in histogram for the horiz axis, in code units
    option : plot_flag, print_flag,
    nbins_horiz : binning energy histogram [200]
    normalized : normalization of the histogram [False]
    return spectrum data as numpy arrays  (horizontal axis (E, or p)), dQd(E,or p), Epeak, dQdE_max, Ewidth 
    """
    
    #  horizontal axis limits (m_e c^2 units, or Lorentz factor)
    horiz_axis_min = E_min  # Max value considered in histogram for the horiz axis, in code units
    horiz_axis_max = E_max   # Min value considered in histogram for the horiz axis, in code units
    horiz_axis_conversion_factor = 0.512 # to convert from Smilei units to MeV
    hist_conversion_factor       = 1.    # if equal to 1, the charge is in pC

    ########## Read data from Track Particles Diag #####################
    sort = False  
    chunk_size=100000000
    track_part = S.TrackParticles(species = species_name, sort = sort, chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())
    
    for particle_chunk in track_part.iterParticles(iteration_to_plot, chunksize=chunk_size):
        # Read data
        #if print_flag==True:
        #    print(particle_chunk.keys())
        px           = particle_chunk["px"]
        py           = particle_chunk["py"]
        pz           = particle_chunk["pz"]
        x            = particle_chunk["x"]
        y            = particle_chunk["y"]
        z            = particle_chunk["z"]
        w            = particle_chunk["w"]
        p            = np.sqrt((px**2+py**2+pz**2))                # momentum
        E            = np.sqrt((1.+p**2))
        Nparticles   = np.size(w)                                    # Number of particles read
        if print_flag == True:
            print("Read ",Nparticles," particles from the file")
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag == True:
            print("Total charge before filter in energy= ",Q," pC")
        # Apply a filter on energy
        filter       = np.intersect1d( np.where( E > E_min )[0] ,  np.where( E < E_max )[0] )
        x            = x[filter]
        y            = y[filter]
        z            = z[filter]
        px           = px[filter]
        py           = py[filter]
        pz           = pz[filter]
        E            = E[filter]
        w            = w[filter]
        p            = p[filter]
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag == True:
            print("Total charge after filter in Energy = ",Q," pC")
            print("Filter energy limits: ",E_min,", ",E_max," (m_e c^2)")

        if Q > 0.0 : 

            # Compute 1D histogram
            possible_axes_names =["x","y","z","px","py","pz","E"]
            axes                =[x,y,z,px,py,pz,E]

            if horiz_axis_name in possible_axes_names:
                horiz_axis = axes[possible_axes_names.index(horiz_axis_name)]
            else:
                print("Error, invalid axis")
                exit(0)

            hist1D, horiz_edges = np.histogram(horiz_axis, \
                                        bins=nbins_horiz, \
                                        range=[horiz_axis_min,horiz_axis_max], weights=w)
            #print(np.shape(horiz_edges))
            dhoriz_axis                    = abs(horiz_edges[1]-horiz_edges[0]) # bin size

            # histogram: integrated in dhoriz_axis and gives the total charge
            histogram_spectrum = hist1D*hist_conversion_factor/dhoriz_axis/horiz_axis_conversion_factor*e * ncrit * onel**3  * 10**(12)
            if normalized == True:
                histogram_spectrum = histogram_spectrum / histogram_spectrum[:].max()

            # horizontal axis 
            horiz_edges = horiz_edges[0:-1]
            binx = dhoriz_axis*horiz_axis_conversion_factor
            horiz_edges = horiz_edges + 0.5*binx
            energy_axis = horiz_edges*horiz_axis_conversion_factor

            # Preparation for Plot
            if normalized == True:
                plot_title   = "Normalized histogram"
            else:
                plot_title   = 'dQ/d'+horiz_axis_name+" (pC/MeV)"

            #print np.shape(histogram_spectrum)
            #
            if print_flag == True:
                print('Bins size: dx = ',binx)

            histogram_spectrum[histogram_spectrum==0.]=float(np.nan)

            #if print_flag==True:
            #    print(len(energy_axis))

            specData = np.array((histogram_spectrum))
            
            # Plot
            if plot_flag == True:
                fig = plt.figure()
                fig.set_facecolor('w')

                plt.xlabel(horiz_axis_name+" (MeV)")
                plt.title(plot_title)

                #extnt = np.array([horiz_axis.min()*horiz_axis_conversion_factor, \
                #            horiz_axis.max()*horiz_axis_conversion_factor ])
                #print("Values extension for ",horiz_axis_name," (all particles):")
                #print(extnt)

                #extnt = np.array([horiz_axis_min*horiz_axis_conversion_factor, \
                #            horiz_axis_max*horiz_axis_conversion_factor])
                #print( "Values extension for ",horiz_axis_name," (particles included in the chosen horiz axis limits):")
                #print(extnt)

                plt.plot(energy_axis,histogram_spectrum)
                plt.xlim([horiz_axis_min*horiz_axis_conversion_factor,horiz_axis_max*horiz_axis_conversion_factor])
                
                #plt.savefig(homedirectory+"/E_Spectrum.png",format='png')
                plt.show()
        
            # compute the full width half maximum using scipy.signal.findpeaks 
            try :
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                    prom = (np.nanmax(specData)-np.nanmin(specData))*0.66 #factor might be adjusted 
                    p , _  = find_peaks(specData,prominence=prom)
                if len(p)==0 :
                    Epeak = 0
                    Ewidth = 0
                    dQdE_max = 0
                else :  
                    Epeak = energy_axis[p[0]]
                    dQdE_max = specData[p[0]]
                    Ewidth = binx*peak_widths(specData, p, rel_height=0.5)[0][0]
            except ValueError :
                Epeak = np.nan
                Ewidth = np.nan
                dQdE_max = np.nan 
                
            if print_flag == True:
                print( "")
                print( "--------------------------------------------")
                print( "")
                print("beam Peak energy: \t",Epeak,"MeV")
                print("beam FWHM energy: \t",Ewidth,"MeV")
                print( "")
                print( "--------------------------------------------")
                print( "")
        # no charge in the energy range  Q = 0.    
        else : 
            energy_axis = np.nan
            specData = np.nan
            Epeak  = np.nan
            dQdE_max = np.nan
            Ewidth = np.nan

    return energy_axis, specData, Epeak, dQdE_max, Ewidth

def getPartParam(S,iteration,species_name="electronfromion",sort= False,chunk_size=100000000,E_min=25, E_max = 520,print_flag = True):
    """return x,y,z,px,py,pz,E,w,p for all particle at timesteps iteration within the filter"""
    track_part = S.TrackParticles(species = species_name,sort = sort,  chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())

    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        #if print_flag==True:
        #	print(particle_chunk.keys())
        px           = particle_chunk["px"]
        py           = particle_chunk["py"]
        pz           = particle_chunk["pz"]
        x            = particle_chunk["x"]
        y            = particle_chunk["y"]
        z            = particle_chunk["z"]
        w            = particle_chunk["w"]
        p            = np.sqrt((px**2+py**2+pz**2))                # momentum
        E            = np.sqrt((1.+p**2))
        Nparticles   = np.size(w)
        if print_flag==True:                                  # Number of particles read
            print("Read ",Nparticles," particles from the file")
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag==True:  
            print("Total charge before filter in energy= ",Q," pC")
            print("Filter energy limits: ",E_min,", ",E_max," (m_e c^2)")
        # Apply a filter on energy
        filter       = np.intersect1d( np.where( E > E_min )[0] ,  np.where( E < E_max )[0] )
        x            = x[filter]
        y            = y[filter]
        z            = z[filter]
        px           = px[filter]
        py           = py[filter]
        pz           = pz[filter]
        E            = E[filter]
        w            = w[filter]
        p            = p[filter]
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag==True:  
            print("Total charge after filter in energy= ",Q," pC")
    return np.array([x,y,z,px,py,pz,E,w,p])

def getPSxrms(S,iteration,species_name="electronfromion",sort= False,chunk_size=100000000,E_min=25, E_max = 520,print_flag = True):
    """return x,px for all particle at timesteps iteration within the filter"""
    track_part = S.TrackParticles(species = species_name,sort = sort,  chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())

    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        #if print_flag==True:
        #	print(particle_chunk.keys())
        px           = particle_chunk["px"]
        py           = particle_chunk["py"]
        pz           = particle_chunk["pz"]
        x            = particle_chunk["x"]               
        w            = particle_chunk["w"]
        p            = np.sqrt((px**2+py**2+pz**2))                # momentum
        E            = np.sqrt((1.+p**2))              
        Nparticles   = np.size(w)
        if print_flag==True:                                  # Number of particles read
            print("Read ",Nparticles," particles from the file")

        # Apply a filter on energy
        filter       = np.intersect1d( np.where( E > E_min )[0] ,  np.where( E < E_max )[0] )
        x            = x[filter]
        px           = px[filter]
        w            = w[filter]
        Nparticles   = np.size(w)
        if print_flag==True:                                  # Number of particles read
            print("After filtering",Nparticles," particles")
  
    return np.array([x,px])
