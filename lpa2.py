#!/usr/bin/python
# Author:       F Massimo, implemented as function by K Cassou
# Date:         2018-02-10, 2020-02-12, 2020-06-21 
# Purpose:      set of function for LPA simulation with SMILEI
# Source:       Python 3 (python2)
#####################################################################

### loading module
from __future__ import (division, print_function, absolute_import,unicode_literals)
import os,sys
import happi
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import scipy.constants as sc
import matplotlib.pyplot as plt

############ Inputs ##############################################

species_name   = "electronfromion"
lambda0        = 0.8e-6 # Wavelength of the laser

default_directory = "/ccc/scratch/cont003/smilei/cassouke"
homedirectory     = "/ccc/work/cont003/smilei/cassouke"

# used to apply a filter in energy (m_e c^2 units, or Lorentz factor)
E_min          = 0.
E_max          = 420

chunk_size     = 100000000  #Chunck of particles treated simultaneously

horiz_axis_conversion_factor = 0.512 # to convert from Smilei units to MeV
hist_conversion_factor       = 1.    # if equal to 1, the charge is in pC

########## Fundamental Physical constants ########################
eps0   = sc.epsilon_0;  # Electric permittivity of vacuum, F/m
mu0    = sc.mu_0;       # Magnetic permittivity of vacuum, kg.m.A-2s-2
e      = sc.e;          # Elementary charge, C
EeV    = sc.eV;         # 1 eV = 1.6e-19 Joules
c      = sc.c;          # Lightspeed, m/s
me     = sc.m_e;        # Electron mass, kg
mp     = sc.m_p;        # Proton mass, kg
h      = sc.h;          # Planck's constant, J.s
hbar   = sc.hbar

########## Physical constants ########################
omega0 = 2*np.pi*c/lambda0
onel   = lambda0/ (2*np.pi)
ncrit  = eps0*me*omega0**2/e**2; # critical density (m^-3, not cm^-3)


######### functions ######################################## 
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

########## load data with happi ##############################

def loadData(directory=default_directory):
    """loading data in the simulation directory and return an object pointing to the various 
    files, see smilei website"""
    S = happi.Open(directory, show=False)
    return S

######### extract normalized var ##############################

def getMaxinMovingWindow(S,var="Env_E_abs"):
    """ return the max of var at the iteration for the moving window
    on axis (r=0)
    S : is the simulation output object return by happi.Open()
    var : check namelist ["Env_E_abs]
    return a numpy array - var.max() and the timestep vector [0:iteration_max]
    """
    # read all timestep Available
    ts = S.Probe(0,var).getTimesteps()
    varmax = []
    for t in ts:
        temp = S.Probe(0,var,t).getData()[0]
        varmax.append(max(temp))
    return np.array([ts,varmax])

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

def beamParam(S,iteration,species_name="electronfromion",E_min=0,E_max=400,chunk_size=100000000,print_flag=True,save_flag=False):
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
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())
    dt_adim    = S.namelist.dt
    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        if print_flag==True:
            print(particle_chunk.keys())
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
                emittance_transverse = np.sqrt(emittancey**2+emittancez**2) # [mm mrad]

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
                print("Read ",np.size(E)," particles")
                print( "Iteration = ",iteration)
                print( "Simulation time = ",iteration*dt_adim*onel/c*1e15," fs")
                print( "E_mean = ",np.mean(E)*0.512," MeV")
                print( "2*DeltaE_rms / E_mean = ", np.std(E)/np.mean(E)*100 , " %.")
                print( "Total charge = ", Q, " pC.")
                print( "Emittance_y = ",emittancey," mm-mrad")
                print( "Emittance_z = ",emittancez," mm-mrad")
                print( "divergence_rms = ",divergence_rms)
                print( "")
                print( "--------------------------------------------")
                print( "")

            # beam paramater list for iteration timestep
            vlist = [iteration,
            iteration*dt_adim*onel/c*1e15,
            np.mean(E)*0.512,
            np.std(E)/np.mean(E)*100,
            Q,
            emittancey,
            emittancez,
            rmssize_longitudinal,
            rmssize_y,
            rmssize_z,
            divergence_rms]

            # save beam parameter in a file
            if save_flag == True:
                print( "data saved in cvs file")
                vdata = np.array(vlist)
                filename = 'smilei-beamparam'+str(iteration)+'.csv'
                filepath = homedirectory+'/'+filename
                vdata.tofile(filepath,sep=',',format='%10.5f')
            return vlist

def getPartAvailableSteps(S,species_name="electronfromion",chunk_size=10000000):
    """return available timesteps for the trackParticles"""
    return S.TrackParticles(species = species_name, chunksize=chunk_size).getAvailableTimesteps()

def getSpectrum(S,iteration_to_plot,species_name= "electronfromion",horiz_axis_name= "E",chunk_size=100000000,E_min=0,E_max=400,plot_flag=False,print_flag=False):
    """ return spectrum plot or data for a given timesteps
    S : smilei output data
    iteration_to_plot : timestep 
    species_name : [electronfromion], electron
    horiz_axis_name : [E] can be px, p or E 
    E_min : [0] min value considered in histogram for the horiz axis, in code units 
    E_max : [400] max value considered in histogram for the horiz axis, in code units
    return spectrum data as (horizontal axis (E, or p),S(E,or p)) 
    """
    #global specData

    # histogram settings,
    normalized     = False
    nbins_horiz    = 200
    
    #  horizontal axis limits (m_e c^2 units, or Lorentz factor)
    horiz_axis_min = E_min  # Max value considered in histogram for the horiz axis, in code units
    horiz_axis_max = E_max   # Min value considered in histogram for the horiz axis, in code units
    horiz_axis_conversion_factor = 0.512 # to convert from Smilei units to MeV
    hist_conversion_factor       = 1.    # if equal to 1, the charge is in pC

    ########## Read data from Track Particles Diag  
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())
    
    dt_adim    = S.namelist.dt
    for particle_chunk in track_part.iterParticles(iteration_to_plot, chunksize=chunk_size):
        # Read data
        if print_flag==True:
            print(particle_chunk.keys())
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
        if print_flag==True:
            print("Read ",Nparticles," particles from the file")
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        if print_flag==True:
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
        if print_flag==True:
            print("Total charge after filter in Energy = ",Q," pC")
            print("Filter energy limits: ",E_min,", ",E_max," (m_e c^2)")

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
        if normalized==True:
            histogram_spectrum = histogram_spectrum / histogram_spectrum[:].max()

        # horizontal axis 
        horiz_edges = horiz_edges[0:-1]
        horiz_edges = horiz_edges + 0.5*dhoriz_axis*horiz_axis_conversion_factor
        energy_axis = horiz_edges*horiz_axis_conversion_factor

        # Preparation for Plot
        if normalized==True:
            plot_title   = "Normalized histogram"
        else:
            plot_title   = 'dQ/d'+horiz_axis_name+" (pC/MeV)"

        #print np.shape(histogram_spectrum)
        #
        print('Total charge in in the histogram =',np.sum(histogram_spectrum[:])*dhoriz_axis*horiz_axis_conversion_factor,' pC')
        print('Bins size: dx = ',dhoriz_axis*horiz_axis_conversion_factor)

        histogram_spectrum[histogram_spectrum==0.]=float(np.nan)
        print(len(energy_axis))
        specData = np.array((histogram_spectrum))
        
        # Plot
        if plot_flag == True:
            fig = plt.figure()
            fig.set_facecolor('w')
            ax=plt.gca()

            plt.xlabel(horiz_axis_name+" (MeV)")
            plt.title(plot_title)

            extnt = np.array([horiz_axis.min()*horiz_axis_conversion_factor, \
                          horiz_axis.max()*horiz_axis_conversion_factor ])
            print("Values extension for ",horiz_axis_name," (all particles):")
            print(extnt)

            extnt = np.array([horiz_axis_min*horiz_axis_conversion_factor, \
                          horiz_axis_max*horiz_axis_conversion_factor])
            print( "Values extension for ",horiz_axis_name," (particles included in the chosen horiz axis limits):")
            print(extnt)

            plt.plot(energy_axis,histogram_spectrum)
            plt.xlim([horiz_axis_min*horiz_axis_conversion_factor,horiz_axis_max*horiz_axis_conversion_factor])
            
            #plt.savefig(homedirectory+"/E_Spectrum.png",format='png')
            plt.show()
            
    return specData, energy_axis

def getParticles(S,iteration,species_name="electronfromion",chunk_size=100000000,print_flag = True):
    """return x,y,z,px,py,pz,E,w,p for all particle at timesteps iteration within the filter"""
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size)
    #print("Available timesteps = ",track_part.getAvailableTimesteps())
    dt_adim    = S.namelist.dt
    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        if print_flag==True:
        	print(particle_chunk.keys())
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
        
    return np.array([x,y,z,px,py,pz,E,w,p])

