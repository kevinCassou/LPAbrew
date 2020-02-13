
#!/usr/bin/python
# Author:       F Massimo, implemented as function by K Cassou
# Date:         2018-02-10, 2020-02-12
# Purpose:      plots 1D spectrum of Smilei Particles data
# Source:       Python
#####################################################################

### loading module
import os,sys
import happi
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as sc
import matplotlib.pyplot as plt

############ Inputs ##############################################
species_name   = "electronfromion"
lambda0        = 0.8e-6 # Wavelength of the laser

default_directory = "/ccc/scratch/cont003/smilei/beckarna/IJC_ionization"
homedirectory     = "/ccc/work/cont003/smilei/cassouke"

# used to apply a filter in energy (m_e c^2 units, or Lorentz factor)
E_min          = 0.
E_max          = 400

chunk_size     = 100000000  #Chunck of particles treated simultaneously

horiz_axis_conversion_factor = 0.512 # to convert from Smilei units to MeV
hist_conversion_factor       = 1.    # if equal to 1, the charge is in pC

########## Fundamental Physical constants ########################
eps0   = sc.epsilon0;   # Electric permittivity of vacuum, F/m
mu0    = sc.mu0; # Magnetic permittivity of vacuum, kg.m.A-2s-2
e      = sc.e; # Elementary charge, C
EeV    = sc.eV; # 1 eV = 1.6e-19 Joules
c      = sc.c;      # Lightspeed, m/s
me     = sc.m_e; # Electron mass, kg
mp     = sc.m_p;     # Proton mass, kg
h      = sc.h;   # Planck's constant, J.s
hbar   = sc.hbar

########## Physical constants ########################
omega0 = 2*math.pi*c/lambda0
onel   = lambda0/ (2*math.pi)
ncrit  = eps0*me*omega0**2/e**2; # critical density (m^-3, not cm^-3)


######### extract normalized a0 ##############################

def lasera0(iteration,directory=default_directory,laserfield="Ey"):
    """ return the a0 at the iteration taking the square of the laserfield max
    iteration : timestep max
    directory : scratch directory path [default_directory]
    laserfield : ["Ey"]
    return arrray - a0 and the timestep vector [0:iteration_max]
    """
    S = happi.Open(directory, show=False)
    dt_adim    = S.namelist.dt
    # read all timestep Available
    all_ts = S.Probe(0,laserfield).getTimeSteps()
    ts = all_ts[:iteration]
    a0 = []
    for t in ts:
        temp = S.Probe(0,laserfield,iteration).getData()[0]
        a0.append(max(temp))
    return np.array([a0,ts])

######### extract beam parameter for one iteration ###########

def beamParam(directory,iteration,species_name="electronfromion",saveflag=False):
    """return beams paramater for the species_name of the Smilei simulation data
    iteration : timestep
    directory : scratch directory path
    species_name : [electronfromion], electron
    saveflag : [False] True to save the data in an csv file
     """
    ########## Read data from Track Particles Diag ############
    S = happi.Open(directory, show=False)
    track_part = S.TrackParticles(species = species_name, chunksize=chunk_size)
    #print "Available timesteps = ",track_part.getAvailableTimesteps()
    dt_adim    = S.namelist.dt
    for particle_chunk in track_part.iterParticles(iteration, chunksize=chunk_size):
        # Read data
        print particle_chunk.keys()
        px           = particle_chunk["px"]
        py           = particle_chunk["py"]
        pz           = particle_chunk["pz"]
        x            = particle_chunk["x"]
        y            = particle_chunk["y"]
        z            = particle_chunk["z"]
        w            = particle_chunk["w"]
        p            = np.sqrt((px**2+py**2+pz**2))                # momentum
        E            = np.sqrt((1.+p**2))
        Nparticles   = np.size(w)                                  # Number of particles read
        print "Read ",Nparticles," particles from the file"
        total_weight = w.sum()
        Q            = total_weight* e * ncrit * onel**3 * 10**(12) # Total charge in pC
        print "Total charge before filter in energy= ",Q," pC"
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
        print "Total charge after filter in Energy = ",Q," pC"
        print "Filter energy limits: ",E_min,", ",E_max," (m_e c^2)"
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
            z2_moy   = (z**2 *w).sum() / total_weigh
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
                emittancey = math.sqrt(emittancey) * onel * 1e6 # [mm mrad]
            else:
                emittancey = 0.
            if emittancez > 0:
                emittancez = math.sqrt(emittancez) * onel * 1e6 # [mm mrad]
            else:
                emittancez = 0.
                emittance_transverse = math.sqrt(emittancey**2+emittancez**2) # [mm mrad]

            rmssize_longitudinal = 2*math.sqrt(x2_moy) * onel * 1e6 # [micron]
            rmssize_y =            2*math.sqrt(y2_moy) * onel * 1e6 # [micron]
            rmssize_z =            2*math.sqrt(z2_moy) * onel * 1e6 # [micron]
            divergence_rms = math.sqrt( py2ovpx2 + pz2ovpx2 )

        print ""
            print "--------------------------------------------"
            print ""
            print "Read ",np.size(E)," particles"
            print "Iteration = ",iteration
            print "Simulation time = ",iteration*dt_adim*onel/c*1e15," fs"
            print "E_mean = ",np.mean(E)*0.512," MeV"
            print "2*DeltaE_rms / E_mean = ", np.std(E)/np.mean(E)*100 , " %."
            print "Total charge = ", Q, " pC."
            print "Emittance_y = ",emittancey," mm-mrad"
            print "Emittance_z = ",emittancez," mm-mrad"
            print ""
            print "--------------------------------------------"
	    print ""

        if saveflag == True:
            print "data saved in cvs file"
            vdata = np.array([iteration,
            iteration*dt_adim*onel/c*1e15,
            np.mean(E)*0.512,
            np.std(E)/np.mean(E)*100,
            Q,
            emittancey,
            emittancez])
            filename = 'smilei-data-it'+str(iteration)+'.csv'
            filepath = homedirectory+'/'+filename
            vdata.tofile(filepath,sep=',',format='%10.5f')
