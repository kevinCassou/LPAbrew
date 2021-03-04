# !/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:53:27 2015

@author: kevincassou
"""

import numpy as np
import scipy.constants as sc

# plasma parameter

def eDensity(Zat,p):
    """ plasma electron density pressure (fully ionized)
    ---
    pressure: p [mbar]
    atomic number: Z;
    electron density: ne [cm^-3] """

    return 2.429e16*Zat*p

def epressure(Zat,ne):
    """ pressure for a (fully ionized) plasma density
    ---
    pressure: ne [cm^-3]
    atomic number: Z;
    pressure: ne [mbar] """

    return ne/(2.429e16*Zat)

def criticalDensity(wL):
    ''' critical electronic density for a EM wave propagating in a plasma
    ---
    plasma critical density: nc  [cm^-3]
    wavelength: wl [micrometer] '''

    return 1.11485e21/(wL**2)

def ePulsation(ne):
    ''' electronic plasma pulsation
    ---
    electron density : ne [cm^-3];
    e plasma pulsation: wp [rad/s] '''

    return 5.64e4*np.sqrt(ne)

def eplasmawL(ne):
    ''' electronic plasma wavelength
    ---
    electron density : ne [cm^-3];
    e plasma wavelength:  [um] '''

    return 3.3e10/np.sqrt(ne)

# some quick and dirty functionf for LPA... 
# based on cross reading of Lu and Esarey

def E0(ne):
    '''plasma waves electic field
    ne : electron plasma density [cm^-3]
    E0 : [V/m]
    '''
    return 96*np.sqrt(ne)

def Emax(a0):
    ''' peak electric field of a plasma wave. linearly polarized laser. squared temporal profile (1D limit)
    a0 : laser potential normalized vector 
    Emax : E0 units'''
    return (a0**2)/(2*np.sqrt(1+(a0**2)/2.0)) 

def eplasmaNwL(ne,a0):
    ''' non linear plasma wavelength 
    a0 : laser potential normalized vector 
    ne : electron plasma density [cm^-3]
    return eplasmaNwL in [um]
    '''
    lp = eplasmawL(ne)
    e0 = EO(ne)
    emax = Emax(a0)
    rE = emax/e0 
    if rE < 1:
        lnp = lp*(1+(3*rE**2)/16)
    if rE > 1:
        lnp = (2/np.pi)*(rE+1/rE)
    return lnp

def bub_radius(a0,ne):
    ''' Lu scaling law match condition k_p R = kp w_0
    a0 : normalized potential vector amplitude [-]
    electron density : ne [cm^-3];
    plasma wake radius :  
    '''
    return 2*np.sqrt(a0)*eplasmawL(ne)/np.pi

def Pcrit(wL,ne):
    '''Critical power for self focusing in ionized plasma
    wL : laser wavelngth [um]
    ne : electron plasma density [e-/cm^3]
    Pcrit : laser critical power  [GW]
    '''
    omegaL = 2*np.pi/(wL*1e-6)
    omegaP = ePulsation(ne)
    return 17.4*(omegaL/omegaP)**2

def Letching(wL,tL,ne):
    ''' front of the laser is not guided etching rate c (w_p/w_L)^2 ~ 1/P_c
    wL : laser wavelngth [um]
    tL : FWHM laser pulse duration [fs]
    ne : electron plasma density [e-/cm^3] 
    Letching : [m]  
    ''' 
    omegaL = 2*np.pi*sc.c/(wL*1e-6) 
    omegaP = ePulsation(ne)
    return  sc.c*tL*(omegaL**2/omegaP)**2

def Ldeph(ne,wL):
    ''' dephasing lenth in lab frame
    ne : [cm^-3]
    wL : [um]
    ldeph in [um]
    '''
    lp = eplasmawL(ne)
    return lp**3/wL**2

def Ldep (ne,tL,wL):
    ''' depletion length
    ne : [10^18 cm^-3]
    tL : [fs]
    wL : [um]
    ldep in [cm]
    '''
    return 0.03*tL/(ne*wL**2)

