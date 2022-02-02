# !/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Aug 26 11:29:27 2020

@author: kevincassou
"""

# few functions for gas flow and conductances in vacuum system 

import numpy as np
import scipy.constants as sc


# conversion

mbar2Pa = 1e2 # conversion mbar to Pa
m2cm = 1e2

# gas viscosity 

def viscosity(molecule):
    """
    return dynamic viscosity at 20°C from tables of Handbook of chemistry. Crane 1988 
    unit [Pa.s]
    """
    if molecule == 'N2':
        visc = 1.756E-5 
    if molecule == 'He':
        visc = 1.96E-5 
    if molecule == 'Air':
        visc =1.87E-5 
    if molecule =='H2':
        visc = 8.7454468E-6 
    return visc



def mfp(pressure,temperature,molecule):
    """ return mean free path for a given pressure and temperature
     l*p=kT/((2)^0.5*pi*dmol**2) (handbook vacuum technology) in [m]
     ---
     pressure : Pa
     temperature : kelvin
     gas molecule : 'H2'.'He' or 'N2'
     He: 0.218e-10 m
     H2: 0.22e-9 m
     N2: 367e-9 m
     """
    if molecule == 'H2':
        dmol = 0.22e-9
    if molecule == 'He':
        dmol = 0.218e-9 #dmol = 0.49e-10 / lafferty p. 40 
    if molecule == 'N2':
        dmol = 367e-9
    if molecule == 'air':
        dmol = 0.37e-9

    return sc.k*temperature/((2)**0.5*np.pi*(dmol**2)*pressure)

def knudsen(pressure,temperature, molecule,d):
    """return the knudsen number for a given pipe diameter and pressure
    ref: Jousten Handbook of vacuum technology
    ---
    pressure: Pa
    temperature: kelvin
    gas molecule : 'H2'.'He' or 'N2'
    d : m
    """
    l = mfp(pressure,temperature,molecule)
    return l/d

def reynolds(qpv,molecule,temperature,d):
    """ return the Reynolds number for a given molecule. gas flow. and pipe diameter
    ref : Jousten Handbook of vacuum technology
    ---
    flow qpv : Pa.m^3/s
    molecule: 'H2'.'He'. 'air' or 'N2'
    d : m
    """
 
    return 32*qpv/(np.pi**2*viscosity(molecule)*cs(molecule,temperature)**2*d)

def cs(molecule,temperature):
    """ return acoustics gas speed [m/s] as function of temperature
    ref: Jousten Handbook of vacuum technology
    ---
    kappa : factor depends on gas type . monoatomic. diatomic etc..
    temperature : kelvin
    """
    if molecule == 'H2':
        Mm = 2.0159e-3
        kappa = 1.4
    if molecule == 'He':
        Mm = 4.0026022e-3
        kappa = 1.667
    if molecule == 'N2':
        Mm = 28.0135e-3
        kappa = 1.4
    if molecule == 'air':
        Mm = 28.96562e-3
        kappa = 1.4
    return (kappa*(sc.R/Mm)*temperature)**0.5


def condApe_lam(d,p1,p2):
    """ conductance of circular aperture at 20°C in the viscous regime
        d: in [m]
        p1 : inlet pressure [Pa]
        p2 : outlet pressure [Pa]
        return conductance in l/s.
    """    
    return 20 * np.pi * (d*m2cm/2.)**2/(1-p2/p1) 

def condApe_mol(d):
    """ conductance of circular aperture at 20°C in the molecular regime 
        ref : CNRS formation permanente - ecole du vide
        ---
        d: in [m]
        return conductance in l/s.
    """    
    return 11.6 * np.pi * (d*m2cm/2.)**2

def condPipe_lam(d,L,p1,p2,species='He'):
    """ conductance of a circular pipe d<<L in laminar regime in m^3/s
    ref: loi de Poiseuille 
    ---
    d : pipe diameter [m]
    L : pipe length [m]
    p1 : inlet pressure [Pa]
    p2 : outlet pressure [Pa]
    species : gas by default Helium 
    return conductance in m^3/s.
    """
    A = (np.pi/256)*(1/viscosity(species))*(d**4)/L
    return A*(p1+p2)

def condPipe_mol(d,L):
    """ conductance of circular pipe d<<L in molecular regime in l/s
    ref : CNRS formation permanente - ecole du vide
    ---
    d : pipe diameter [m]
    L : pipe length [m]
    return conductance in l/s.
    """
    return 12.4e4*(d**3)/L

def flow(p1,p2,C):
    """ gas flow calculation through a tube or object of conductance C
    p1 : inlet pressure [Pa]
    p2 : outlet pressure [Pa]
    C : m^3/s 
    """
    return C*(p1-p2)

def Reynolds_test(q,d,verbose=False):
    """ return if flow is laminar or not 
    q: Pa.m^3/s
    d: m 
    """
    if q < 1.42e5*d:
        r = True
        mess = 'laminar flow'
    else:
        r = False 
        if q < 2.62e5*d:
            mess = 'transient flow'
        else:
            mess = 'turbulent flow'

    if verbose==True:
        print(mess) 
    return r


