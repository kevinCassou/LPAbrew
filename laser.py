# !/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:53:27 2015

@author: kevincassou
"""

import numpy as np
import scipy.constants as sc

# laser parameter

def waist0(wL,f,rL,M2):
    ''' return diffraction limited waist Gaussian beam [mum]
    ---
    laser wavelength : wl [mum]
    focal length : f [m]
    beam radius focusing optic: rL [m]
    beam quality: M2'''
    return M2*wL*f/(np.pi*rL)

def rayleighLength(w0,wL):
    ''' rayleigh length [mum]
    ---
    waist : w0 [mum]
    laser wavelength : wl [mum]'''
    return np.pi*(w0**2/wL)

def waistZ(w0,z,zR):
    '''Gaussiant beam waist [mum]
    ---
    waist: w0 [mum]
    propagation axis coordinate: z [mum]
    rayleigh length :zr [mum]'''
    return w0*np.sqrt(1+(z/zR)**2)

def pulseLengthFWHM(tL):
    '''laser pulse length [=tl]
    ---
    pulse length: tL
    return pulse FWHM'''
    return np.sqrt(2*np.log(2))*tL

def laserIntensity(EL,w0,tL):
    ''' laser intensity in [W/cm^2]
    laser energy: EL [J]
    laser beam waist: w0 [m]
    tL: pulse length [s]'''
    return EL/(np.pi*((w0*1e2)**2)*tL)

def waist0p(EL,IL,tL):
    ''' waist optimum
    EL in J
    laser intensity IL: 1e18 [W/cm-^2]
    laser pulse length: tL [fs]'''
    return 225.38*np.sqrt(EL/(IL*tL))

def a0(IL,wL):
    '''laser potential
    laser intensity: IL 1e18 [W/cm^2]
    laser wavelength: wL [mum]'''
    return 0.855*np.sqrt((IL/1e18)*wL**2)

def aL(wL,r,z,t,epL,IL,tL,w0):
    ''' laser potential
    laser wavelength: wl [mum]
    laser waist w0 [mum]
    propagation axis coordinate :z [m]
    time: t [s]
    laser pulse length: tL [s]'''

    kL = 2*np.pi/(wL*1e-6)
    epsil = np.zeros(z.shape[0])
    profileTrans = np.zeros((r.shape[0],z.shape[0]))
    profileLong = np.zeros(z.shape[0])
    a = np.zeros((r.shape[0],z.shape[0]))
    for i in range(r.shape[0]):
        for j in range(z.shape[0]):
            epsil[j] = z[j]-nu.c0*t+epL
            profileTrans[i,j]= np.exp(-r[i]**2/waistZ(w0,z[j],rayleighLength(w0,wL))**2)
            profileLong[j] = np.exp(-(epsil[j])**2/(nu.c0*tL)**2)
            a[i,j] = a0(IL,wL)*np.cos(kL*(epsil[j]))*profileTrans[i,j]*profileLong[j]
    return a
