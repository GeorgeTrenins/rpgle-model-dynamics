#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   kernel_transform.py
@Time    :   2024/01/18 09:34:14
@Author  :   George Trenins
@Desc    :   Functions for transforming a classical memory-friction kernel to kernels for non-centroid normal modes in RPMD simulations.
'''

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.integrate import quad
from scipy.fftpack import dct, idct

__all__ = ["shift_spectral_to_kernel", "kernel_to_spectral"]

def shift_spectral_to_kernel(J, UNITS, T, P, n, wmax, eps, tvec):
    """Convert the classical spectral density to the memory-friction kernel for a non-centroid normal mode.

    Args:
        J (callable): classical spectral density
        UNITS (object): unit system
        T (float): temperature in Kelvin
        P (int) : number of beads
        n (int): index of normal mode
        wmax (float): upper bound for the integral over frequency
        eps (float): absolute error margin for the integration (a small fraction of the zero-time value of the friction)
        tvec (ndarray): array of times for which to calculate the kernel
    """
    beta = UNITS.betaTemp(T)
    wP = P/(beta*UNITS.hbar)
    wn = 2*wP * np.sin(np.pi*n/P)
    wn2 = wn**2
    fun = lambda w: J(np.sqrt(w**2-wn2))/w
    K = list()
    tvec = np.ravel(tvec)
    for t in tvec:
        K.append(quad(fun, wn*(1+1.0e-8), np.sqrt(wn2+wmax**2), weight='cos', wvar=t, epsabs=eps, limit=1000)[0])
    return wn, 2/np.pi * np.asarray(K)

def shift_lambda_to_kernel(Lambda, UNITS, T, P, n, wmax, eps, tvec):
    """Convert the classical spectral density to the memory-friction kernel for a non-centroid normal mode.

    Args:
        Lambda (callable): classical spectral density divided by frequency
        UNITS (object): unit system
        T (float): temperature in Kelvin
        P (int) : number of beads
        n (int): index of normal mode
        wmax (float): upper bound for the integral over frequency
        eps (float): absolute error margin for the integration (a small fraction of the zero-time value of the friction)
        tvec (ndarray): array of times for which to calculate the kernel
    """
    if n == 0:
        wn = 0.0
        K = list()
        tvec = np.ravel(tvec)
        for t in tvec:
            K.append(quad(Lambda, wn, wmax, weight='cos', wvar=t, epsabs=eps, limit=1000)[0])
    else:
        beta = UNITS.betaTemp(T)
        wP = P/(beta*UNITS.hbar)
        wn = 2*wP * np.sin(np.pi*n/P)
        wn2 = wn**2
        def fun(w):
            freq2 = w**2 - wn2
            if freq2 < 0:
                return 0.0
            else:
                freq = np.sqrt(freq2)
                return (freq/w) * Lambda(freq)
        K = list()
        tvec = np.ravel(tvec)
        for t in tvec:
            K.append(quad(fun, wn, np.sqrt(wn2+wmax**2), weight='cos', wvar=t, epsabs=eps, limit=1000)[0])
    return wn, 2/np.pi * np.asarray(K)

def timefreqs(times):
    # This also works to get times from freqs
    return np.arange(len(times)) * np.pi/times[-1]

def kernel_to_lambda(kvec, tvec, hh):
    return dct(kvec, type=1) * hh/2

def kernel_to_spectral(kvec, tvec, hh):
    wvec = timefreqs(tvec)
    return wvec*dct(kvec, type=1) * hh/2

def shift_spectral(J, wvec, UNITS, T, P, n):
    beta = UNITS.betaTemp(T)
    wP = P/(beta*UNITS.hbar)
    wn = 2*wP * np.sin(np.pi*n/P)
    tmp = wvec**2 - wn**2
    mask = tmp > 0
    tmp[np.logical_not(mask)] = 0
    return np.where(mask, J(np.sqrt(tmp)), 0.0)    

def spectral_to_kernel(jvec, wvec, axis=-1):
    mask = wvec > 1.0e-20
    dw = wvec[1]-wvec[0]
    tmp = np.where(mask, wvec, 1)
    ftk = np.where(mask, jvec/tmp, 0.0)
    return idct(ftk, type=1, axis=axis) * dw / (np.pi)

def lambda_to_kernel(lvec, wvec, axis=-1):
    dw = wvec[1]-wvec[0]
    return idct(lvec, type=1, axis=axis) * dw / (np.pi)