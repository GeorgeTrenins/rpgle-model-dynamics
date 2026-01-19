#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2024/10/02 11:07:00
@Author  :   George Trenins
@Desc    :   Base class for manipulating spectral densities for use in the Caldeira-Leggett Hamiltonian
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES
from rpmdgle.sysbath.coupling import linear
from rpmdgle import units
from typing import Optional
import numpy as np


class BaseSpectralDensity(BasePES):

    def __init__(self, 
                 PES: BasePES, 
                 Nmodes: int, 
                 coupling: Optional[BasePES] = None, 
                 *args, **kwargs):
        """Model spectral densities for one-dimensional systems
        with bilinear coupling.

        Args:
            PES (object): child of BasePES
            Nmodes (int): number of bath modes
        """
        self.PES = PES
        UNITS = kwargs.get('UNITS', 'atomic') 
        self.UNITS = getattr(units, UNITS)()
        if PES.UNITS.__class__.__name__ != UNITS:
            raise RuntimeError(
                "The spectral density and the potential appear to have "
                f"different unit systems, {PES.UNITS.__class__.__name__} "
                f"and {UNITS}, respectively")
        self.Nmodes = Nmodes
        if coupling is None:
            coupling = linear.Coupling(UNITS = PES.UNITS.__class__.__name__)
        if (self.UNITS.__class__.__name__ != coupling.UNITS.__class__.__name__):
            raise RuntimeError(f"The coupling and the potential appear to have different unit systems, "
                               f"{coupling.UNITS.__class__.__name__} and "
                               f"{self.UNITS.__class__.__name__}, respectively.")
        self.coupling = coupling
        self.c, self.w, self.bath_mass = self.bath_params()
        self.ww = self.w*self.w
        self.mww = self.bath_mass*self.ww
        self.mass = np.concatenate((np.atleast_1d(PES.mass), self.bath_mass))

    def quadpoints(self):
        raise NotImplementedError
    
    def bath_params(self):
        m = np.ones(self.Nmodes)
        m *= self.PES.mass # same as system mass
        w = self.quadpoints()
        kappa = np.sqrt(self.exact_reorganisation()/(2*self.Nmodes))
        c = kappa * np.sqrt(m) * w
        return c, w, m
    
    def quadrature(self, f):
        """Calculate the quadrature approximation to an integral
        Integrate[ J(ω) f(ω), {ω, 0, ∞}]

        Args:
            f (1d-array): input function evaluated at the quadrature points.
        """

        return (np.pi/2) * np.sum(self.c**2/(self.bath_mass*self.w) * f, axis=-1)
    
    def l_quadrature(self, f):
        """Calculate the quadrature approximation to an integral
        Integrate[ Λ(ω) f(ω), {ω, 0, ∞}] for Λ(ω) = J(ω)/ω

        Args:
            f (1d-array): input function evaluated at the quadrature points.
        """

        return (np.pi/2) * np.sum(self.c**2/(self.bath_mass*self.w**2) * f, axis=-1)
    
    def exact_reorganisation(self):
        raise NotImplementedError
    
    def reorganisation_energy(self):
        """Use quadrature to calculate the reorganisation energy,
        λ = (4/π) * Integrate[ J(ω)/ω, {ω, 0, ∞}]
        """
        return (4/np.pi) * self.quadrature(1/self.w)
    
    def J(self, omega):
        raise NotImplementedError
    
    def Lambda(self, omega):
        raise NotImplementedError
    
    def K(self, t):
        raise NotImplementedError
    
    def bath_eq(self, q):
        """Return the equilibrium positions of the bath modes
        given the system coordinate q.
        """
        return self.c*self.coupling.potential(q)[...,None] / self.mww
    
    def potential(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        y = (r - self.c*self.coupling.potential(q)[...,None] / self.mww)
        pot = np.sum(self.mww/2 * y**2, axis=-1)
        pot += self.PES.potential(q)
        return pot

    def gradient(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        g = np.zeros_like(x)
        gq = g[...,:f]
        gr = g[...,f:]
        gq[:] = self.PES.gradient(q)
        y = (r - self.c*self.coupling.potential(q)[...,None] / self.mww)
        dgdq = self.coupling.gradient(q)
        gr[:] = self.mww * y
        gq[:] -= np.sum(self.c * y * dgdq, axis=-1, keepdims=True)
        return g
    
    def both(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        g = np.zeros_like(x)
        gq = g[...,:f]
        gr = g[...,f:]
        pot, gq[:] = self.PES.both(q)
        y = (r - self.c*self.coupling.potential(q)[...,None] / self.mww)
        dgdq = self.coupling.gradient(q)
        gr[:] = self.mww * y
        gq[:] -= np.sum(self.c * y * dgdq, axis=-1, keepdims=True)
        pot = pot + np.sum(self.mww/2 * y**2, axis=-1)
        return pot, g
    
    

class BaseNumericalSpectralDensity(BaseSpectralDensity):

    def __init__(self, PES, Nmodes, wmax, eps, *args, **kwargs):
        # Upper integration limit in the frequency domain
        self.wmax = wmax
        # Quadrature tolerance parameter
        self.eps = eps
        self._reorganization_lambda = 2*self.K(0)
        super().__init__(PES, Nmodes, *args, **kwargs)
        
    def quadpoints(self):
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """
        from scipy.integrate import quad
        from scipy.optimize import root_scalar, RootResults
        freqs = []
        prev = 0.0
        for j in range(self.Nmodes):
            fun = lambda x: (
                quad(self.Lambda, 0.0, x, epsabs=self.eps, limit=1000)[0] -
                (j+1/2) / self.Nmodes * (np.pi*self.exact_reorganisation() / 4))
            ans: RootResults = root_scalar(fun, method="bisect", bracket=[prev, self.wmax])
            if not ans.converged:
                raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
            freqs.append(ans.root)
            prev = ans.root
        return np.asarray(freqs)
    
    def exact_reorganisation(self):
        return self._reorganization_lambda

