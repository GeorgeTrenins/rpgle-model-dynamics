#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2024/10/02 11:07:00
@Author  :   George Trenins
@Desc    :   Base classes for the spectral densities of harmonic baths in the Caldeira-Leggett Hamiltonian
'''


from __future__ import print_function, division, absolute_import
import numpy as np


class BaseSpectralDensity(object):
    """Continuous spectral density J(ω) of a harmonic bath. Parameters are
    in the internal units of the system PES and include the system mass, so
    that the static friction η = Λ(0) has units of mass/time.

    The discrete bath is built from one or more spectral densities by
    rpmdgle.sysbath.bath.CaldeiraLeggett.
    """

    # Upper end of the default frequency range for plots of the spectrum
    wplot = None

    def J(self, omega):
        """Spectral density at frequency omega.
        """
        raise NotImplementedError

    def Lambda(self, omega):
        """Spectral density at frequency omega, divided by omega.
        """
        raise NotImplementedError

    def K(self, t):
        """Memory-friction kernel at time t.
        """
        raise NotImplementedError

    def quadpoints(self, Nmodes):
        """Discrete bath frequencies according to https://doi.org/10.1002/jcc.24527

        Args:
            Nmodes (int): number of bath modes

        Returns:
            1d-array: the frequencies ω_j, j = 1, ..., Nmodes, in ascending order, such that
            Integrate[ Λ(ω), {ω, 0, ω_j}] / Integrate[ Λ(ω), {ω, 0, ∞}] = (j - 1/2) / Nmodes
        """
        raise NotImplementedError

    def exact_reorganisation(self):
        """Reorganisation energy, λ = (4/π) * Integrate[ Λ(ω), {ω, 0, ∞}]
        """
        raise NotImplementedError


class BaseNumericalSpectralDensity(BaseSpectralDensity):

    def __init__(self, wmax, eps):
        # Upper integration limit in the frequency domain
        self.wmax = wmax
        self.wplot = wmax
        # Quadrature tolerance parameter
        self.eps = eps
        self._reorganization_lambda = 2*self.K(0)

    def quadpoints(self, Nmodes):
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """
        from scipy.integrate import quad
        from scipy.optimize import root_scalar, RootResults
        freqs = []
        prev = 0.0
        for j in range(Nmodes):
            fun = lambda x: (
                quad(self.Lambda, 0.0, x, epsabs=self.eps, limit=1000)[0] -
                (j+1/2) / Nmodes * (np.pi*self.exact_reorganisation() / 4))
            ans: RootResults = root_scalar(fun, method="bisect", bracket=[prev, self.wmax])
            if not ans.converged:
                raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
            freqs.append(ans.root)
            prev = ans.root
        return np.asarray(freqs)

    def exact_reorganisation(self):
        return self._reorganization_lambda
