#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   brownian.py
@Time    :   2026/10/22 16:52:20
@Author  :   George Trenins
@Desc    :   Brownian-oscillator spectral density
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
import numpy as np
from typing import Union


class Density(BaseSpectralDensity):

    def __init__(self,
                 eta: float,
                 omega0: float,
                 gamma: float) -> None:
        """Brownian-oscillator spectral density, Garg et al. (1985), https://doi.org/10.1063/1.449017

            J(ω) = η ω ω0⁴ / [(ω² - ω0²)² + γ² ω²]

        corresponding to the system coupled to a single oscillator of frequency ω0
        that is itself damped by Ohmic friction with rate γ.

        Args:
            eta (float): static friction, Λ(0)
            omega0 (float): undamped frequency of the oscillator mode
            gamma (float): damping rate of the oscillator mode; the mode is
                underdamped for γ < 2ω0, critically damped for γ = 2ω0 and
                overdamped for γ > 2ω0
        """
        for name, value in zip(('eta', 'omega0', 'gamma'), (eta, omega0, gamma)):
            if not value > 0:
                raise ValueError(f"Expected a positive value for {name}, instead got {value}")
        self.eta = eta
        self.omega0 = omega0
        self.gamma = gamma
        # s = ω̄², where ω̄ is the damped frequency of the oscillator mode
        s = omega0**2 - gamma**2/4
        if s > 0:
            self._regime = 'underdamped'
            self._wbar = np.sqrt(s)
        elif s < 0:
            self._regime = 'overdamped'
            self._kappa = np.sqrt(-s)
        else:
            self._regime = 'critical'
        # default plotting range used by visualisation/bathplot.py
        self.wplot = self._invert_cdf([0.95])[0]

    def J(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega.
        """
        return omega * self.Lambda(omega)

    def Lambda(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega, divided by omega.
        """
        w0 = self.omega0
        return self.eta * w0**4 / (
            (omega**2 - w0**2)**2 + (self.gamma*omega)**2
        )

    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t,

            K(t) = (η ω0²/γ) exp(-γ|t|/2) [cos(ω̄|t|) + γ/(2ω̄) sin(ω̄|t|)]

        with ω̄ = sqrt(ω0² - γ²/4). For γ > 2ω0, cos and sin become cosh
        and sinh of κt, with κ = sqrt(γ²/4 - ω0²); this branch is evaluated
        as a sum of decaying exponentials to avoid overflow at long times.
        """
        t = np.abs(t)
        gamma = self.gamma
        prefactor = self.eta * self.omega0**2 / gamma
        if self._regime == 'underdamped':
            wbar = self._wbar
            return prefactor * np.exp(-gamma*t/2) * (
                np.cos(wbar*t) + 
                gamma/(2*wbar) * np.sin(wbar*t))
        elif self._regime == 'overdamped':
            kappa = self._kappa
            rate = self.omega0**2 / (gamma/2 + kappa)
            return prefactor * np.exp(-rate*t) * (
                (1 + np.exp(-2*kappa*t))/2 - 
                gamma/(4*kappa) * np.expm1(-2*kappa*t))
        else:
            return prefactor * np.exp(-gamma*t/2) * (1 + gamma*t/2)

    def cdf(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Normalised cumulative spectrum,

            F(ω) = Integrate[ Λ(ω'), {ω', 0, ω}] / Integrate[ Λ(ω'), {ω', 0, ∞}]

        Args:
            omega (float or ndarray): non-negative frequency
        """
        w0 = self.omega0
        gamma = self.gamma
        x = 2*omega / (omega**2 + w0**2)
        if self._regime == 'underdamped':
            G = np.arctanh(self._wbar*x) / self._wbar
        elif self._regime == 'overdamped':
            G = np.arctan(self._kappa*x) / self._kappa
        else:
            G = x
        return 1/2 + np.arctan2(omega**2 - w0**2, gamma*omega)/np.pi + gamma/(2*np.pi) * G

    def _invert_cdf(self, p: np.ndarray) -> np.ndarray:
        """Solve F(ω) = p for the frequencies ω.

        Args:
            p (1d-array): strictly increasing values in the interval (0, 1)

        Returns:
            1d-array: corresponding frequencies, in ascending order
        """
        from scipy.optimize import root_scalar, RootResults
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("Cumulative spectrum values must lie strictly between 0 and 1")
        # upper bound of the root bracket
        hi = self.omega0
        while self.cdf(hi) <= p[-1]:
            hi *= 2
        # absolute tolerance on the scale of the lowest characteristic frequency
        xtol = 1.0e-14 * min(self.omega0, self.omega0**2/self.gamma)
        freqs = []
        prev = 0.0
        for j, target in enumerate(p):
            fun = lambda x: self.cdf(x) - target
            ans: RootResults = root_scalar(fun, method="brentq", bracket=[prev, hi], xtol=xtol)
            if not ans.converged:
                raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
            freqs.append(ans.root)
            prev = ans.root
        return np.asarray(freqs)

    def quadpoints(self, Nmodes: int) -> np.ndarray:
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """
        return self._invert_cdf(
            (np.arange(1, Nmodes+1) - 1/2) / Nmodes
        )

    def exact_reorganisation(self) -> float:
        return 2 * self.eta * self.omega0**2 / self.gamma
