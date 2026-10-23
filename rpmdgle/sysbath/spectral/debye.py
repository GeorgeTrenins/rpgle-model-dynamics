#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   debye.py
@Time    :   2024/10/02 12:13:54
@Author  :   George Trenins
@Desc    :   Debye spectral density
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
import numpy as np
from typing import Union


class Density(BaseSpectralDensity):
    def __init__(self,
                 eta: float,
                 omega_cut: float) -> None:
        """Debye spectral density,

            J(ω) = η ω ω_c² / (ω² + ω_c²)

        Args:
            eta (float): static friction, Λ(0)
            omega_cut (float): cutoff frequency
        """
        self.eta = eta
        self.omega_cut = omega_cut
        # default plotting range used by visualisation/bathplot.py
        self.wplot = 5*omega_cut

    def J(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega.
        """
        return self.eta * self.omega_cut**2 * omega / (
            omega**2 + self.omega_cut**2
        )
    
    def Lambda(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega, divided by omega.
        """
        return self.eta * self.omega_cut**2 / (
            omega**2 + self.omega_cut**2
        )
    
    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t. This tends to
        2η * δ(t) as ω_c -> Inf.
        """
        return self.eta * self.omega_cut * np.exp(-self.omega_cut*np.abs(t))
    
    def quadpoints(self, Nmodes: int) -> np.ndarray:
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """
        return self.omega_cut * np.tan(
            np.pi * (2*np.arange(1, Nmodes+1) - 1) / (4*Nmodes)
        )
    
    def exact_reorganisation(self) -> float:
        return 2 * self.eta * self.omega_cut
