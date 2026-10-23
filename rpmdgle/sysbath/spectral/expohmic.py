#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   expohmic.py
@Time    :   2024/10/02 11:21:35
@Author  :   George Trenins
@Desc    :   Exponentially damped Ohmic spectral density
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
import numpy as np
from typing import Union


class Density(BaseSpectralDensity):

    def __init__(self,
                 eta: float,
                 omega_cut: float) -> None:
        """Exponentially damped Ohmic spectral density,

            J(ω) = η |ω| exp(-|ω|/ω_c)

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
        return self.eta * np.abs(omega) * np.exp(-np.abs(omega)/self.omega_cut)
    
    def Lambda(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Spectral density at frequency omega, divided by omega.
        """
        return self.eta * np.exp(-np.abs(omega)/self.omega_cut)
    
    def K(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Memory-friction kernel at time t. This tends to
        2η * δ(t) as ω_c -> Inf.
        """
        wc = self.omega_cut
        return 2 * self.eta * wc / (1 + (wc*t)**2) / np.pi
    
    def quadpoints(self, Nmodes: int) -> np.ndarray:
        """Calculate the discrete frequencies
        according to Craig and Manolopoulos (2004), https://doi.org/10.1063/1.1850093
        """
        return -self.omega_cut * np.log(
            (np.arange(1, Nmodes+1)-1/2) / Nmodes
        )[::-1]
    
    def exact_reorganisation(self) -> float:
        return (4/np.pi) * self.eta * self.omega_cut
