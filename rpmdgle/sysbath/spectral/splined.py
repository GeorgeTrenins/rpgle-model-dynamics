#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   splined.py
@Time    :   2024/10/02 12:16:25
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
from rpmdgle.utils.kernel_transforms import shift_lambda_to_kernel
import numpy as np
from typing import Optional


class Density(BaseSpectralDensity):
    def __init__(self,
                 eta: float,
                 eps: float,
                 x: np.ndarray,
                 y: np.ndarray,
                 axis: int = 0,
                 bc_type = 'not-a-knot',
                 extrapolate: Optional[bool] = None) -> None:
        """Spectral density interpolated from grid data, Λ(ω) = η * spline(ω)
        for ω up to the end of the grid and zero beyond.

        Args:
            eta (float): scaling of the spectral density
            eps (float): quadrature tolerance
            x (1d-array): frequency grid, starting at ω = 0
            y (1d-array): shape of Λ on the grid

        axis, bc_type and extrapolate: see scipy.interpolate.CubicSpline
        """
        from scipy.interpolate import CubicSpline
        self.eta = eta # scaling of the spectral density
        self.eps = eps # quadrature tolerance
        self.xgrid = np.asarray(x)
        self.ygrid = np.asarray(y)
        self.wmax = np.max(self.xgrid)
        # default plotting range used by visualisation/bathplot.py
        self.wplot = self.wmax
        self._spline = CubicSpline(
            self.xgrid,
            self.eta*self.ygrid,
            axis = axis,
            bc_type = bc_type,
            extrapolate = extrapolate)
        self._antiderivative = self._spline.antiderivative(nu=1)
        self._reorganization_lambda = 2*self.K(0)

        
    def Lambda(self, omega):
        y = np.abs(omega)
        return np.where(y > self.wmax, 0.0, self._spline(y))
    
    def J(self, omega):
        return np.abs(omega) * self.Lambda(omega)
    
    def K(self, t):
        t = 1.0*t
        tvec = np.reshape(t, -1)
        ans = np.zeros_like(t)
        ans_flat = np.reshape(ans, -1)
        for i,t in enumerate(tvec):
           ans_flat[i] = np.asarray(
               shift_lambda_to_kernel(
                   self.Lambda, None, None, None, 0, self.wmax, self.eps, t)[1]
               ).item()
        if ans.ndim == 0:
            return ans.item()
        else:
            return ans
        
    def quadpoints(self, Nmodes):
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """
        from scipy.optimize import root_scalar, RootResults
        value_at_w0 = self._antiderivative(0.0)
        cumulative_spectrum = lambda x: np.nan_to_num(
            self._antiderivative(x, extrapolate=False) - value_at_w0, nan=0.0)
        exact_reorganisation = 4/np.pi * cumulative_spectrum(self.wmax)  # Eq. (2.5)
        weight = np.pi * exact_reorganisation / (4 * Nmodes)
        freqs = []
        prev = 0.0
        for j in range(Nmodes):
            fun = lambda x: ( cumulative_spectrum(x) - (j+1/2) * weight)  # Eq. (2.6)
            ans: RootResults = root_scalar(fun, method="bisect", bracket=[prev, self.wmax])
            if not ans.converged:
                raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
            freqs.append(ans.root)
            prev = ans.root
        return np.asarray(freqs)
        
    def exact_reorganisation(self):
        return self._reorganization_lambda
