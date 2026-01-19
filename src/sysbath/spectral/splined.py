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
from rpmdgle.pes._base import BasePES
from rpmdgle.utils.kernel_transforms import shift_lambda_to_kernel
import numpy as np


class Density(BaseSpectralDensity):
    def __init__(self, 
                 PES: BasePES, 
                 Nmodes: int, 
                 eta: float,
                 eps: float,
                 x: np.ndarray,
                 y: np.ndarray,
                 *args, **kwargs) -> None:
        from scipy.interpolate import CubicSpline
        self.eta = eta # scaling of the spectral density
        self.eps = eps # quadrature tolerance
        self.xgrid = np.asarray(x)
        self.ygrid = np.asarray(y)
        self.wmax = np.max(self.xgrid)
        self._cs = CubicSpline(
            self.xgrid, 
            self.eta*self.ygrid,
            axis = kwargs.pop('axis', 0),
            bc_type = kwargs.pop('bc_type', 'not-a-knot'),
            extrapolate = kwargs.pop('extrapolate', None))
        self._integral = self._cs.antiderivative()
        self._reorganization_lambda = 2*self.K(0)
        super().__init__(PES, Nmodes, *args, **kwargs)
        
        
    def Lambda(self, omega):
        y = np.abs(omega)
        return np.where(y > self.wmax, 0.0, self._cs(y))
    
    def J(self, omega):
        return np.abs(omega) * self.Lambda(omega)
    
    def K(self, t):
        t = 1.0*t
        tvec = np.reshape(t, -1)
        ans = np.zeros_like(t)
        ans_flat = np.reshape(ans, -1)
        for i,t in enumerate(tvec):
            ans_flat[i] = shift_lambda_to_kernel(self.Lambda, None, None, None, 0, self.wmax, self.eps, t)[1]
        if ans.ndim == 0:
            return ans.item()
        else:
            return ans
        
    def quadpoints(self):
        """Calculate the discrete frequencies according to https://doi.org/10.1002/jcc.24527
        """     
        from scipy.optimize import root_scalar, RootResults
        freqs = []
        prev = 0.0
        I0 = self._integral(0)
        for j in range(self.Nmodes):
            RHS = I0 + (j+1/2)/self.Nmodes * (np.pi*self.exact_reorganisation()/4)
            fun = lambda x: self._integral(x) - RHS
            ans: RootResults = root_scalar(
                fun, method="bisect", bracket=[prev, self.wmax])
            if not ans.converged:
                raise RuntimeError(f"Failed to find discrete frequency number {j+1}")
            freqs.append(ans.root)
            prev = ans.root
        return np.asarray(freqs)
        
    def exact_reorganisation(self):
        return self._reorganization_lambda
