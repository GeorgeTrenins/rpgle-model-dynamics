#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   exptanh.py
@Time    :   2024/10/02 16:53:42
@Author  :   George Trenins
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES, check_onedim
from rpmdgle import units
import numpy as np
from typing import Optional, Union


class Coupling(BasePES):

    ndim = 1  # one-dimensional
    
    def __init__(self, 
                 eps1: Optional[float] = -1.0, 
                 eps2: Optional[float] = 0.0,
                 delta: Optional[Union[str,float]] = 1.0,
                 UNITS: Optional[str] = "atomic"):
        """Coupling function for separable friction calculations,
            g(q) = q * (1 + ε1 exp[-q^2/2*δ^2] + ε2 tanh[q/δ])

        Args:
            eps1 (float, optional): Defaults to -1.0.
            eps2 (float, optional): Defaults to 0.0.
            delta (float, optional): Defaults to 1.0.
            UNITS (str, optional): Defaults to "atomic".

        """
        self.UNITS: units.SI = getattr(units, UNITS)()
        self.eps1 = eps1
        self.eps2 = eps2
        self.delta = self.UNITS.str2base(delta)
        for foo in [eps1, eps2, delta]:
            if not np.isscalar(foo):
                raise ValueError("ExpTanhG expects scalar parameters, got and array instead: {:s}".format(foo.__repr__()))
        
    @check_onedim
    def potential(self, x):
        y = x/self.delta
        ans = self.eps2 * np.tanh(y)
        ans += self.eps1 * np.exp(-y**2/2)
        ans += 1.0
        ans *= x
        if x.ndim > 1:
            return np.reshape(ans, x.shape[:-1])
        else:
            return ans.item()
        
    @check_onedim
    def gradient(self, x):
        y = x/self.delta
        ans = self.eps1 * np.exp(-y**2/2) * (1-y**2) + 1
        ans += self.eps2 * (np.tanh(y) + y/np.cosh(y)**2)
        return ans
