#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   splined.py
@Time    :   2024/03/27 11:23:13
@Author  :   George Trenins
@Desc    :   One-dimensional potentials from splining numerical data
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES, check_onedim as check_dims
from rpmdgle import units
import numpy as np
from typing import Optional


class OneDCubic(BasePES):

    def __init__(
            self, 
            x: np.ndarray,
            y: np.ndarray,
            mass: Optional[float] = 1.0, 
            UNITS: Optional[str] = None,
            **kwargs) -> None:
        """One-dimensional PES constructed by cubic spline interpolation of numerical grid data.

        Args:
            x (np.ndarray): position grid
            y (np.ndarray): potential grid
            mass (Optional[float], optional): particle mass. Defaults to 1.0.
            UNITS (Optional[str], optional): name of unit system. Defaults to 'atomic'.

        **kwargs: see scipy.interpolate.CubicSpline
        """

        super().__init__()
        self.xgrid = np.asarray(x)
        self.ygrid = np.asarray(y)
        try:
            self.mass = np.asarray(mass).item()
        except ValueError as err:
            raise ValueError("Expecting a scalar mass for 1D interpolated PES.") from err
        if UNITS is not None:
            self.UNITS = getattr(units, UNITS)()
        self.make_splines(**kwargs)

    def make_splines(self, **kwargs):
        from scipy.interpolate import CubicSpline
        self._pot = CubicSpline(self.xgrid, self.ygrid, **kwargs)
        self._grad = self._pot.derivative(nu=1)
        self._hess = self._pot.derivative(nu=2)


    @check_dims
    def potential(self, x):
        """Potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            energy (float or ndarray): an array is returned is x.ndim > 1; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        ans = self._pot(x)
        if x.ndim > 1:
            return np.reshape(ans, x.shape[:-1])
        else:
            return ans.item()
    
    @check_dims
    def gradient(self, x):
        """Gradient of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            grad (ndarray): same shape as x; the leading dimensions are interpreted as indexing different realisaions of the system.
        """

        return self._grad(x)
    
    @check_dims
    def hessian(self, x):
        """Hessian of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            hess (ndarray): shape (...,1,1); the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        return self._hess(x)[...,None]
