#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   splined.py
@Time    :   2024/10/02 16:55:41
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import check_onedim
from rpmdgle.pes.splined import OneDCubic as _SplinedPES
import numpy as np
from typing import Optional


class Coupling(_SplinedPES):
    """Interpolated one-dimensional interaction potential.
    """

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
            y (np.ndarray): grid of derivatives of the coupling potential
            mass (Optional[float], optional): particle mass. Defaults to 1.0.
            UNITS (Optional[str], optional): name of unit system. Defaults to 'atomic'.

        **kwargs: see scipy.interpolate.CubicSpline
        """

        super().__init__(x, y, mass=mass, UNITS=UNITS, **kwargs)

    def make_splines(self, **kwargs):
        from scipy.interpolate import CubicSpline
        a, b = self.xgrid[0], self.xgrid[-1]
        period = b-a
        dfdq0 = CubicSpline(self.xgrid, self.ygrid, **kwargs)
        f0 = dfdq0.antiderivative()
        self._mean = (f0(b) - f0(a)) / period
        self._antiderivative_grid = f0(self.xgrid) - self._mean * (self.xgrid - a)
        self._pot = CubicSpline(self.xgrid, self._antiderivative_grid, **kwargs)
        self._grad = self._pot.derivative(nu=1)
        self._hess = self._pot.derivative(nu=2)
        return
    
    @check_onedim
    def potential(self, x):
        """Interaction potential at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            F (float or ndarray): an array is returned is x.ndim > 1; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        ans = self._pot(x) + self._mean*x
        if x.ndim > 1:
            return np.reshape(ans, x.shape[:-1])
        else:
            return ans.item()
        
    @check_onedim
    def gradient(self, x):
        """Gradient of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            grad (ndarray): same shape as x; the leading dimensions are interpreted as indexing different realisaions of the system.
        """

        return self._grad(x) + self._mean