#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   polynomial.py
@Time    :   2023/04/05 10:26:20
@Author  :   George Trenins
@Desc    :   One-dimensional polynomial potentials
'''

from rpmdgle.pes._base import BasePES, check_onedim as check_dims
from rpmdgle import units
import numpy as np

class PES(BasePES):

    def __init__(self, mass=1.0, coeffs=None, x0=0.0, UNITS=None):
        """A one-dimensional polynomial potential 
            V(x) = c0 + c1*(x-x0) + c2*(x-x0)**2 + ...

        Args:
            mass (float or 1D-array, optional): particle mass(es).
                Defaults to 1.0.
            coeffs (sequence, optional): list of coefficients
        """
        
        if coeffs is None:
            coeffs = [0.0, 0.0, 0.5]
        if UNITS is not None:
            self.UNITS = getattr(units, UNITS)()
        self.coeffs = np.atleast_1d(coeffs).astype(float)
        self.x0 = float(x0)
        if self.coeffs.ndim != 1:
            raise RuntimeError("Expecting a 1D array of coefficients in polynomial PES.")
        if len(self.coeffs) == 0:
            raise RuntimeError("Expecting a non-empty list of coefficients for polynomial PES.")
        try:
            self.mass = np.asarray(mass).item()
        except ValueError as err:
            raise ValueError("Expecting a scalar mass for polynomial PES.") from err
        self._pot = np.polynomial.polynomial.Polynomial(self.coeffs)
        self._grad = self._pot.deriv(m=1)
        self._hess = self._pot.deriv(m=2)


    @check_dims
    def potential(self, x):
        """Potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            energy (float or ndarray): an array is returned is x.ndim > 1; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        ans = self._pot(x-self.x0)
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

        return self._grad(x-self.x0)
    
    @check_dims
    def hessian(self, x):
        """Hessian of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., 1)

        Returns:
            hess (ndarray): shape (...,1,1); the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        return self._hess(x-self.x0)[...,None]