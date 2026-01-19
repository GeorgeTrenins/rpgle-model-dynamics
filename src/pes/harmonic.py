#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   harmonic.py
@Time    :   2023/03/22 15:12:42
@Author  :   George Trenins
@Desc    :   One- and multi-dimensional harmonic potentials
'''

from rpmdgle.pes._base import BasePES, check_ndims as check_dims
from typing import Union
from rpmdgle import units
import numpy as np


class PES(BasePES):

    def __init__(self, hess=[[1.0,],], mass=1.0, shift=0.0, bias=0.0, UNITS="atomic"):
        """Harmonic potential, specified as

            V(x) = 0.5 * (x - shift)·hess·(x-shift) - bias

        Args:
            hess (2D-array, optional): hessian (not mass weighted).
                Defaults to [[1.0,],].
            mass (float or 1D-array, optional): particle mass(es).
                Defaults to 1.0.
            shift (float or 1D-array, optional): Location of well minimum.
                Defaults to 0.0.
            bias (float, optional): ─V(shift), see above. Defaults to 0.0.
        """

        self.UNITS = getattr(units, UNITS)()
        if np.isscalar(bias):
            self.bias = bias
        else:
            raise TypeError("Bias expected to be a scalar!")
        self._check_shapes(hess, mass, shift)
        # Final shape checks
        self.hess = np.atleast_2d(hess)
        if self.hess.shape != (self.ndim, self.ndim):
            raise ValueError("Unexpected shape of Hessian")
        self.mass = np.atleast_1d(mass)
        if self.mass.ndim != 1:
            raise ValueError("Mass expected to be scalar or 1D")
        self.shift = np.atleast_1d(shift)
        if self.shift.ndim != 1:
            raise ValueError("Shift expected to be scalar or 1D")
        self.get_nmfreqs()

    def _check_shapes(self, *args):
        # Check broadcasting
        to_check = list(args)
        try:
            shape = np.broadcast_arrays(*to_check)[-1].shape
        except:
            raise ValueError(
                "Parameters could not be broadcast to each other.")
        # Record dimensionality
        if shape == tuple():
            self.ndim = 1
        else:
            self.ndim = max(shape)

    def get_nmfreqs(self):
        """Diagonalise the mass-weighted hessian and store 
        result in self.nmfreqs.
        """
        sqm = np.sqrt(self.mass)
        if sqm.ndim != 1:
            raise ValueError("Mass expected to be scalar or 1D")
        mw_hess = self.hess / sqm[:,None] / sqm[None,:]
        self.nmfreqs = np.emath.sqrt(np.linalg.eigvalsh(mw_hess))

    @check_dims
    def potential(self, x):
        """Potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., self.ndim)

        Returns:
            energy (float or ndarray): an array is returned is x.ndim > 1; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        y = x - self.shift
        ans = np.einsum('...i,ij,...j->...', y, self.hess, y)/2
        return ans - self.bias
    
    @check_dims
    def gradient(self, x):
        """Gradient of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., self.ndim)

        Returns:
            grad (ndarray): same shape as x; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        y = x - self.shift
        ans = np.einsum('ij,...j->...i', self.hess, y)
        return ans
    
    @check_dims
    def hessian(self, x):
        """Hessian of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., self.ndim)

        Returns:
            hess (ndarray): shape (..., self.ndim,self.ndim); the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        to_tile = x.shape[:-1]+(1,1)
        return np.tile(self.hess, to_tile)
    
class nmPES(PES):

    def __init__(self, omega=1.0, mass=1, shift=0, bias=0, UNITS="atomic"):
        """Harmonic potential in normal mode coordinates, specified as

            V(x) = 0.5 * (x - shift)·diag[m*omega**2]·(x-shift) - bias

        Args:
            hess (1D-array, optional): normal mode frequencies.
                Defaults to 1.0.
            mass (float or 1D-array, optional): particle mass(es).
                Defaults to 1.0.
            shift (float or 1D-array, optional): Location of well minimum.
                Defaults to 0.0.
            bias (float, optional): ─V(shift), see above. Defaults to 0.0.
        """
    
        self._check_shapes(omega, mass, shift)
        self.nmfreqs = np.ones(self.ndim) * np.atleast_1d(omega).flatten()
        self._mw2 = mass * self.nmfreqs**2
        hess = np.diag(self._mw2)
        super().__init__(hess, mass, shift, bias, UNITS=UNITS)
    
    def eigenenergies(self, n: Union[int, list[int], tuple[int]]) -> float :
        if isinstance(n, int):
            n = [n]
        assert len(n) == self.nmfreqs.ndim, "Must specify one quantum number for every dimension"
        ans = 0.0
        for i, n_i in enumerate(n):
            ans += self.UNITS.hbar * self.nmfreqs[i] * (n_i+0.5)
        return ans

    def get_nmfreqs(self):
        pass

    @check_dims
    def potential(self, x):
        """Potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., self.ndim)

        Returns:
            energy (float or ndarray): an array is returned is x.ndim > 1; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        y = x - self.shift
        ans = np.sum(self._mw2*y**2 / 2, axis=-1)
        return ans - self.bias
    
    @check_dims
    def gradient(self, x):
        """Gradient of the potential energy at position `x`.

        Args:
            x (scalar or ndarray): if array, must have shape (..., self.ndim)

        Returns:
            grad (ndarray): same shape as x; the leading dimensions are interpreted as indexing different realisaions of the system.
        """
        
        y = x - self.shift
        return self._mw2 * y
    
    
if __name__ == "__main__":
    HO = PES(shift = [-1, -2], hess = [[1, 0.5], [0.5, 1]])
    print(HO.ndim)
    print(HO.nmfreqs)
    x = np.arange(4*3*2).reshape((4,3,2))
    print(HO.potential(x))
    print(HO.potential(x))
    print(HO.gradient(x))
    print(HO.hessian(x))
    

    HO = nmPES([1.0, 2.0])
    print(HO.ndim)
    print(HO.nmfreqs)
    x = np.arange(4*3*2).reshape((4,3,2))
    print(HO.potential(x))
    print(HO.gradient(x))
    print(HO.hessian(x))
