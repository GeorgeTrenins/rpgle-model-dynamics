#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pi.py
@Time    :   2023/04/05 11:00:36
@Author  :   George Trenins
@Desc    :   Potentials for various discretised path integrals.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
from rpmdgle.utils.arrays import slice_along_axis, idx_along_axis
from functools import wraps
from typing import Any


def check_shape(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        q = np.atleast_1d(args[0])
        n = len(self.rpxshape)
        if q.shape[-n:] != self.rpxshape:
            raise ValueError("Input has unexpected shape")
        return f(self, q, *args[1:], **kwargs)
    return wrapper


class BasePI(object):
    """Base class for all discretised path-integral potentials.
    """

    def __init__(self, N, xshape, extPES, *args, **kwargs):
        """Initialise the path-integral potential 

        Args:
            N (int): number of ring-polymer beads
            xshape (tuple): shape of the configuration array for
                a single replica; input arrays are expected to
                have shapes rpshape = preshape + xshape, where preshape[-1]
                ix the number of beads and preshape[:-1] refers to some
                shaped arrangement of non-interacting realisations of the 
                ring-polymer system.
            extPES (subclass of BasePES): the "external", i.e. physical
                potential energy surface
        """

        self.N = int(N)
        if N < 1:
            raise ValueError("Number of beads must be positive")
        self.xshape = xshape
        self.rpxshape = (self.N,)+self.xshape
        self.extPES = extPES
        self.UNITS = extPES.UNITS
        self.hbar = self.UNITS.hbar
        self.mass = extPES.mass*np.ones(self.rpxshape)
        self.sqm = np.sqrt(self.mass)

    @check_shape
    def external_potential(self, x):
        return np.sum(self.extPES.potential(x), axis=-1)
    
    @check_shape
    def external_gradient(self, x):
        return self.extPES.gradient(x)
    
    @check_shape
    def external_force(self, x):
        return self.extPES.force(x)
    
    @check_shape
    def external_hessian(self, x):
        return self.extPES.hessian(x)
    
    @check_shape
    def external_both(self, x):
        pot, grad = self.extPES.both(x)
        return np.sum(pot, axis=-1), grad
    
    @check_shape
    def external_all(self, x):
        pot, grad, hess = self.extPES.all(x)
        return np.sum(pot, axis=-1), grad, hess

    def __getattr__(self, name):
        """Catch calls to potential, returning self.polymer_potential(x) + 
        self.external_potential(x) and similarly for gradient, etc.
        """

        ext_string = "external_"
        funcset1 = {'potential', 'gradient', 'hessian', 'force'}
        funcset2 = {'both', 'all'}
        if name in funcset1:
            return lambda x: (getattr(self, 'polymer_'+name)(x) + 
                              getattr(self, ext_string+name)(x))
        elif name in funcset2:
            def fun(x):
                pi = getattr(self, 'polymer_'+name)(x)
                ext = getattr(self, ext_string+name)(x)
                return tuple(map(lambda x, y : x + y, pi, ext))
            return fun
        else:
            raise AttributeError
    
    @check_shape
    def polymer_potential(self, x):
        pass

    @check_shape
    def polymer_gradient(self, x):
        pass

    @check_shape
    def polymer_hessian(self, x):
        pass

    def polymer_force(self, x):
        return -self.polymer_gradient(x)

    def polymer_both(self, x):
        return self.polymer_potential(x), self.polymer_gradient(x)
    
    def polymer_all(self, x):
        return (self.polymer_potential(x),
                self.polymer_gradient(x), 
                self.polymer_hessian(x))
    

class Ring(BasePI):
    """Standard ring polymer in cartesian coordinates."""

    def __init__(self,
                 N : int,
                 xshape : tuple[int],
                 extPES : Any,
                 beta : float,
                 *args, **kwargs):
        """Initialise the ring-polymer potential

        Args:
            N (int): number of ring-polymer beads
            xshape (tuple): shape of the configuration array for
                a single replica; input arrays are expected to
                have shapes rpshape = preshape + xshape, where preshape[-1]
                ix the number of beads and preshape[:-1] refers to some
                shaped arrangement of non-interacting realisations of the 
                ring-polymer system.
            extPES (subclass of BasePES): the "external", i.e. physical
                potential energy surface
            beta (float): 1/kB*T for the springs
        """
        super().__init__(N, xshape, extPES, *args, **kwargs)
        self.beta = np.asarray(beta, float).item()
        if self.beta <= 0:
            raise ValueError("Temperature is expected to be positive!")
        self.omegaN = N/(self.beta * self.hbar)
        self._rpdim = len(self.rpxshape)
        self._indim = None

    @property
    def indim(self):
        return self._indim
    
    @indim.setter
    def indim(self, ndim):
        update = False
        if self.indim is None:
            update = True
        elif self.indim == ndim:
            update = True
        if update:
            self._indim = ndim
            self._bead_ax = ndim - self._rpdim
            self._to_sum = tuple(range(-1, -ndim+self._bead_ax-1, -1))

    @check_shape
    def polymer_potential(self, x):
        self.indim = x.ndim
        q = self.sqm * x
        ans = np.sum(
            (slice_along_axis(q, self._bead_ax, start=1) -
             slice_along_axis(q, self._bead_ax, end=-1))**2, 
            axis = self._to_sum)
        ans += np.sum(
            (idx_along_axis(q, self._bead_ax, 0) -
             idx_along_axis(q, self._bead_ax, -1))**2, 
            axis = self._to_sum[:-1])
        ans *= 0.5 * self.omegaN**2
        return ans        

    @check_shape
    def polymer_gradient(self, x):
        q = self.mass * x
        grad = 2*q
        # Next bead
        grad_chunk = slice_along_axis(grad, self._bead_ax, start=0, end=-1)
        grad_chunk -= slice_along_axis(q, self._bead_ax, start=1)
        grad_chunk = idx_along_axis(grad, self._bead_ax, -1)
        grad_chunk -= idx_along_axis(q, self._bead_ax, 0)
        # Previous bead
        grad_chunk = slice_along_axis(grad, self._bead_ax, start=1)
        grad_chunk -= slice_along_axis(q, self._bead_ax, start=0, end=-1)
        grad_chunk = idx_along_axis(grad, self._bead_ax, 0)
        grad_chunk -= idx_along_axis(q, self._bead_ax, -1)
        return grad * self.omegaN**2

    @check_shape
    def polymer_hessian(self, x):
        raise NotImplementedError


class RestrainedRing(Ring):

    def __init__(self, shift, k, direction, *args, **kwargs):
        """Add a restraining potential on the centroid of the system,
        V(q0) = HeavisideTheta[-sgn(direction)*(q0-shift)] * 
        k/2 * (q0 - shift) --- CAUTION this class is specific to
        1D system--bath problems, such that the configurations
        x[...,0] refers to the system ring-polymers.

        Args:
            shift (float): point at which the harmonic 
                restraint is activated
            k (float): spring constant for the harmonic restraint
            direction (scalar): "permitted" direction; 
                if `direction` > 0 then the restraint is zero 
                to the right of `shift`, otherwise zero to the 
                left of `shift`.
        """
        self.shift = float(shift)
        self.k = float(k)
        self.sgn = np.sign(float(direction))
        super().__init__(*args, **kwargs)

    def external_potential(self, x):
        pot = super().external_potential(x)
        q = x[...,0]  # system ring-polymer(s)
        q0 = np.mean(q, axis=-1)  # system centroid(s)
        y = q0 - self.shift
        bool_arr = self.sgn * y < 0
        restraint_pot = np.where(bool_arr, 
                                 self.N * self.k * y**2 / 2,
                                 0.0)
        pot += restraint_pot
        return pot
    
    def external_gradient(self, x):
        g = super().external_gradient(x)
        q = x[...,0]  # system ring-polymer(s)
        gq = g[...,0]
        q0 = np.mean(q, axis=-1)  # system centroid(s)
        y = q0 - self.shift
        bool_arr = self.sgn * y < 0
        restraint_grad = np.where(
            bool_arr, 
            self.k * y,
            0.0)
        gq += restraint_grad
        return g
    
    def external_force(self, x):
        return -self.external_gradient(x)
    
    def external_both(self, x):
        pot, g = super().external_both(x)
        q = x[...,0]  # system ring-polymer(s)
        gq = g[...,0]
        q0 = np.mean(q, axis=-1)  # system centroid(s)
        y = q0 - self.shift
        bool_arr = self.sgn * y < 0
        restraint_pot = np.where(bool_arr, 
                                 self.N * self.k * y**2 / 2,
                                 0.0)
        pot += restraint_pot
        restraint_grad = np.where(
            bool_arr, 
            self.k * y,
            0.0)
        gq += restraint_grad[...,None]
        return pot, g
    
    def external_all(self, x):
        raise NotImplementedError
    
    def external_hess(self, x):
        raise NotImplementedError  