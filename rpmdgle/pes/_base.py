#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
@File    :   _base.py
@Time    :   2022/11/10 11:05:04
@Author  :   George Trenins
@Desc    :   Defines base potential energy class, adapted from J. O. Richardson's polylib.
'''

from __future__ import absolute_import, division, print_function
from rpmdgle import numderiv, units
from functools import wraps
import numpy as np


def check_onedim(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        q = np.atleast_1d(args[0])
        if q.shape[-1] != 1:
            raise ValueError(f"Input has unexpected dimensions, {q.shape[-1]} != 1")
        return f(self, q, *args[1:], **kwargs)
    return wrapper

def check_ndims(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        q = np.atleast_1d(args[0])
        if q.shape[-1] != self.ndim:
            raise ValueError(f"Input has unexpected dimensions, {q.shape[-1]} != {self.ndim}")
        return f(self, q, *args[1:], **kwargs)
    return wrapper

class BasePES(object):
    """Base class upon which all other PESs should be based.
    By default, gradients and hessians are computed by finite differences
    and "both" and "all" functions simply call the separate routines.
    """

    mass = None
    UNITS = units.atomic()
    gradh = 1.0e-4
    hessh = 1.0e-3

    def potential(self, x):
        return None

    def gradient(self, x):
        return numderiv.grad2(x, self.potential, self.gradh)

    def hessian(self, x):
        return numderiv.hess2(x, self.potential, self.hessh)

    def force(self, x):
        return -self.gradient(x)

    def both(self, x):
        return self.potential(x), self.gradient(x)

    def all(self, x):
        V, G = self.both(x)
        H = self.hessian(x)
        return V, G, H
    
class NullPES(BasePES):

    def potential(self, x):
        return 0.0
    def gradient(self, x):
        return np.zeros_like(x)
    def hessian(self, x):
        return np.zeros(2*(x.size,))
