#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   special.py
@Time    :   2023/08/29 11:24:22
@Author  :   George Trenins
@Desc    :   Special functions.
'''

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy as sp

def sinc(x, nu=0):
    """
    Return sinc(x) = sin(x)/x or its nu'th derivative (nu = 0, 1, or 2).

    Args:
        x (float or ndarray): argument
        nu (int, optional): order of derivative (0, 1, or 2)

    Returns:
        float or ndarray: value of the sinc function or its derivative
    """
    y = x/np.pi
    if nu == 0:
        return sp.special.sinc(y)
    else:
        eps = 1.0e-4
        mask = np.abs(y) > eps
        temp = np.where(mask, x, 1.0)
        sinc_x = sp.special.sinc(y)
        if nu == 1:
            ans = np.where(mask, 
                           (np.cos(x)-sinc_x)/temp,
                           x * (x**2 * ((x**2/45360 - 1/840) * x**2 + 1/30) - 1/3)
                           )
        elif nu == 2:
            ans = np.where(mask, 
                           2*sinc_x/temp**2 - 2*np.cos(x)/temp**2 - sinc_x,
                           x**2 * ((x**2/6480 - 1/168) * x**2 + 1/10) - 1/3)
        else:
            raise NotImplementedError(f"Could not calculate the nu = {nu} derivative of sinc.")
        return ans

def logcosh(x):
    """
    Logarithm of cosh(x).

    Args:
        x (float or ndarray): argument

    Returns:
        float or ndarray: log(cosh(x))
    """
    s = np.sign(x)
    x = s*x # real part now positive
    return x + np.log1p(np.exp(-2*x)) - np.log(2.0)

def logsinh(x):
    """
    Logarithm of sinh(x).

    Args:
        x (float or ndarray): argument

    Returns:
        float or ndarray: log(sinh(x))
    """
    s = np.sign(x)
    x = s*x # real part now positive
    ans = np.where(np.real(s) < 0, 1j*np.pi, 0)
    ans += x + np.log1p(-np.exp(-2*x)) - np.log(2.0)
    return ans