#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   special.py
@Time    :   2023/08/29 11:24:22
@Author  :   George Trenins
@Desc    :   Special functions.
'''

from __future__ import print_function, division, absolute_import
import functools
import math
from fractions import Fraction
import numpy as np
from scipy.special import spherical_jn


@functools.lru_cache(maxsize=None)
def _sinc_coeffs(m):
    """
    Exact coefficients of the odd derivatives of sinc in spherical Bessel functions,

        d^(2m+1)/dx^(2m+1) sinc(x) = sum_{p=0}^{m} c[m,p] j_{2p+1}(x),
        c[m,p] = (-1)^(m+p+1) (4p+3) (2m+1)! / (2^(m-p) (m-p)! (2m+2p+3)!!).

    They follow from sinc(x) = (1/2) int_{-1}^{1} exp(ixt) dt: the |c[m,p]|
    are the Legendre coefficients of t^(2m+1) and therefore sum to 1.

    Args:
        m (int): non-negative index; the order of the derivative is 2m+1

    Returns:
        tuple of Fraction: c[m,p] for p = 0, ..., m
    """
    num = math.factorial(2*m + 1)
    return tuple(
        Fraction((-1)**(m+p+1) * (4*p+3) * num,
                 2**(m-p) * math.factorial(m-p) * math.prod(range(1, 2*m+2*p+4, 2)))
        for p in range(m+1))


def sinc(x, nu=0):
    """
    Return sinc(x) = sin(x)/x or its nu'th derivative. Unlike numpy.sinc and
    scipy.special.sinc, there is no factor of pi in the argument.

    Args:
        x (float, complex or ndarray): argument
        nu (int, optional): order of the derivative, nu >= 0. Defaults to 0.

    Returns:
        float, complex or ndarray: nu'th derivative of sinc at x, with the shape
        of x; float64 for real x, complex128 for complex x. For complex x, scipy
        overflows to inf/nan for |Im x| > 700.9 and returns nan for |Re x| > 2^51.

    Raises:
        ValueError: if nu is not a non-negative integer
    """
    if isinstance(nu, bool) or not isinstance(nu, (int, np.integer)) or nu < 0:
        raise ValueError(f"The order of the derivative must be a non-negative integer, got nu = {nu!r}.")
    nu = int(nu)
    x = np.asarray(x)
    x = x.astype(complex if np.iscomplexobj(x) else float)
    zero = (x == 0)
    # scipy negates j_l(x) for x = -a - 0j, so evaluate at Re x >= +0 and use parity
    neg = np.signbit(np.real(x))
    xs = np.where(zero, 1.0, np.where(neg, -x, x))
    if nu == 0:
        ans = spherical_jn(0, xs)
    else:
        m, even = (nu-1)//2, (nu % 2 == 0)
        ans = np.zeros_like(xs)
        for p, c in enumerate(_sinc_coeffs(m)):
            ans += float(c) * spherical_jn(2*p+1, xs, derivative=even)
    ans = np.where(neg, (-1)**nu * ans, ans)
    ans = np.where(zero, (-1)**(nu//2) / (nu+1) if nu % 2 == 0 else 0.0, ans)
    return ans[()]

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