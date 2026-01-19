#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
@File    :   numderiv.py
@Time    :   2022/09/16 17:08:38
@Author  :   George Trenins
@Desc    :   Functions for calculating numerical derivatives
'''

from __future__ import division, print_function, absolute_import
import numpy as np
import itertools


def gradient(fun, x, order=2, h=1.0e-4):
    """Approximate the gradient of a function
    by using central finite differences.
    """
    if order == 2:
        coeffs = [-1/2, 0, 1/2]
        shifts = [-1, 0, 1]
    elif order == 4:
        coeffs = [1/12, -2/3, 0, 2/3, -1/12]
        shifts = [-2, -1, 0, 1, 2]
    else:
        raise ValueError
    x = np.atleast_1d(x).copy()
    flatx = np.reshape(x, -1)
    f0 = fun(x)
    fshape = np.shape(f0)
    ftype = type(f0)
    jac = np.zeros(fshape+x.shape, dtype=ftype)
    flatjac = np.reshape(jac, fshape+(-1,))
    ans = np.zeros(fshape, dtype=ftype)
    for i in range(flatx.size):
        xcache = flatx[i]
        ans *= 0
        for j, shift in zip(coeffs, shifts):
            if j == 0:
                continue
            if shift == 0:
                ans += j*f0
            else:
                flatx[i] = xcache + shift*h
                ans += j*fun(x)
        flatx[i] = xcache
        flatjac[...,i] = ans/h
    return jac

def hessian(fun, x, order=2, h=1.0e-3):
    """Approximate the Hessian of a scalar function
    by using central finite differences.
    """
    if order == 2:
        coeffs1 = [-1/2, 0, 1/2]
        coeffs2 = [1, -2, 1]
        shifts = [-1, 0, 1]
    elif order == 4:
        coeffs1 = [1/12, -2/3, 0, 2/3, -1/12]
        coeffs2 = [-1/12, 4/3, -5/2, 4/3, -1/12]
        shifts = [-2, -1, 0, 1, 2]
    else:
        raise RuntimeError
    x = np.atleast_1d(x).copy()
    flatx = np.reshape(x, -1)
    f0 = fun(x)
    fshape = np.shape(f0)
    ftype = type(f0)
    n = x.size
    hess = np.zeros(fshape+2*(n,), dtype=ftype)
    ans = np.zeros_like(f0)

    for i in range(n):
        xcache = flatx[i]
        # Take care of 2nd order derivative here
        ans *= 0
        for coeffx, shiftx in zip(coeffs2, shifts):
            if coeffx == 0:
                continue
            if shiftx == 0:
                ans += coeffx*f0
            else:
                flatx[i] = xcache + shiftx*h
                ans += coeffx*fun(x)
        hess[...,i,i] = ans/h**2
        # Do mixed derivatives now
        for j in range(i+1,n):
            ycache = np.copy(flatx[j])
            ans *= 0
            for coeffx, shiftx in zip(coeffs1, shifts):
                if coeffx == 0:
                    continue
                flatx[i] = xcache + shiftx*h
                for coeffy, shifty in zip(coeffs1, shifts):
                    if coeffy == 0:
                        continue
                    flatx[j] = ycache + shifty*h
                    if shiftx == shifty == 0:
                        ans = coeffx*coeffy*f0
                    else:
                        ans += coeffx*coeffy*fun(x)
            flatx[j] = ycache
            hess[...,i,j] = ans/h**2
            hess[...,j,i] = hess[...,i,j]
        flatx[i] = xcache
    return hess

########################################

def adiff2(y, x):
    """Calculate the derivative of a scalar function using
    asymmetric "central" difference.

    Args:
        y (ndarray): grid of pre-computed values of size 3. The derivative
        is calculated at the coordinate corresponding to y[1]
        x (ndarray or scalar): if size=3 then the values of arguments of y;
        if size=2 then the step sizes; if scalar then step size of a uniform grid.
    """

    y = np.ravel(np.asarray(y))
    if y.size != 3:
        raise ValueError("Function expects exactly three y-values.")
    if np.isscalar(x):
        steps = np.array([x,x])
    else:
        x = np.ravel(np.asarray(x))
        if x.size == 3:
            steps = np.diff(x)
        elif x.size == 2:
            steps = x.copy()
        else:
            raise ValueError("Grid specification must be a scalar or an array of size 2 or 3.")
    ym, y0, yp = y
    hm, hp = steps
    return y0 * (hm**(-1)-hp**(-1)) + yp*hm / (hp*(hp+hm)) - ym*hp / (hm*(hp+hm))


if __name__ == "__main__":

    def fun(x): return 0.5*(x.item()-1)**2 + np.cos(x.item()**2)
    def jac(x) : return (x-1) - np.sin(x**2) * 2*x
    def hess(x) : return 1 - 4*x**2 * np.cos(x**2) - 2*np.sin(x**2)
    def third(x) : return 8*x**3 * np.sin(x**2) - 12*x*np.cos(x**2)
    def fourth(x) : return 48*x**2 * np.sin(x**2) + 4*(4*x**4 - 3) * np.cos(x**2)
    # x0 = 0.75
    # h = 1.0e-4
    # print(grad2(fun, x0, h=h))
    # xgrid = np.array([x0-h, x0, x0+h])
    # ygrid = fun(xgrid)
    # print(adiff2(ygrid, xgrid))
    # xgrid = np.array([x0-h, x0, x0-3*h])
    # ygrid = fun(xgrid)
    # print(adiff2(ygrid, [h, -3*h]))
    # print(np.diff(ygrid[:2])/np.diff(xgrid[:2]))
    # print(jac(x0))
    h = 2.0e-3
    x0 = 2.0
    print(jac(x0), gradient(fun, x0, h=h, order=2), gradient(fun, x0, h=h, order=4))
    print(hess(x0), hessian(fun, x0, h=h, order=2),
           gradient(
            lambda y: gradient(fun, y, h=h/2, order=2),
            x0, h=h/2, order=2),
           hessian(fun, x0, h=h, order=4),
           gradient(
            lambda y: gradient(fun, y, h=h/2, order=4),
            x0, h=h/2, order=4))
    print(
        third(x0), 
        gradient(
            lambda y: hessian(fun, y, h=h, order=2),
            x0, h=h, order=2),
        gradient(
            lambda y: hessian(fun, y, h=h, order=4),
            x0, h=h, order=4),
            )
    print(
        fourth(x0), 
        hessian(
            lambda y: hessian(fun, y, h=h, order=2),
            x0, h=h, order=2),
        hessian(
            lambda y: hessian(fun, y, h=h, order=4),
            x0, h=h, order=4),
            )