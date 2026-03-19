#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   kde.py
@Time    :   2024/09/19 14:39:10
@Author  :   George Trenins
@Desc    :   Custom tools for kernel density estimation
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from typing import Union, Optional


########### Kernels ###########
def gaussian_kernel(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = (x-mu)/sigma
    return np.exp(-y**2 / 2) / np.sqrt(2*np.pi) / sigma

def tophat_kernel(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)
    return 1/(2*sigma) * np.where(y < 1, 1.0, 0.0)

def triangular_kernel(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)
    return 1/sigma * np.where(y < 1, 1-y, 0.0)

def epanechnikov_kernel(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = ((x-mu)/sigma)**2
    return 3/(4*sigma) * np.where(y < 1, 1-y, 0.0)

def tricube_kernel(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)**3
    return 70/(81*sigma) * np.where(y < 1, (1-y)**3, 0.0)

####### Kernel density estimate #######
def kde(data: np.ndarray, 
        grid: np.ndarray, 
        bw: float,
        kernel: Optional[str] = "gaussian", 
        axis: Optional[Union[int, tuple[int]]] = -1) -> Union[float, np.ndarray]:
    if kernel == "gaussian":
        f = gaussian_kernel
    elif kernel == "tophat":
        f = tophat_kernel
    elif kernel == "triangular":
        f = triangular_kernel
    elif kernel == "epanechnikov":
        f = epanechnikov_kernel
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'.")
    return np.mean(f(data, grid, bw), axis=axis)

    



