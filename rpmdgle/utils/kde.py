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
from typing import Callable, Union, Optional


Kernel = Callable[[np.ndarray, Union[float, np.ndarray], float], np.ndarray]
# kernels available to `kde`, keyed by name
KERNELS: dict[str, Kernel] = {}

def register(f: Kernel) -> Kernel:
    """Make the kernel `f` available to `kde` under its name without the `_kernel` suffix.
    """
    KERNELS[f.__name__.removesuffix("_kernel")] = f
    return f


########### Kernels ###########
@register
def gaussian_kernel(x: np.ndarray, mu: Union[float, np.ndarray], sigma: float) -> np.ndarray:
    y = (x-mu)/sigma
    return np.exp(-y**2 / 2) / np.sqrt(2*np.pi) / sigma

@register
def tophat_kernel(x: np.ndarray, mu: Union[float, np.ndarray], sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)
    return 1/(2*sigma) * np.where(y < 1, 1.0, 0.0)

@register
def triangular_kernel(x: np.ndarray, mu: Union[float, np.ndarray], sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)
    return 1/sigma * np.where(y < 1, 1-y, 0.0)

@register
def epanechnikov_kernel(x: np.ndarray, mu: Union[float, np.ndarray], sigma: float) -> np.ndarray:
    y = ((x-mu)/sigma)**2
    return 3/(4*sigma) * np.where(y < 1, 1-y, 0.0)

@register
def tricube_kernel(x: np.ndarray, mu: Union[float, np.ndarray], sigma: float) -> np.ndarray:
    y = np.abs((x-mu)/sigma)**3
    return 70/(81*sigma) * np.where(y < 1, (1-y)**3, 0.0)

####### Kernel density estimate #######
def kde(data: np.ndarray,
        grid: Union[float, np.ndarray],
        bw: float,
        kernel: str = "gaussian",
        axis: Optional[Union[int, tuple[int]]] = -1) -> Union[float, np.ndarray]:
    """Kernel density estimate, the mean of kernel(data, grid, bw) over `axis`.

    Args:
        data (ndarray): samples, broadcast against `grid` by the caller
        grid (ndarray): points at which to evaluate the density
        bw (float): bandwidth
        kernel (str, optional): name of a kernel in `KERNELS`. Defaults to "gaussian".
        axis (int or tuple of int, optional): sample axis or axes to average over. Defaults to -1.

    Raises:
        ValueError: if `kernel` is not in `KERNELS`
    """
    try:
        f = KERNELS[kernel]
    except KeyError:
        raise ValueError(f"Unknown kernel type '{kernel}'; expected one of {sorted(KERNELS)}.") from None
    return np.mean(f(data, grid, bw), axis=axis)
