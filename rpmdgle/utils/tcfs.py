#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   tcfs.py
@Time    :   2023/04/26 12:12:30
@Author  :   George Trenins
@Desc    :   On-the-fly calculation of time-correlation functions.
'''

from __future__ import print_function, division
import numpy as np
import copy
from typing import Optional
from rpmdgle.utils.arrays import slice_along_axis, append_dims
     
def correlate(
        A: np.ndarray, 
        B: np.ndarray, 
        axis: Optional[int] = 0) -> np.ndarray:
    
    assert A.ndim == B.ndim
    shape_a = A.shape
    len_a = shape_a[axis]
    shape_b = B.shape
    len_b = shape_b[axis]
    
    if len_a >= len_b:
        a_wkspace = slice_along_axis(A, axis, end=len_b)
        len_fft = 2*len_b
        len_tcf = len_b
        norm_tcf = np.arange(len_b, 0, -1, dtype=int)
    else:
        len_tcf = len_b
        len_fft = len_a + len_b
        a_wkspace = A
        norm_tcf = np.arange(len_b, 0, -1, dtype=int)
        norm_tcf = np.where(norm_tcf > len_a, len_a, norm_tcf)

    dims_to_append = np.arange(B.ndim-1, -1, -1, dtype=int)[axis]
    norm_tcf = append_dims(norm_tcf, dims_to_append)
    ftA = np.fft.rfft(a_wkspace, axis=axis, n=len_fft)
    ftB = np.fft.rfft(B, axis=axis, n=len_fft)
    np.conj(ftA, out=ftA)
    ftB *= ftA
    out = slice_along_axis(
        np.fft.irfft(ftB, axis=axis, n=len_fft),
        axis=axis, start=0, end=len_tcf) / norm_tcf
    return out


def time_averaged_correlation(
    A: np.ndarray, 
    B: np.ndarray, 
    lag: int,
    axis: int = 0
) -> np.ndarray:
    
    out_shape = list(A.shape)
    stride = lag + 1
    out_shape[axis] = stride
    out = np.zeros(out_shape)
    #
    a_indices = np.arange(stride, dtype=int)
    a_shape = copy.copy(out_shape)
    a_shape[axis] = stride
    a_wkspace = np.empty(a_shape)
    #
    b_indices = np.arange(stride+lag, dtype=int)
    b_shape = copy.copy(out_shape)
    b_shape[axis] = stride+lag
    b_wkspace = np.empty(b_shape)
    #
    sample_counter = 0
    while True:
        try:
            a_wkspace[:] = np.take(A, a_indices, axis=axis)
            b_wkspace[:] = np.take(B, b_indices, axis=axis)
        except IndexError:
            break
        out += slice_along_axis(
            correlate(a_wkspace, b_wkspace, axis=axis), 
            axis=axis, start=0, end=lag+1)
        sample_counter += 1
        a_indices += stride
        b_indices += stride
    out /= sample_counter
    return out
