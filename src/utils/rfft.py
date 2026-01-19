#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rfft.py
@Time    :   2023/04/12 11:03:18
@Author  :   George Trenins
@Desc    :   Convenience class for discrete Fourier transform
'''

from __future__ import print_function, division, absolute_import
from scipy import fftpack
from numpy import fft
from functools import wraps
from rpmdgle.utils.arrays import slice_along_axis


def specify_out(fxn):
    @wraps(fxn)
    def wrapper(self, in_arr, out=None, **kwargs):
        if out is None:
            return fxn(self, in_arr, **kwargs)
        else:
            out[...] = fxn(self, in_arr, **kwargs)
            return out
    return wrapper


class RealFFT(object):

    def __init__(self, axis=-1):
        """A wrapper for discrete Fourier transforms.

        Args:
            axis (int, optional): array dimension along which to calculate the transform. Defaults to -1.
        """
        self.axis = axis
 
    @specify_out
    def crfft(self, in_arr, ksize=None):
        """
        Calculate the Fourier transform of a real dataset with
        complex output, truncated if the specified number of points
        is smaller in Fourier space

        Args:
            in_arr (np.ndarray): input array
            out (np.ndarray, optional): output destination
            ksize (int, optional): number of Fourier modes to keep in output
        """
        ans = fft.rfft(in_arr, axis=self.axis)
        if ksize is None:
            return ans
        else:
            return slice_along_axis(ans, axis=self.axis, start=None, end=ksize)

    @specify_out
    def cirfft(self, in_arr, rsize=None):
        """
        Calculate the inverse Fourier transform from complex input.
        The input is padded with zeros if the number of points in
        r-space exceeds that in k-space

        Args:
            in_arr (np.ndarray): input array
            out (np.ndarray, optional): output destination
            rsize (int, optional): number of r-space points
        """
        return fft.irfft(in_arr, n=rsize, axis=self.axis)

    @specify_out
    def rfft(self, in_arr, ksize=None):
        """
        Calculate the Fourier transform of a real dataset with
        real output. Output is truncated if the number of points
        is smaller in Fourier space

        Args:
            in_arr (np.ndarray): input array
            out (np.ndarray, optional): output destination
            ksize (int, optional): number of Fourier modes to keep in output
        """

        ans = fftpack.rfft(in_arr, axis=self.axis)
        if ksize is None:
            return ans
        else:
            return slice_along_axis(ans, axis=self.axis, start=None, end=ksize)

    @specify_out
    def irfft(self, in_arr, rsize=None):
        """
        Calculate the inverse Fourier transform from real input.
        Input array is padded with zeros if the number of points
        is greater in r-space

        Args:
            in_arr (np.ndarray): input array
            out (np.ndarray, optional): output destination
            rsize (int, optional): number of r-space points
        """
        return fftpack.irfft(in_arr, n=rsize, axis=self.axis)