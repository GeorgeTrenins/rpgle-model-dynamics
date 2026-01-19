#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   arrays.py
@Time    :   2023/04/05 13:16:17
@Author  :   George Trenins
@Desc    :   Custom utilities for numpy array manipulation
'''

from __future__ import absolute_import, division, print_function
import sys

def slice_along_axis(arr, axis, start=None, end=None, step=1):
    """Return arr[..., slc, ...], where the slice is applied to 
    a specified axis

    Args:
        arr (np.ndarray)
        axis (int)
        start (int, optional): Defaults to None.
        end (int, optional): Defaults to None.
        step (int, optional): Defaults to 1.
    """
    return arr[ (axis % arr.ndim)*(slice(None),) + (slice(start, end, step),)]

def idx_along_axis(arr, axis, idx):
    """Return arr[..., idx, ...], where the idx refers to the specified axis

    Args:
        arr : np.ndarray
        axis : int
        idx : int
    """
    return arr[ (axis % arr.ndim)*(slice(None),) + (idx,)]

def append_dims(arr, ndims=1):
    """Return a view of the input array with `ndims` axes of
    size one appended.
    """
    return arr[(Ellipsis,) + ndims*(None,)] 



def index_in_slice(slice_obj: slice, index: int) -> bool:
    """
    Check if the given index is within the specified slice object.

    Parameters:
    slice_obj (slice): The slice object to check against.
    index (int): The index to check.

    Returns:
    bool: True if the index is within the slice, False otherwise.
    """
    # Create a range object from the slice object's start, stop, and step attributes
    range_obj = range(slice_obj.start if slice_obj.start else 0,
                      slice_obj.stop if slice_obj.stop else sys.maxsize,
                      slice_obj.step if slice_obj.step else 1)

    # Check if the index is within the range object
    return index in range_obj


def string_to_slice(s: str) -> slice:
    """Convert a string representation of a single integer, list of integers, or slice into a slice object

    Args:
        s (str): slice specification, e.g. '1', '0,1,3' or '2:-1:2'

    Returns:
        slice: corresponding slice object
    """
    if s == ":":
        slc = slice(None)
    else:
        try:
            slc = int(s)
        except ValueError:
            if ':' in s:
                slc = slice(*[int(i) if i else None for i in s.split(":")])
            else:
                slc = list(int(i) for i in s.split(","))
    return slc

