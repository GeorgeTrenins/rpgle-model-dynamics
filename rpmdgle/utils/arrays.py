#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   arrays.py
@Time    :   2023/04/05 13:16:17
@Author  :   George Trenins
@Desc    :   Custom utilities for numpy array manipulation
'''

from __future__ import absolute_import, division, print_function

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



def index_in_slice(slice_obj: slice, index: int, length: int | None = None) -> bool:
    """
    Check whether seq[index] is one of the elements selected by seq[slice_obj].

    Args:
        slice_obj (slice): slice with the usual Python conventions
        index (int): index into seq; negative values count from the end
        length (int, optional): len(seq). If None, index, start and stop must
            be None or non-negative and the step positive, which guarantees that
            the answer does not depend on len(seq). Defaults to None.

    Returns:
        bool: True if index is a valid index into seq and is selected by the slice

    Raises:
        ValueError: if the step is zero, or if length is None and index, start,
            stop or step is negative
    """
    if length is None:
        start = slice_obj.start or 0
        stop = index + 1 if slice_obj.stop is None else slice_obj.stop
        step = 1 if slice_obj.step is None else slice_obj.step
        if min(index, start, stop, step) < 0:
            raise ValueError(f"index_in_slice({slice_obj}, {index}) may depend on the length of the sequence; pass it as `length`.")
        return index in range(start, stop, step)
    if not -length <= index < length:
        return False
    return index % length in range(length)[slice_obj]


def string_to_slice(s: str) -> int | list[int] | slice:
    """Convert a string representation of a single integer, list of integers, or slice into an index

    Args:
        s (str): slice specification, e.g. '1', '0,1,3' or '2:-1:2'

    Returns:
        int | list[int] | slice: an int for '1', a list for '0,1,3', a slice for '2:-1:2' or ':'
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

