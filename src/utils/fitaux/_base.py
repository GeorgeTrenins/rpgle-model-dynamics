#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   _base.py
@Time    :   2024/12/18 11:39:36
@Author  :   George Trenins
@Desc    :   Return scaled auxiliary variable propagation coefficients computed from interpolating the data stored in reduced units.
'''


from __future__ import print_function, division, absolute_import
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from scipy.interpolate import UnivariateSpline
from typing import Optional, Union


def get_nm_freqs(
        beta: float, 
        hbar: float,
        N: int,
        n: Optional[int] = None):
    """Return the frequency of the n-th normal mode of a free ring polymer
    of N beads at a reciprocal temperature beta = 1/kB*T
    """
    if n: 
        indices = np.asarray(n, dtype=int)
        assert np.all(np.abs(indices) <= N//2), "invalid normal-mode index!"
    else:
        indices = [0,] + [j for i in range(1,N//2+1) for j in [-i, i]]
        if N%2 == 0:
            indices.pop(-1)
        indices = np.array(indices, dtype=int)
    return 2*N / (beta*hbar) * np.sin(np.abs(indices) * np.pi/N)


class BaseOneAuxParam(object):

    def __init__(
            self, 
            datafile: str, 
            column: str, 
            A: Optional[float] = 0.0, 
            wc: Optional[float] = 1.0, 
            b: Optional[float] = 1.0,
            smoothing: Optional[float] = 1.0e-4):
        """Load the fitting coefficients calculated for a single auxiliary variable
        (really a pair of variables if oscillatory), construct a smoothing spline with
        weights A * EXP( -omega / wc) + b and return a callable object that gives
        the value of the coefficient for an arbitrary positive normal-mode frequency
        by either interpolating or extrapolating the data.

        Args:
            datafile (str): name of the CSV file where the data is stored
            column (str): name of the column that corresponds to the desired parameter
            A (float): amplitude of the exponential used in defining the weight
            wc (float): decay rate of the weighting exponential
            b (float): asymptotic value of the weighting function
            smoothing (float): strength of smoothing
        """
        self.df = (pd.read_csv(
          datafile, 
          delimiter=',', 
          header=0,
          index_col=None).iloc[1:]).set_index('nmfreq', inplace=False)
        self.col = column
        weights = A*np.exp(-self.df.index / wc) + b
        self.spl = UnivariateSpline(
            self.df.index, 
            self.df[self.col], 
            w=weights, k=3, s=smoothing)
        self.wmax = np.max(self.df.index)

    def __call__(self, 
                 w: Union[list[float], np.ndarray, float]) -> Union[float, np.ndarray]:
        frequencies = np.atleast_1d(w)
        coefficients = np.zeros_like(frequencies)
        flatfreqs = np.reshape(frequencies, -1)
        flatcoeffs = np.reshape(coefficients, -1)
        mask = frequencies < self.wmax
        flatcoeffs[mask] = self.spl(flatfreqs[mask])
        mask = ~mask
        flatcoeffs[mask] = self.extrapolate(flatfreqs[mask])
        if np.isscalar(w):
            return coefficients.item()
        else:
            return coefficients
        
    def extrapolate(self, w):
        raise NotImplementedError
    

class MultiAuxParams(object):

    def __init__(
            self, 
            datafile: str, 
            column: str, 
            A: Optional[float] = 0.0, 
            wc: Optional[float] = 1.0, 
            b: Optional[float] = 1.0,
            smoothing: Optional[float] = 1.0e-4,
            sort: Optional[str] = "none",
            sorting_cutoff: Optional[float] = 0.0
            ) -> None:
        """Load the fitting coefficients calculated for multiple auxiliary variable
        construct a smoothing spline with weights A * EXP( -omega / wc) + b and return a callable object that gives the value of the coefficient for a normal-mode frequency
        falling within the range spanned by the data file.

        `sort` may be one of "none", "ascending" or "descending"; if `sort` is not "none", values at frequencies beyond the `sorting_cutoff` are sorted in ascending/descending order (a hack for smoother interpolation).

        Args:
            naux (int): number of auxiliary variables (pairs of aux. vars. in the case of oscillatory modes)
            datafile (str): name of the CSV file where the data is stored
            column (str): name of the column that corresponds to the desired parameter
            A (float): amplitude of the exponential used in defining the weight
            wc (float): decay rate of the weighting exponential
            b (float): asymptotic value of the weighting function
            smoothing (float): strength of smoothing
        """
            
        self.load_dataframe(datafile)
        self.extract_data(column, sort=sort, cutoff=sorting_cutoff)
        weights = A*np.exp(-self.nmfreqs / wc) + b
        self.build_splines(weights, smoothing)

    def load_dataframe(self, datafile):
        df_ = pd.read_csv(
            datafile, 
            delimiter=',',
            header=None, 
            index_col=None)
        new_labels = pd.MultiIndex.from_arrays(
            [df_.iloc[0], np.asarray(df_.iloc[1].astype(float), dtype=int)],
            names=['parameter', 'index'])
        self.df = df_.set_axis(new_labels, axis=1).iloc[2:]
        self.df.set_index(('nmfreq', 0), inplace=True)
        self.df.index.name = 'nmfreq'
        return
    
    def extract_data(self, column, sort="none", cutoff=0.0):
        self.nmfreqs = self.df.index.to_numpy(dtype=float)
        self.data = self.df[column]
        self.naux = len(self.data.columns)
        if self.naux == 1:
            print(f"WARNING: detected a single-variable fitting, better use dedicated `OneAux...` classes, which implement intelligent extrapolation for high normal-mode frequencies.")
        if sort == "none":
            return
        elif sort in {"ascending", "descending"}:
            data = self.data.to_numpy(dtype=float)
            mask = self.nmfreqs > cutoff
            masked_data: np.ndarray = data[mask]
            if sort == "ascending":
                masked_data.sort(axis=0)
            else:
                masked_data[::-1].sort(axis=0)
            data[mask] = masked_data
            self.data.loc[:,:] = data
            return
        else:
            raise ValueError(f"Unknown sorting type {sort}.")

    def build_splines(self, weights, smoothing):
        self.splines = []
        for i in range(self.naux):
            data = self.data[i].to_numpy(dtype=float)
            self.splines.append(
                UnivariateSpline(
                    self.nmfreqs, data, 
                    w=weights, k=3, s=smoothing))
        return
    
    def __call__(self, 
                 w: Union[list[float], np.ndarray, float]) -> np.ndarray:
        frequencies = np.reshape(w, -1)
        ans = np.zeros((len(frequencies), self.naux))
        if np.any(frequencies > np.max(self.nmfreqs)):
            raise RuntimeError(f"Maximum requested normal-mode frequency is outside the fitting range for naux = {self.naux}. Consider reducing the number of auxiliary variables for this normal-mode index.")
        for i,s in enumerate(self.splines):
            ans[:,i] = s(frequencies)
        return np.asarray(ans)