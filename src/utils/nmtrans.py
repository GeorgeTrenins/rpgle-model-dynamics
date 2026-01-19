#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   nmtrans.py
@Time    :   2023/04/12 11:26:12
@Author  :   George Trenins
@Desc    :   Transformations between bead and normal-mode coordinates.
             This uses the normalisation convention from the Matsubara paper (DOI: 10.1063/1.4916311)
'''

from __future__ import print_function, division, absolute_import
import numpy as np
import opt_einsum as oe
from rpmdgle.utils.rfft import RealFFT


class BaseNMTransform(object):

    def __init__(self, nbeads, nmodes=None, bead_ax=-1):
        """The base class for carrying out normal-mode transformations.

        Args:
            nbeads (int): number of ring-polymer beads
            nmodes (int, optional): number of non-zero ring-polymer modes. Defaults to None.
            bead_ax (int, optional): axis along which the bead index varies
        """

        self.axis = bead_ax
        self.nbeads = nbeads
        self.nmodes = nbeads if nmodes is None else nmodes
        if self.nbeads < 1: 
            raise ValueError("Number of beads must be positive")
        if self.nmodes > self.nbeads:
            raise ValueError("Number of modes cannot be greater than the number of beads")
        indices = [0,] + [j for i in range(1,self.nbeads//2+1) for j in [-i, i]]
        if self.nbeads%2 == 0:
            indices.pop(-1)
        self.indices = np.array(indices, dtype=int)
        self.get_nm_freqs()

    def get_nm_freqs(self):

        self.nm_freqs = 2*np.sin(np.abs(self.indices) * np.pi/self.nbeads)
        self.mats_freqs = 2*np.abs(self.indices)

    def cart2mats(self, cart, mats=None):
        raise NotImplementedError
    

    def mats2cart(self, mats, cart=None, overwrite_mats=False):
        raise NotImplementedError


class MatMulNormalModes(BaseNMTransform):

    def __init__(self, nbeads, nmodes=None, bead_ax=-1):
        """A matrix-multiplication class for normal-mode transformations.

        Args:
            nbeads (int): number of ring-polymer beads
            nmodes (int, optional): number of non-zero ring-polymer modes. Defaults to None.
            bead_ax (int, optional): axis along which the bead index varies
        """

        super().__init__(nbeads, nmodes=nmodes, bead_ax=bead_ax)
        self.construct_transformation_matrices()
        self._path_dict = dict()

    def construct_transformation_matrices(self):

        T = []
        beads = np.arange(self.nbeads)
        for idx in self.indices:
            if idx == 0:
                T.append(np.ones_like(beads)/self.nbeads)
            elif idx > 0:
                T.append(np.sqrt(2)/self.nbeads * np.sin(
                    2*np.pi*beads*idx/self.nbeads
                ))
            else:
                T.append(np.sqrt(2)/self.nbeads * np.cos(
                    2*np.pi*beads*idx/self.nbeads
                ))
        if self.nbeads%2 == 0:
            T[-1] /= np.sqrt(2)

        self.forward_matrix = np.asarray(T)[:self.nmodes]
        self.backward_matrix = np.transpose(self.forward_matrix) * self.nbeads

    def get_einsum_indices(self, ndim):
        in_idx = list(range(ndim))
        bead_dim = in_idx[self.axis]
        mat_idx = [ndim, bead_dim]
        out_idx = in_idx.copy()
        out_idx[self.axis] = ndim
        return mat_idx, in_idx, out_idx

    def cart2mats(self, cart, mats=None):
        mat_idx, in_idx, out_idx = self.get_einsum_indices(cart.ndim)
        if mats is None:
            mats = oe.contract(self.forward_matrix, mat_idx, cart, in_idx, out_idx)
        else:
            oe.contract(self.forward_matrix, mat_idx, cart, in_idx, out_idx, out=mats)
        return mats

    def mats2cart(self, mats, cart=None, overwrite_mats=False):
        mat_idx, in_idx, out_idx = self.get_einsum_indices(mats.ndim)
        if cart is None:
            cart = oe.contract(self.backward_matrix, mat_idx, mats, in_idx, out_idx)
        else:
            oe.contract(self.backward_matrix, mat_idx, mats, in_idx, out_idx, out=cart)
        return cart
    

class FFTNormalModes(BaseNMTransform,RealFFT):

    def __init__(self, nbeads, nmodes=None, bead_ax=-1):
        """An FFT-based class for normal-mode transformations.

        Args:
            nbeads (int): number of ring-polymer beads
            nmodes (int, optional): number of non-zero ring-polymer modes. Defaults to None.
            bead_ax (int, optional): axis along which the bead index varies
        """

        super().__init__(nbeads, nmodes=nmodes, bead_ax=bead_ax)
        self.calculate_dft_norm()
            
    def calculate_dft_norm(self):
        _norm = np.sign(self.indices).astype(float)
        _norm[0] = 1.0
        _norm[1:-1] *= -np.sqrt(2.0)
        if self.nbeads%2 == 1 and self.nbeads > 1:
            _norm[-1] *= -np.sqrt(2.0)
        elif self.nbeads%2 == 0:
            _norm[-1] *= -1
        _norm /= self.nbeads
        self._norm = _norm[:self.nmodes]

    def _shaped_norm(self, arr):
        shape = [slice(None),]
        for i in range(1,len(arr.shape[self.axis:])):
            shape.append(None)
        return self._norm[tuple(shape)]

    def cart2mats(self, cart, mats=None):
        mats = self.rfft(cart, mats, ksize=self.nmodes)
        mats *= self._shaped_norm(cart)
        return mats

    def mats2cart(self, mats, cart=None, overwrite_mats=False):
        if not overwrite_mats:
            mats = mats / self._shaped_norm(mats)
        else:
            mats /= self._shaped_norm(mats)
        cart = self.irfft(mats, cart, rsize=self.nbeads)
        return cart
    


if __name__ == "__main__":

    for nbeads, nmodes in zip([1, 5, 15, 16, 16], [1, 5, 9,  16, 4]):
        arr = np.random.uniform(size=(4,nbeads,3,3))
        matrix_trans = MatMulNormalModes(nbeads, nmodes, bead_ax=1)
        fft_trans = FFTNormalModes(nbeads, nmodes, bead_ax=1)

        nm_matrix = matrix_trans.cart2mats(arr)
        nm_fft = fft_trans.cart2mats(arr)
        cart_matrix = matrix_trans.mats2cart(nm_matrix)
        cart_fft = fft_trans.mats2cart(nm_fft)
        np.testing.assert_allclose(nm_matrix, nm_fft)
        np.testing.assert_allclose(cart_matrix, cart_fft)
        if nbeads == nmodes:
            np.testing.assert_allclose(arr, cart_matrix)

