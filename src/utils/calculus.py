# File: calculus.py
"""
Created on Sun Jun  3 15:24:27 2018

@author: gt317

Tools for numerical integration and differentiation.

"""

from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from scipy import special

def GL_quad(n, lo = -1.0, hi = 1.0):
    """Calculate `n` points and weights for Gauss-Legendre quadrature 
    from `lo` to `hi`

    Args:
        n (int): number of points
        lo (float, optional): lower integration bound. Defaults to -1.
        hi (float, optional): upper integration bound. Defaults to 1.

    Returns:
        roots, weights (ndarrays)
    """

    roots, weights = special.roots_legendre(n)
    roots *= (hi-lo)/2 
    roots += (hi+lo)/2
    weights *= (hi-lo)/2
    return roots, weights


class FFTFourierIntegral(object):
    """
    Numerical evaluation of Fourier integrals of the kind
        I = Integrate[Exp[-2*pi*i*k*x] f(x), {x, a, b}]
    using the fast Fourier transform algorithm. See Chapter 13.9
    of Numerical Recipes for FORTRAN 77

    """

    _W_c = np.array([1.0, 0.0, -11.0/720, 23.0/15120,
                     -139.0/1814400, 37.0/14968800,
                     -49139.0/871782912000,
                     7549.0/7846046208000])

    _real_alpha_c = np.array([
        -2.0/3, 7.0/24, -1.0/6, 1.0/24,
         1.0/45, -7.0/180, 1.0/45, -1.0/180,
         103.0/15120, 5.0/3456, -5.0/6048, 5.0/24192,
        -169.0/226800, -7.0/259200, 1.0/64800, -1.0/259200,
        761.0/19958400, 7.0/22809600, -1.0/5702400, 1.0/22809600,
        -1009.0/817296480, -1.0/424569600, 1.0/742996800,
        -1.0/2971987200,
        319.0/11321856000, 1.0/76640256000,
        -1.0/134120448000, 1.0/536481792000,
        -192487.0/400148356608000,
        -1.0/18292496302080,
        1.0/32011868528640,
        -1.0/128047474114560
        ]).reshape((-1,4))


    _imag_alpha_c = np.array([
         2.0/45, 7.0/72, -7.0/90, 7.0/360,
         2.0/105, -1.0/168, 1.0/210, -1.0/840,
        -8.0/2835, 11.0/72576, -11.0/90720, 11.0/362880,
         86.0/467775, -13.0/5987520, 13.0/7484400, -13.0/29937600,
         -4.0/552825, 5.0/249080832, -1.0/62270208, 1.0/249080832,
         124.0/638512875, -17.0/130767436800, 17.0/163459296000,
         -17.0/653837184000,
         -124.0/32564156625, 19.0/30487493836800,
         -19.0/38109367296000, 19.0/152437469184000,
         106.0/1856156927625,
         -1.0/434446787174400,
         1.0/543058483968000,
         -1.0/2172233935872000
    ]).reshape((-1,4))


    def __init__(self, lo, hi, npts, pad):
        """
        Initialise the integrator with the grid over which the
        data is defined and the amount of padding applied to the integrand
        (results in more closely-spaced data in k-space.)

        Input:
            lo, hi (scalar or array-like): the lower and upper bounds on the grid.
            npts (int or array-like): the number of points along each direction
            pad (int or array-like): amount of padding applied to the input data.


        """

        lo_arr, hi_arr, npts_arr, pad_arr = [np.asarray(val).reshape(-1)
                                             for val in (lo, hi, npts, pad)]
        self.hh = (hi_arr-lo_arr)/(npts_arr-1)
        self.npts = (np.ones_like(self.hh)*npts_arr).astype(int)
        # Padded points:
        pad_arr = np.where(pad_arr < 2, 1, pad_arr)
        self.ppts = np.asarray(pad_arr*self.npts, dtype=int).reshape(-1)
        # Padded array bounds
        self.lo = np.ones_like(self.hh)*lo_arr
        self.hi = np.ones_like(self.hh)*hi_arr
        # Indices of the arrays to be returned
        self.ilo = (self.ppts-self.npts)//2
        self.ihi = self.ilo+self.npts

        # FFT frequencies and correction factors
        self.freqs = []
        self.theta = []
        self.W = []
        self.alpha = []
        self.exp_ka = []
        self.exp_kb = []
        for i, ppt in enumerate(self.ppts):
            self.freqs.append(np.fft.fftshift(
                              np.fft.fftfreq(ppt))[self.ilo[i]:self.ihi[i]])
            self.theta.append(-2*np.pi*self.freqs[-1])
            self.freqs[-1] /= self.hh[i]
            W, alpha = self.calc_coeffs(self.theta[-1])
            self.W.append(W)
            self.alpha.append(alpha)
            self.exp_ka.append(
                self.hh[i]*np.exp(1j*self.theta[-1] *
                                  self.lo[i]/self.hh[i])
                )
            self.exp_kb.append(
                np.exp(1j*np.asarray(self.theta[-1]) *
                       (self.hi[i] - self.lo[i])/self.hh[i])
                )

    @classmethod
    def calc_coeffs(cls, theta):
        alpha_real = np.ones((4,len(theta)))
        alpha_imag = np.ones_like(alpha_real)
        W = np.ones_like(theta)
        Re_alpha = np.reshape(cls._real_alpha_c, (-1,4,1))
        Im_alpha = np.reshape(cls._imag_alpha_c, Re_alpha.shape)

        W *= cls._W_c[0]
        alpha_real *= Re_alpha[0]
        alpha_imag *= Im_alpha[0]

        theta_sq = theta**2
        temp = theta_sq.copy()
        alpha_real += Re_alpha[1]*temp
        alpha_imag += Im_alpha[1]*temp

        for i in range(2, len(cls._W_c)):
            temp *= theta_sq
            alpha_real += Re_alpha[i]*temp
            alpha_imag += Im_alpha[i]*temp
            W += cls._W_c[i]*temp

        alpha_imag *= theta
        return W, alpha_real + 1j*alpha_imag

    def __call__(self, a, axes=-1):
        """
        Return the integral of Exp[-2*pi*i*k*x]*f(x) over the specified
        axis of the grid (default -1).

        """

        arr = a.copy()
        axarr = list(range(arr.ndim))
        for axis in np.asarray(axes).reshape(-1):
            axarr[axis], axarr[-1] = axarr[-1], axarr[axis]
            wkspace = np.transpose(arr, axarr)
            ans = np.fft.fft(wkspace, n=self.ppts[axis], axis=-1)
            ans = np.fft.fftshift(
                    ans, axes=-1)[...,self.ilo[axis]:self.ihi[axis]]
            ans *= self.W[axis]

            endpoints = np.zeros_like(ans)
            # Calculate the sum a0*h0 + a1*h1 + a2*h2 + a3*h3
            for i in range(4):
                endpoints += wkspace[...,i:i+1]*self.alpha[axis][i]
            ans += endpoints
            # Calculate conj(a0)*h{M} + conj(a1)*h{M-1} + ...
            endpoints *= 0
            for i in range(4):
                endpoints += wkspace[...,-(i+1):-(i+2):-1] * \
                             np.conj(self.alpha[axis][i])
            endpoints *= self.exp_kb[axis]
            ans += endpoints
            ans *= self.exp_ka[axis]
            arr = np.transpose(ans, axarr)
            axarr[axis], axarr[-1] = axarr[-1], axarr[axis]

        return arr