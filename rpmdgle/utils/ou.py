#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ou.py
@Author  :   George Trenins, Hannah Bertschi
@Desc    :   Analytical results for multi-dimensional Ornstein-Uhlenbeck processes.
'''

from __future__ import print_function, division, absolute_import
import numpy as np
from typing import Union, Optional

def wnle_cqq(
        t: Union[float, np.ndarray], 
        beta: float,
        omega: float,
        tau: float,
        m: Optional[float] = 1):
    """
    Position auto-correlation function for a harmonic oscillator at thermal equilibrium,
    under the action of a white-noise Langevin equation thermostat (WNLE).

    Args:
        t (float or ndarray): time(s) for which to compute the covariance matrix
        beta (float): reciprocal temperature, 1/kB*T
        omega (float): frequency of the harmonic oscillator
        tau (float): reciprocal of the friction, tau = 1/gamma
        m (float, optional): mass of the harmonic oscillator

    Returns:
        ndarray: covariance matrix
    """
    Omega = np.sqrt(np.abs(omega**2 - 1/(2*tau)**2))
    eps = 1.0e-4
    test = omega*tau - 0.5
    if test > eps:
        ans = np.exp(-t/(2*tau)) * (np.cos(Omega*t) + np.sin(Omega*t) / (2*Omega*tau))
    elif test < eps:
        from rpmdgle.utils.special import logcosh, logsinh
        ans = np.exp(-t/(2*tau) + logcosh(Omega*t))
        ans += np.exp(-t/(2*tau) + logsinh(Omega*t).real) / (2*Omega*tau)
    else:
        ans = np.exp(-t/(2*tau)) * (1 + t/(2*tau))
    return ans / (beta * m * omega**2)

def gle_cxx(
        t: Union[float, np.ndarray], 
        omega: float,
        Ap: Union[float, np.ndarray],
        Cp: Optional[Union[float, np.ndarray]] = None,
        C0: Optional[Union[float, np.ndarray]] = None,
        u: Optional[Union[float, np.ndarray]] = 0,
        beta: Optional[float] = None,
        mass: Optional[float] = 1.0) -> np.ndarray:
    """
    Calculate the covariance matrix <v(0) v(t)^T> where v(t) = (q, p, s^T) are the
    position and momentum of a harmonic oscillator plus the auxiliary variable coordinates.
    See https://doi.org/10.1021/ct900563s for notation.

    Args:
        t (float or ndarray): time(s) for which to compute the covariance matrix
        omega (float): frequency of the harmonic oscillator
        Ap (ndarray): momentum + auxiliary variable block of the drift matrix, for dynamics expressed in mass-weighted coordinates
        Cp (ndarray, optional): stationary covariance matrix for the momentum + auxvar subsystem in the absence of an external potential (mass-weighted coordinates); allows specifyng a non-thermal distribution
        C0 (ndarray, optional): covariance matrix at time 0, in mass-weighted coordinates
        u (float or ndarray, optional): second time variable for cross-correlation, to calculate <v(t) v(u)^T>
        beta (float, optional): reciprocal temperature, 1/kB*T
        mass (float, optional): mass of the harmonic oscillator

    Returns:
        ndarray: covariance matrix
    """
    from scipy.linalg import expm
    t = np.asarray(t)
    if t.ndim == 1:
        t = np.reshape(t, (1,-1,1,1)) 
    elif t.ndim == 2:
        t = t[...,None,None]
    else:
        raise RuntimeError('t should either be a one- or two-dimensional array')
    if np.any(t < 0):
        raise RuntimeError("Can only request positive times!")
    u = np.atleast_1d(u)
    if u.ndim != 1:
        raise RuntimeError('u should be a scalar or a one-dimensional array')
    u = np.reshape(u, (-1,1,1,1))
    if np.any(u < 0):
        raise RuntimeError("Can only request positive times!")
    if (beta is None and Cp is None) or (beta is not None and Cp is not None):
        raise RuntimeError("Specify either beta (for detailed balance) or Cp (for a general thermostat), not both.")
    Ap = np.atleast_2d(Ap)
    A_qp = get_Aqp(Ap, omega)
    n = Ap.shape[0]
    if Cp is None:
        C_qp = get_Cqp_canonical(beta, n, omega)
    else:
        Cp = np.atleast_2d(Cp)
        # the first element of C_p is optimized to be kT for the quantum thermostat 
        # (need temperature in case of non-equilibrium starting covariance matrix)
        beta = 1/Cp[0, 0]
        D_qp = get_Dqp(Ap, Cp)
        C_qp = get_Cqp(A_qp, D_qp)
    if C0 is None:
        term0 = 0
    else:
        C0_ = np.atleast_2d(C0)
        check_matrix(C0_, 'C_qp0')
        nqp = C0_.shape[0]
        C0 = np.eye(n+1) * (1/beta)
        if nqp in {1,2}:
            C0[:nqp,:nqp] = C0_[:nqp,:nqp]
        else:
            C0[:,:] = C0_
        term0 = expm(-A_qp * t) @ (C0 - C_qp) @ expm(-np.transpose(A_qp) * u)
    term1 = expm(-A_qp * np.abs(t - u)) @ C_qp
    tcf = term0 + term1
    sqm = np.sqrt(mass) # rescaling for mass != 1 
    sqmvec = np.ones(n+1)
    sqmvec[0] = 1/sqm
    sqmvec[1] = sqm
    sqmmat = sqmvec[:,None] * sqmvec[None,:]
    return np.squeeze(tcf * sqmmat)

def check_matrix(
        M : Union[float, np.ndarray],
        name : str) -> None:
    """
    Check if matrix M is 2-dimensional and square. Raise RuntimeError if not.

    Args:
        M (ndarray): matrix to check
        name (str): name of the matrix (for error messages)
    """
    if M.ndim != 2:
        raise RuntimeError(f"2-dimensional array expected for {name}, instead {M.ndim = }")
    if M.shape[0] != M.shape[1]:
        raise RuntimeError(f"Square {name} expected, instead got {M.shape = }")  
    return

def get_Aqp(
        Ap : Union[float, np.ndarray], 
        omega : float) -> np.ndarray:
    """
    Construct the full drift matrix for system+auxiliary variables.

    Args:
        Ap (ndarray): momentum + auxiliary variable block of the drift matrix, shape (n, n)
        omega (float): harmonic oscillator frequency

    Returns:
        ndarray: full drift matrix, shape (n+1, n+1)
    """
    check_matrix(Ap, 'Ap') 
    n = Ap.shape[0]
    Aqp = np.zeros((n+1, n+1))
    Aqp[1:,1:] = Ap
    Aqp[0, 1] = -1
    Aqp[1, 0] = omega**2
    return Aqp

def get_Cqp_canonical(
        beta : float, 
        n : int, 
        omega : float) -> np.ndarray: 
    """
    Compute the canonical covariance matrix for mass-weighted phase-space variables.

    Args:
        beta (float): reciprocal temperature, 1/kB*T
        n (int): size of the Ap block of the drift matrix (1 + number of auxvars)
        omega (float): harmonic oscillator frequency

    Returns:
        ndarray: covariance matrix for the canonical ensemble, shape (n+1, n+1)
    """
    Cqp = np.eye(n+1) / beta
    Cqp[0,0] /= omega**2
    return Cqp   

def get_Dqp(
        Ap : Union[float, np.ndarray], 
        Cp : Union[float, np.ndarray]) -> np.ndarray:
    """
    Construct the diffusion matrix D = B.B^T for the OU process.

    Args:
        Ap (ndarray): momentum + auxiliary variable block of the drift matrix, shape (n, n)
        Cp (ndarray): corresponding C_p matrix

    Returns:
        ndarray: D_qp matrix, shape (n+1, n+1)
    """
    check_matrix(Cp, 'C_p')
    n = Cp.shape[0]
    Dp = Ap @ Cp + Cp @ np.transpose(Ap)
    Dqp = np.zeros((n+1, n+1))
    Dqp[1:, 1:] = Dp
    return Dqp   

def get_Cqp(
        Aqp : Union[float, np.ndarray], 
        Dqp : Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute the stationary covariance matrix for mass-weighted variables that solves
    Aqp Cqp + Cqp Aqp^T = Dqp.

    Args:
        Aqp (ndarray): full drift matrix, shape (n+1, n+1)
        Dqp (ndarray): diffusion matrix, shape (n+1, n+1)

    Returns:
        ndarray: stationary covariance matrix, shape (n+1, n+1)
    """
    import scipy.linalg as sclin
    eigs, O = sclin.eig(Aqp)
    O_inv = np.linalg.inv(O)
    n = Aqp.shape[0]
    Cqp = np.zeros_like(Aqp, dtype=complex)
    M = O_inv @ Dqp @ np.transpose(O_inv)
    for i in range(n): 
        for j in range(n):
            C = 0j
            for k in range(n):
                for l in range(n):
                    o = O[i, k] * M[k, l] * O[j, l]
                    a = eigs[k] + eigs[l]
                    C += o/a
            Cqp[i, j] = C
    return np.real(Cqp)