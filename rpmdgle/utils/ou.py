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
        m: float = 1):
    """
    Position auto-correlation function for a harmonic oscillator at thermal equilibrium,
    under the action of a white-noise Langevin equation thermostat (WNLE),

        <q(t) q(0)> = exp(-γ|t|/2) [cos(ω̄|t|) + γ/(2ω̄) sin(ω̄|t|)] / (β m ω²)

    with γ = 1/τ and ω̄ = sqrt(ω² - γ²/4). For γ > 2ω, cos and sin become cosh
    and sinh of κt, with κ = sqrt(γ²/4 - ω²); this branch is evaluated
    as a sum of decaying exponentials to avoid overflow at long times. For
    γ = 2ω, the bracket is 1 + γ|t|/2. The time dependence is the same as that of
    the Brownian-oscillator kernel, `rpmdgle.sysbath.spectral.brownian.Density.K`.

    Args:
        t (float or ndarray): time(s) at which to evaluate the correlation function
        beta (float): reciprocal temperature, 1/kB*T
        omega (float): frequency of the harmonic oscillator
        tau (float): reciprocal of the friction, tau = 1/gamma
        m (float, optional): mass of the harmonic oscillator

    Returns:
        float or ndarray: position auto-correlation function at time(s) t
    """
    for name, value in zip(('beta', 'omega', 'tau', 'm'), (beta, omega, tau, m)):
        if not value > 0:
            raise ValueError(f"Expected a positive value for {name}, instead got {value}")
    t = np.abs(t)
    gamma = 1/tau
    prefactor = 1 / (beta * m * omega**2)
    # s = ω̄², where ω̄ is the damped frequency of the oscillator
    s = omega**2 - gamma**2/4
    if s > 0:
        wbar = np.sqrt(s)
        return prefactor * np.exp(-gamma*t/2) * (
            np.cos(wbar*t) +
            gamma/(2*wbar) * np.sin(wbar*t))
    elif s < 0:
        kappa = np.sqrt(-s)
        rate = omega**2 / (gamma/2 + kappa)
        return prefactor * np.exp(-rate*t) * (
            (1 + np.exp(-2*kappa*t))/2 -
            gamma/(4*kappa) * np.expm1(-2*kappa*t))
    else:
        return prefactor * np.exp(-gamma*t/2) * (1 + gamma*t/2)

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
    Calculate the covariance matrix <v(t) v(0)^T> where v(t) = (q, p, s^T) are the
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

def check_Ap_structure(
        Ap : np.ndarray) -> None:
    """
    Check that Ap is a square matrix with no Markovian friction, Ap[0,0] = 0, and
    antisymmetric coupling between the momentum and the auxiliary variables,
    Ap[1:,0] = -Ap[0,1:]. This is the structure assumed by `get_friction_kernel`
    and `get_friction_spectrum`.

    Args:
        Ap (ndarray): momentum + auxiliary variable block of the drift matrix, shape (n, n)

    Raises:
        RuntimeError: if Ap is not a square matrix
        NotImplementedError: if Ap[0,0] != 0
        ValueError: if Ap[1:,0] != -Ap[0,1:]
    """
    check_matrix(Ap, 'Ap')
    if Ap[0,0] != 0:
        raise NotImplementedError(f"Markovian friction Ap[0,0] = {Ap[0,0]} is not supported, only Ap[0,0] = 0.")
    if not np.allclose(Ap[1:,0], -Ap[0,1:], rtol=1.0e-12, atol=0.0):
        raise ValueError(f"Expected antisymmetric coupling Ap[1:,0] = -Ap[0,1:], instead got Ap[1:,0] = {Ap[1:,0]} and Ap[0,1:] = {Ap[0,1:]}.")
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

def get_friction_kernel(
    t: Union[float, np.ndarray], 
    Ap: np.ndarray,
) -> np.ndarray:
    """Calculate the friction kernel for a GLE with drift matrix Ap, using the formula K(t) = theta^T exp(-t A) theta, where theta is the first row of Ap, excluding Ap[0,0] and A is the lower-right block Ap[1:,1:]. See https://doi.org/10.1021/ct900563s for notation.

    Args:
        t (Union[float, np.ndarray]): time or times for which to compute the friction kernel
        Ap (np.ndarray): drift matrix for the momentum + auxiliary variables block

    Returns:
        np.ndarray: friction kernel values at the requested times

    Raises:
        NotImplementedError: if Ap[0,0] != 0
        ValueError: if Ap[1:,0] != -Ap[0,1:]
    """
    from scipy.linalg import expm
    is_scalar = np.isscalar(t)
    t = np.atleast_1d(np.asarray(t))
    if np.ndim(t) != 1:
        raise RuntimeError("t should be a one-dimensional array")
    t = np.abs(t)
    Ap = np.asarray(Ap)
    check_Ap_structure(Ap)
    theta = Ap[0,1:]
    A = Ap[1:,1:]
    kernel: np.ndarray = np.einsum(
        'ijk,j,k->i', 
        expm(-t[:,None,None] * A), theta, theta)
    if is_scalar:
        return kernel.item()
    else:
        return kernel

def get_friction_spectrum(
    omega: Union[float, np.ndarray], 
    Ap: np.ndarray
) -> np.ndarray:
    """Calculate the friction spectrum for a GLE with drift matrix Ap.
    
    Computes Λ(ω) = θᵀ A (A² + ω² I)⁻¹ θ, where θ is the first row of Ap
    (excluding Ap[0,0]) and A is the lower-right block Ap[1:,1:].

    Raises:
        NotImplementedError: if Ap[0,0] != 0
        ValueError: if Ap[1:,0] != -Ap[0,1:]
    """
    is_scalar = np.isscalar(omega)
    omega = np.atleast_1d(np.asarray(omega))
    if np.ndim(omega) != 1:
        raise RuntimeError("omega should be a one-dimensional array")
    Ap = np.asarray(Ap)
    check_Ap_structure(Ap)
    theta = Ap[0,1:]
    A = Ap[1:,1:]
    lamda, Q = np.linalg.eig(A)
    y = np.linalg.solve(Q, theta)  # y = Q⁻¹θ
    z = Q.T @ theta           
    spectrum: np.ndarray = np.real(np.sum(
        y*z*lamda / (lamda**2 + omega[:,None]**2),
        axis=-1
    ))
    if is_scalar:
        return spectrum.item()
    else:
        return spectrum
    