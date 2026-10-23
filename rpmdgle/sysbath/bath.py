#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bath.py
@Time    :   2026/10/23 16:36:56
@Author  :   George Trenins
@Desc    :   Caldeira-Leggett potential for a system coupled to a discretised harmonic bath
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES
from rpmdgle.sysbath.coupling import linear
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
from typing import Optional, Sequence, Tuple
import numpy as np


class CaldeiraLeggett(BasePES):

    def __init__(self,
                 PES: BasePES,
                 terms: Sequence[Tuple[BaseSpectralDensity, int]],
                 coupling: Optional[BasePES] = None) -> None:
        """One-dimensional system coupled to a harmonic bath whose spectral density
        is a sum of one or more terms, J(ω) = Σ_k J_k(ω). Each term is discretised
        separately into its own block of bath modes, and all modes couple to the
        system through the same function F(q),

            V(q, r) = V_sys(q) + Σ_ν (1/2) m_ν ω_ν² [r_ν - c_ν F(q) / (m_ν ω_ν²)]²

        Args:
            PES (BasePES): system potential; also sets the unit system
            terms (sequence): pairs (density, Nmodes) of a spectral density and
                the number of bath modes used to discretise it
            coupling (BasePES, optional): coupling function F(q) shared by all
                terms. Defaults to linear coupling, F(q) = q.
        """
        self.PES = PES
        self.UNITS = PES.UNITS
        units_name = PES.UNITS.__class__.__name__
        if coupling is None:
            coupling = linear.Coupling(UNITS=units_name)
        if coupling.UNITS.__class__.__name__ != units_name:
            raise RuntimeError(f"The coupling and the potential appear to have different unit systems, "
                               f"{coupling.UNITS.__class__.__name__} and "
                               f"{units_name}, respectively.")
        self.coupling = coupling
        terms = list(terms)
        if len(terms) == 0:
            raise ValueError("Expected at least one spectral density")
        nmodes = []
        for i, (_, N) in enumerate(terms):
            if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 1:
                raise ValueError(f"Expected a positive integer number of bath modes for term {i}, instead got {N!r}")
            nmodes.append(int(N))
        # spectral densities and the slices of the bath-mode arrays that discretise them
        self.densities = tuple(density for density, _ in terms)
        bounds = np.cumsum([0] + nmodes).tolist()
        self.term_slices = tuple(slice(a, b) for a, b in zip(bounds[:-1], bounds[1:]))
        self.Nmodes = bounds[-1]
        self.c, self.w, self.bath_mass = self.bath_params()
        self.ww = self.w*self.w
        self.mww = self.bath_mass*self.ww
        self.mass = np.concatenate((np.atleast_1d(PES.mass), self.bath_mass))

    def bath_params(self):
        """Discretise each term according to https://doi.org/10.1002/jcc.24527
        and concatenate the modes. Every bath mass equals the system mass,
        and the modes of term k each carry an equal share of its reorganisation
        energy, c_ν² / (m_ν ω_ν²) = λ_k / (2 N_k).

        Returns:
            c, w, m (1d-arrays): coupling coefficients, frequencies and masses of the bath modes
        """
        m = np.ones(self.Nmodes)
        m *= self.PES.mass # same as system mass
        w = np.empty(self.Nmodes)
        c = np.empty(self.Nmodes)
        for density, sl in zip(self.densities, self.term_slices):
            N = sl.stop - sl.start
            w[sl] = density.quadpoints(N)
            kappa = np.sqrt(density.exact_reorganisation()/(2*N))
            c[sl] = kappa * np.sqrt(m[sl]) * w[sl]
        return c, w, m

    def J(self, omega):
        """Total spectral density at frequency omega.
        """
        return sum(density.J(omega) for density in self.densities)

    def Lambda(self, omega):
        """Total spectral density at frequency omega, divided by omega.
        """
        return sum(density.Lambda(omega) for density in self.densities)

    def K(self, t):
        """Total memory-friction kernel at time t.
        """
        return sum(density.K(t) for density in self.densities)

    def exact_reorganisation(self):
        """Total reorganisation energy, λ = (4/π) * Integrate[ Λ(ω), {ω, 0, ∞}]
        """
        return sum(density.exact_reorganisation() for density in self.densities)

    @property
    def wplot(self):
        """Upper end of the default frequency range for plots of the spectrum.
        """
        hints = [density.wplot for density in self.densities if density.wplot is not None]
        return max(hints) if hints else None

    def quadrature(self, f):
        """Calculate the quadrature approximation to an integral
        Integrate[ J(ω) f(ω), {ω, 0, ∞}]

        Args:
            f (1d-array): input function evaluated at the quadrature points.
        """

        return (np.pi/2) * np.sum(self.c**2/(self.bath_mass*self.w) * f, axis=-1)

    def l_quadrature(self, f):
        """Calculate the quadrature approximation to an integral
        Integrate[ Λ(ω) f(ω), {ω, 0, ∞}] for Λ(ω) = J(ω)/ω

        Args:
            f (1d-array): input function evaluated at the quadrature points.
        """

        return (np.pi/2) * np.sum(self.c**2/(self.bath_mass*self.w**2) * f, axis=-1)

    def reorganisation_energy(self):
        """Use quadrature to calculate the reorganisation energy,
        λ = (4/π) * Integrate[ J(ω)/ω, {ω, 0, ∞}]
        """
        return (4/np.pi) * self.quadrature(1/self.w)

    def bath_eq(self, q):
        """Return the equilibrium positions of the bath modes
        given the system coordinate q.

        Args:
            q (ndarray): system coordinate, shape (..., 1)

        Returns:
            ndarray: equilibrium bath positions, shape (..., Nmodes)
        """
        # the couplings return a float for a single configuration
        F = np.asarray(self.coupling.potential(q))
        return self.c*F[...,None] / self.mww

    def potential(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        y = r - self.bath_eq(q)
        pot = np.sum(self.mww/2 * y**2, axis=-1)
        pot += self.PES.potential(q)
        return pot

    def gradient(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        g = np.zeros_like(x)
        gq = g[...,:f]
        gr = g[...,f:]
        gq[:] = self.PES.gradient(q)
        y = r - self.bath_eq(q)
        dgdq = self.coupling.gradient(q)
        gr[:] = self.mww * y
        gq[:] -= np.sum(self.c * y * dgdq, axis=-1, keepdims=True)
        return g

    def both(self, x):
        ndof = x.shape[-1]
        f = ndof - self.Nmodes
        q = x[...,:f]
        r = x[...,f:]
        g = np.zeros_like(x)
        gq = g[...,:f]
        gr = g[...,f:]
        pot, gq[:] = self.PES.both(q)
        y = r - self.bath_eq(q)
        dgdq = self.coupling.gradient(q)
        gr[:] = self.mww * y
        gq[:] -= np.sum(self.c * y * dgdq, axis=-1, keepdims=True)
        pot = pot + np.sum(self.mww/2 * y**2, axis=-1)
        return pot, g
