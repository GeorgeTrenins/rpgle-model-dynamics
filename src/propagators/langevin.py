#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   langevin.py
@Time    :   2023/11/30 14:31:28
@Author  :   George Trenins
@Desc    :   Propagators for langevin dynamics
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from rpmdgle.propagators.verlet import VelocityVerlet, RingNM
from rpmdgle.utils import arrays
from typing import Any, Union


class Langevin(VelocityVerlet):

    def __init__(self, PES, dt, xshape, rng, beta, tau, *args, **kwargs):
        super().__init__(PES, dt, xshape, rng, *args, **kwargs)
        tau = float(self.UNITS.str2base(tau))
        self.tau = tau
        self.c1 = np.exp(-self.dt / tau)
        self.c2 = np.sqrt(self.m3/beta * (1.0-self.c1**2))
        
    def set_x(self, x):
        super().set_x(x)
        # Tracks energy changes minus thermostat contributions
        self.ethermo = np.zeros_like(self.V)
    
    def step0(self, slices=None, **kwargs):
        self.ethermo -= super().econs()
        if slices is None:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key) for key in ('p', 'x', 'f', 'm3')]
        else:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key)[slices] for key in ('p', 'x', 'f', 'm3')]
        p_slc[:] += f_slc*self.dt/2           # B
        x_slc[:] += p_slc*self.dt/(2*m_slc)   # A
        self.fix_positions()
        self.fix_momenta()
        self.ethermo += self.kinetic_energy()
        pi = (self.c1*self.p)[slices]         # O
        scale = (self.c2)[slices]
        pi += self.rng.normal(scale=scale)
        p_slc[:] = pi
        self.fix_momenta()    
        self.ethermo -= self.kinetic_energy()
        x_slc[:] += p_slc*self.dt/(2*m_slc)   # A
        self.fix_positions()
        self.force_update()

    def step1(self, slices=None, **kwargs):
        super().step1(slices, **kwargs)
        self.ethermo += super().econs()
    
    def econs(self, *args, **kwargs):
        """Return the total energy of the system.
        """
        return self.ethermo.copy()
    

class SVR(VelocityVerlet):

    def __init__(self, PES, dt, xshape, rng, beta, tau, *args, **kwargs):
        """Stochastic velocity rescaling, a.k.a. Bussi-Donadio-Parrinello thermostat.

        References:
            https://doi.org/10.1063/1.2408420
            https://doi.org/10.1063/1.3489925

        Args:
            PES (Any): potential energy surface
            dt (str of float): propagation timestep
            xshape (tuple): shape of the propagated arrays
            rng (np.random.Generator or int): an object to (re-)initialize np.random.default_rng
            beta (float): reciprocal temperature in the unit system of PES
            tau (str of float): reciprocal friction constatn in the time units of PES. 

        Returns:
            None
        """
        super().__init__(PES, dt, xshape, rng, *args, **kwargs)
        self.beta = beta
        tau = float(self.UNITS.str2base(tau))
        self.tau = tau
        self.c = np.exp(-self.dt / tau)
        self.num_active_dofs = None

    def compute_active_dofs(self) -> None:
        """
        Calculate the number of active degrees of freedom for every independent 
        system replicant, accounting for the fixed degrees of freedom. This _may_ be
        different for different replicants
        """
        
        # number of replicant dimensions only
        dimrep = np.ndim(self.V)
        # total number of DoFs in individual replicant
        ftot = np.prod(self.xshape[dimrep:])
        all_indices = np.reshape(np.arange(np.prod(self.xshape), dtype=int), self.xshape)
        num_fixed_dofs = np.zeros(self.xshape[:dimrep], dtype=int)
        nfd_flat = np.reshape(num_fixed_dofs, -1)
        replicant_indices = np.reshape(all_indices, (len(nfd_flat), -1))
        # Accumulate all fixed indices into single list
        fixed_indices = []
        for f in self.fixed:
            fixed_indices.extend(np.ravel(all_indices[f]).tolist())
        fixed_index_set = set(fixed_indices)
        # For the indices of every replicant, see how many appear in the list
        # of fixed indices
        for i, index_array in enumerate(replicant_indices):
            index_set = set(index_array.tolist())
            nfd_flat[i] = len(index_set.intersection(fixed_index_set))
        self.num_active_dofs = ftot - num_fixed_dofs

    def _alpha(self):
        if self.num_active_dofs is None:
            self.compute_active_dofs()
        f = self.num_active_dofs
        KE = self.kinetic_energy()
        xi0 = self.rng.normal(size=f.shape)
        # compute sum of squared Gaussian variate from Gamma distribution,
        # see Appendix of https://doi.org/10.1063/1.2408420
        df = f - 1
        mask = df > 0
        tmp = np.where(mask, df, 1)
        xiRest = np.where(mask, self.rng.chisquare(tmp, size=f.shape), 0)
        a2 = (1-self.c) / (2 * self.beta * KE) * (xi0**2 + xiRest)
        a2 += 2*xi0 * np.sqrt(self.c * (1-self.c) / (2 * self.beta * KE)) + self.c
        sgn_a = np.sign(xi0 + np.sqrt((2 * self.beta * KE * self.c) / (1 - self.c)))
        return sgn_a * np.sqrt(a2)
        
    def set_x(self, x):
        super().set_x(x)
        # Tracks energy changes minus thermostat contributions
        self.ethermo = np.zeros_like(self.V)

    def step0(self, slices=None, **kwargs):
        self.ethermo -= super().econs()
        if slices:
            raise NotImplementedError("Cannot fix parts of system during simulation time when using the SVR thermostat. Fixed DOFs must be specified upon instantiation via the 'fixed' keyword")
        else:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key) for key in ('p', 'x', 'f', 'm3')]
        p_slc[:] += f_slc*self.dt/2           # B
        x_slc[:] += p_slc*self.dt/(2*m_slc)   # A
        self.fix_positions()
        self.fix_momenta()
        self.ethermo += self.kinetic_energy()
        alpha: np.ndarray = self._alpha()
        shaped_alpha = arrays.append_dims(alpha, self.p.ndim - alpha.ndim)
        p_slc *= shaped_alpha
        self.ethermo -= self.kinetic_energy()
        x_slc[:] += p_slc*self.dt/(2*m_slc)   # A
        self.fix_positions()
        self.force_update()

    def step1(self, slices=None, **kwargs):
        if slices:
            raise NotImplementedError("Cannot fix parts of system during simulation time when using the SVR thermostat. Fixed DOFs must be specified upon instantiation via the 'fixed' keyword")
        super().step1(**kwargs)
        self.ethermo += super().econs()
    
    def econs(self, *args, **kwargs):
        """Return the total energy of the system.
        """
        return self.ethermo.copy()

    
class RingPILE(RingNM):

    def __init__(self,
                 PES : Any,
                 dt : Union[float, str],
                 xshape : tuple[int],
                 rng : Any,
                 beta : float,
                 tau : float,
                  **kwargs):
        """Initialise a reference system propagator (RESPA) based on
        the free ring polymer, with a Langevin thermostat attached
        to each mode. The friction of centroid modes is set to
        1/tau, and the centroid modes to 2λ times the free RP
        normal-mode frequency.

        Args:
            PES (Ring): ring-polymer potential
            dt (float): propagator time step
            xshape (tuple): shape of the arrays to be propagated
            rng (various): random number generator or seed for the
                initialisation thereof
            beta (float): temperature of the simulation
            tau (float): reciprocal of centroid-mode friction
            lamda (float): scale factor for non-centroid friction.
                Defaults to 0.5
            mscale (1d-array, optional): scale factor for 
                normal-mode masses. Defaults to None.
        """

        super().__init__(PES, dt, xshape, rng, **kwargs)
        lamda = kwargs.get('lamda', 0.5)
        tau = float(self.UNITS.str2base(tau))
        self.get_pile_coeffs(self.dt, beta, tau, lamda)
        # Overwrite RESPA coeffs - doing two half-step
        self.get_coeffs(self.dt/2)
        self.ethermo = 0.0

    def get_pile_coeffs(self, dt, beta, tau, lamda):
        self.friction = 2*lamda*np.abs(self.freqs)
        self.friction[0] = 1/tau
        friction = arrays.append_dims(
            self.friction, ndims=len(self.PES.xshape))
        self.pile_coeffs = np.zeros((2,)+self.xshape)
        self.pile_coeffs[0] = np.exp(-dt * friction)
        self.pile_coeffs[1] = np.sqrt(self.m3/beta * (1.0-self.pile_coeffs[0]**2))

    def set_x(self, x):
        super().set_x(x)
        # Tracks energy changes minus thermostat contributions
        self.ethermo = np.zeros_like(self.V)

    def step0(self, slices=None, **kwargs):
        self.ethermo -= super().econs()
        super().step0(slices, **kwargs)

    def step1(self, slices=None, **kwargs):
        super().step1(slices, **kwargs)
        self.ethermo += super().econs()

    def econs(self, *args, **kwargs):
        """Return the total energy of the system.
        """
        return self.ethermo.copy()
        
    def RESPA_propa(self, p_slc, x_slc, slices=None, therm=True, **kwargs):
        """Execute the free ring-polymer propagation half-step,
        followed by thermostat, followed by another free RP half-step
        """

        super().RESPA_propa(p_slc, x_slc, slices=slices, **kwargs)
        if therm:
            self.fix_momenta()
            self.ethermo += self.kinetic_energy()
            pi = (self.pile_coeffs[0]*self.pnm)[slices]
            scale = (self.pile_coeffs[1])[slices]
            pi += self.rng.normal(scale=scale)
            p_slc[:] = pi        
            self.fix_momenta()
            self.ethermo -= self.kinetic_energy()
        super().RESPA_propa(p_slc, x_slc, slices=slices, **kwargs)

    def centroid_virial(self):
        predims =  len(self.xshape) - len(self.PES.xshape)
        new_shape = self.xshape[:predims] + (-1,)
        qnm = self.xnm.reshape(new_shape)
        fnm = self.fnm.reshape(new_shape)
        ndof = qnm.shape[-1]
        ans = -np.sum(qnm[...,1:,:]*fnm[...,1:,:], axis=(-1,-2)) / 2
        ans += ndof / (2*self.beta)
        return ans