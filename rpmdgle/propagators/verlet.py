#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   verlet.py
@Time    :   2023/11/30 14:30:07
@Author  :   George Trenins
@Desc    :   Verlet-style propagators for NVE dynamics
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from rpmdgle.utils.nmtrans import FFTNormalModes
from rpmdgle.utils import arrays


class BasePropa(object):

    def __init__(self, PES, dt, xshape, rng, *args, **kwargs):
        self.PES = PES
        self.UNITS = PES.UNITS
        self.dt = float(self.UNITS.str2base(dt))
        self._t = 0.0
        self.m = np.atleast_1d(PES.mass).copy()
        self.xshape = xshape
        try:
            self.rng = np.random.default_rng(seed=rng)
        except TypeError as err:
            raise TypeError("Propagator expects one of None, int, array_like[ints], SeedSequence, BitGenerator, Generator to initialise the RNG.") from err
        try:
            super().__init__(PES, dt, xshape, rng, *args, **kwargs)
        except TypeError:
            pass

    @property
    def xshape(self):
        return self._xshape
    
    @xshape.setter
    def xshape(self, xshape):
        self._xshape = xshape
        for attr in ['p', 'x', 'f']:
            setattr(self, attr, np.empty(xshape))
        self.fnm = self.f
        self.pnm = self.p
        self.xnm = self.x
        # Dynamic masses
        self.m3 = np.broadcast_to(self.m, xshape)
        # ... and their square roots
        self.sqm3 = np.sqrt(self.m3)
            
    def step(self, *args, **kwargs):
        pass

    def econs(self, *args, **kwargs):
        pass

    def force_update(self):
        self.V, self.f[:] = self.PES.both(self.x)
        self.f *= -1

    def set_x(self, x):
        self.x[:] = x
        # for fixed positions
        self._cached_x = np.copy(x)
        self.force_update()

    set_xnm = set_x

    def set_p(self, p):
        self.p[:] = p

    set_pnm = set_p

    def psample(self, beta, **kwargs):
        """Draw momenta from the Boltzmann distribution.

        Args:
            beta (float): 1/kB*T

        Kwargs:
            slices (optional): which slices of position array to sample. 
                Unselected elements are set to zero. All elements are
                resampled by default

        Returns:
            p (ndarray): resampled momenta
        """

        slices = kwargs.get("slices", slice(None))
        ans = np.zeros(self.xshape)
        ans_slc = ans[slices]
        m_slc = self.m3[slices]
        ans_slc[:] = self.rng.normal(size=ans_slc.shape) * np.sqrt(m_slc / beta)
        return ans


class VelocityVerlet(BasePropa):

    def __init__(self, PES, dt, xshape, rng, *args, **kwargs):
        super().__init__(PES, dt, xshape, rng, *args, **kwargs)
        self.fixed = kwargs.get("fixed", [])

    def fix_positions(self):
        for f in self.fixed:
            self.x[f] = self._cached_x[f]

    def fix_momenta(self):
        for f in self.fixed:
            self.p[f] = 0.0

    def set_p(self, p):
        super().set_p(p)
        self.fix_momenta()

    def step0(self, slices=None, **kwargs):
        """First half of the velocity Verlet algorithm.

        Args:
            slices (optional): which slices of position/momentum arrays 
                to propagate. All are propagated by default.
        """
        if slices is None:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key) for key in ('p', 'x', 'f', 'm3')]
        else:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key)[slices] for key in ('p', 'x', 'f', 'm3')]
        p_slc[...] += f_slc*self.dt/2
        x_slc[...] += p_slc*self.dt/m_slc
        self.fix_positions()
        self.fix_momenta()
        self.force_update()

    def step1(self, slices=None, **kwargs):
        """Second half of the velocity Verlet algorithm.

        Args:
            slices (optional): which slices of position/momentum arrays 
                to propagate. All are propagated by default.
        """
        p_slc, f_slc = [
            getattr(self, key)[slices] for key in ('p', 'f')]
        p_slc[...] += f_slc*self.dt/2
        self.fix_momenta()

    def step(self, s=1, **kwargs):
        """Iterate over Velocity Verlet steps

        Args:
            s (int, optional): Number of iterations. Defaults to 1.

        Kwargs:
            slices (optional): which slices of position/momentum arrays 
                to propagate. All are propagated by default.
        """

        for i in range(s):
            self.step0(**kwargs)
            self.step1(**kwargs)
        self._t += s*self.dt

    def kinetic_energy(self, *args, **kwargs):
        dimtot = np.ndim(self.p)
        dimrep = np.ndim(self.V)
        if dimrep == 0:
            axis = None
        else:
            axis = tuple(range(-1,dimrep-dimtot-1,-1))
        return np.sum(self.p**2/ self.m3, axis=axis)/2
    
    def potential_energy(self, *args, **kwargs):
        return self.V

    def econs(self, *args, **kwargs):
        """Return the total energy of the system.
        """
        return self.kinetic_energy() + self.V
    

class RingNM(VelocityVerlet):

    def __init__(self, PES, dt, xshape, rng, *args, **kwargs):
        """Initialise a reference system propagator (RESPA) based on
        the free ring polymer

        Args:
            PES (Ring): ring-polymer potential
            dt (float): propagator time step
            xshape (tuple): shape of the arrays to be propagated
            rng (various): random number generator or seed for the
                initialisation thereof
            mscale (1d-array, optional): scale factor for normal-mode masses.
                Defaults to None.
        """
        self.omegaN = PES.omegaN
        self.nbeads = PES.N
        self.N = self.nbeads # alias
        self.beta = PES.beta
        self.hbar = PES.UNITS.hbar
        self.PES = PES
        self._test_potential(PES)
        self.setup_nmtrans(xshape)
        self.parse_mscale(kwargs.get('mscale'))
        super().__init__(PES, dt, xshape, rng, *args, **kwargs)
        self.get_coeffs(self.dt)
        
    @property
    def xshape(self):
        return super().xshape
    
    @xshape.setter
    def xshape(self, xshape):
        self._xshape = xshape
        if self._xshape[-len(self.PES.rpxshape):] != self.PES.rpxshape:
            raise ValueError("Input arrays do not conform to the shape declared to the PES.")
        for attr in ['p', 'x', 'f', 'pnm', 'xnm', 'fnm']:
            setattr(self, attr, np.empty(xshape))
        self.m3 = np.broadcast_to(self.m, xshape) * self.mscale[
            (slice(None),)+len(self.PES.xshape)*(None,)]
        self.sqm3 = np.sqrt(self.m3)

    @staticmethod
    def _test_potential(PES):
        # if PES.__class__ is not Ring:
        #     raise TypeError("Expecting a ring-polymer potential for the free RP RESPA.")
        pass
    
    def setup_nmtrans(self,xshape):
        axis = len(xshape) - len(self.PES.rpxshape)
        self.nmtrans = FFTNormalModes(
            self.nbeads, bead_ax=axis)
        
    def fix_momenta(self):
        for f in self.fixed:
            self.pnm[f] = 0.0
        self.to_bead('p')

    def fix_positions(self):
        for f in self.fixed:
            self.xnm[f] = self._cached_xnm[f]
        self.to_bead('x')

    def parse_mscale(self, mscale):
        self.freqs = self.nmtrans.nm_freqs * self.omegaN
        if mscale is not None:
            self.mscale = np.atleast_1d(mscale)
            if self.mscale.ndim != 1:
                raise ValueError("Mass scale must be a one-dimensional array.")
            if len(self.mscale) != self.N:
                raise ValueError("Length of mass scaling array must be the same as the number of beads")
            self.freqs /= np.sqrt(self.mscale)
        else:
            self.mscale = np.ones(self.N)

    def get_coeffs(self, dt):
        self.coeffs = np.ones((4,)+self.m3.shape)
        freqs = arrays.append_dims(self.freqs, ndims=len(self.PES.xshape))
        self.coeffs *= freqs*dt
        self.coeffs[0] = np.cos(self.coeffs[0])
        self.coeffs[3] = self.coeffs[0].copy()
        self.coeffs[1] = np.sin(self.coeffs[1])
        self.coeffs[2] = self.coeffs[1].copy()
        self.coeffs[1] *= -freqs
        eps = 1.0e-8
        temp = np.where(np.abs(freqs) > eps, freqs, eps)
        self.coeffs[2] = np.where(np.abs(freqs) > eps,
                                  self.coeffs[2]/temp,
                                  dt*(1.0 - (freqs*dt)**2/6))
        
    def to_mode(self, attr):
        if attr not in {'p', 'x', 'f'}:
            raise RuntimeError("Trying to convert unknown attribute '{:s} to normal mode coordinates".format(attr))
        cart = getattr(self, attr)
        nm = getattr(self, '{:s}nm'.format(attr))
        nm[:] = self.nmtrans.cart2mats(cart)

    def to_bead(self, attr):
        if attr not in {'p', 'x', 'f'}:
            raise RuntimeError("Trying to convert unknown attribute '{:s} to bead coordinates".format(attr))
        cart = getattr(self, attr)
        nm = getattr(self, '{:s}nm'.format(attr))
        cart[:] = self.nmtrans.mats2cart(nm)

    def force_update(self):
        # f now only stores the external forces
        self.Vext, self.f[:] = self.PES.external_both(self.x)
        self.f *= -1
        self.to_mode('f')
        # calculate energy contributions from the RP springs
        self.VRP = self.PES.polymer_potential(self.x)
        self.V = self.Vext + self.VRP

    def set_x(self, x):
        self.set_xnm(self.nmtrans.cart2mats(x))

    def set_xnm(self, nm):
        self.xnm[:] = nm
        self._cached_xnm = np.copy(self.xnm)
        self.to_bead('x')
        self._cached_x = np.copy(self.x)
        self.force_update()

    def set_p(self, p):
        self.set_pnm(self.nmtrans.cart2mats(p))

    def set_pnm(self, nm):
        self.pnm[:] = nm
        self.fix_momenta()

    def step0(self, slices=None, **kwargs):
        if slices is None:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key) for key in ('pnm', 'xnm', 'fnm', 'm3')]
        else:
            p_slc, x_slc, f_slc, m_slc = [
                getattr(self, key)[slices] for key in ('pnm', 'xnm', 'fnm', 'm3')]
        p_slc[...] += f_slc*self.dt/2
        self.fix_momenta()

        self.RESPA_propa(p_slc, x_slc, slices=slices, **kwargs)
        self.fix_positions()
        self.fix_momenta()
        self.force_update()

    def step1(self, slices=None, **kwargs):
        """Second half of the velocity Verlet algorithm.

        Args:
            slices (optional): which slices of position/momentum arrays 
                to propagate. All are propagated by default.
        """
        p_slc, f_slc = [
            getattr(self, key)[slices] for key in ('pnm', 'fnm')]
        p_slc[...] += f_slc*self.dt/2
        self.fix_momenta()
    
    def RESPA_propa(self, p_slc, x_slc, slices=None, **kwargs):
        """Execute the free ring-polymer propagation step.
        """

        if slices is None:
            coeffs = self.coeffs
        else:
            slc = [slice(None),]
            try:
                slc.extend(slices)
            except TypeError:
                slc.extend((slices,))
            coeffs = self.coeffs[slices]
        # mass-weight
        p_slc /= self.sqm3[slice(None) if slices is None else slices]
        x_slc *= self.sqm3[slice(None) if slices is None else slices]
        tmp = np.copy(p_slc)
        p_slc *= coeffs[0]
        p_slc += coeffs[1]*x_slc
        x_slc *= coeffs[3]
        x_slc += coeffs[2]*tmp
        # Undo mass weighting
        p_slc *= self.sqm3[slice(None) if slices is None else slices]
        x_slc /= self.sqm3[slice(None) if slices is None else slices]


    def kinetic_energy(self, *args, **kwargs):
        dimtot = np.ndim(self.pnm)
        dimsys = np.ndim(self.V)
        if dimsys == 0:
            axis = None
        else:
            axis = tuple(range(-1,dimsys-dimtot-1,-1))
        sqmp = self.nmtrans.mats2cart(self.pnm / self.sqm3)
        return np.sum(sqmp**2, axis=axis)/2