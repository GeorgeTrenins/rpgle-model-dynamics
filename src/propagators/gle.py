#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   gle.py
@Time    :   2023/11/30 14:35:08
@Author  :   George Trenins
@Desc    :   Propagators for generalized Langevin dynamics
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from string import ascii_lowercase
from rpmdgle.propagators.verlet import RingNM
from rpmdgle.propagators.langevin import RingPILE
from rpmdgle.utils.nmtrans import MatMulNormalModes


_squeeze1d = lambda arr: np.atleast_1d(np.squeeze(arr))
        
class SepGLEPILE(RingPILE):

    def __init__(self, rpPES, SB, dt, xshape, rng, beta, 
                 tau=None, *args, **kwargs):
        
        """Propagator for a ring-polymer system coupled to a harmonic bath. 
        This propagator is meant for PIMD (as opposed to RPMD) calculations, in that
        it incorporates the potential of mean field into the dynamics, and
        otherwise uses a PILE thermostat to sample the canonical distribution
        (the frictional and random forces are fictitious).
        
        Args:
            rpPES (Ring): ring-polymerised external potential
            SB (BasePES): a Caldeira--Leggett representation of the system+dissipative environment.
            dt (float): propagation time step
            xshape (ndarray): shape of the array to propagate
            rng (int or Generator): seed for random number generator
            beta (float): reciprocal temperature, 1/kB*T
            tau (float or str): reciprocal friction constant for the system, default None.
            lamda (float): scale factor for non-centroid friction. Defaults to 0.5
        """
        self.SB = SB
        RingNM.__init__(self, rpPES, dt, xshape, rng, *args, **kwargs)
        # Set up thermostatting of system coordinates
        self.beta = beta
        if tau is not None:
            self.tau = float(self.UNITS.str2base(tau))
            # Using OBABO splitting here
            lamda = kwargs.get('lamda', 0.5)
            self.get_pile_coeffs(self.dt/2, self.beta, self.tau, lamda)
        else:
            self.tau = None

        # Determine layout of independent system realisations
        try:
            # Shape of single realisation of the ring-polymerised system
            rpxshape = self.PES.rpxshape
        except AttributeError:
            # must be a classical potential, no beads
            raise RuntimeError("Expecting a ring-polymerised potential even for classical simulations!")
        # dimensionality of the configuration array for a single realisation
        n = len(rpxshape)                       
        # layout of independent system realisations
        self.replica_layout = self.xshape[:-n]  
        # Set up the mean-field potential
        self.build_pmf()
        self.ethermo = np.zeros_like(self.PMF)

    @staticmethod
    def _parse_einsum_input(operands):
        """our input is a narrow subset of what np.einsum can parse - reduce generality for
        greater speed
        """
        rhs_idx = operands[1::2]
        rhs = []
        for idx in rhs_idx:
            rhs.append(''.join(['...' if i is Ellipsis else ascii_lowercase[i] for i in idx]))
        rhs = ','.join(rhs)
        lhs_idx = operands[-1]
        lhs = ''.join(['...' if i is Ellipsis else ascii_lowercase[i] for i in lhs_idx])
        arrs = operands[:-1:2]
        return rhs, lhs, arrs

    def _contract(self, *operands, **kwargs):
        lhs, rhs, arrs = self._parse_einsum_input(operands)
        expr = '->'.join([lhs, rhs])
        return np.einsum(expr, *arrs, **kwargs)

    def build_pmf(self):
        """Allocate arrays for storing normal-mode couplings, potential of mean field; store the indices for einstein summation
        """
        # Bath parameters in Caldeira-Leggett representation
        c = self.SB.c
        w = self.SB.w
        mu = self.SB.bath_mass
        # Ring-polymer normal-mode frequencies
        wn = self.freqs
        # Frequency-dependent part of the PMF
        mwb2 = mu*w**2
        mwn2 = mu*wn[:,None]**2
        mwbn2 = mwb2 + mwn2
        c2 = c**2
        # sum over the bath modes
        self._alpha = np.sum(c2 * (1/mwb2 - 1/mwbn2), axis=-1)
        # Build the transformation matrices
        coupling_transform = MatMulNormalModes(self.nbeads, self.nbeads, self.nmtrans.axis)
        self.coupling_grad_mat = (coupling_transform.forward_matrix[:,None,:] * 
                                  coupling_transform.backward_matrix.T[None,:,:])
        # Allocate arrays for coupling potentials and gradients in bead representation
        self.F = np.zeros(self.replica_layout+(self.nbeads,))
        self.dFdX = np.zeros(self.xshape)
        self.PMF = np.zeros(self.replica_layout)
        self.fext = np.zeros_like(self.f)
        # Same in normal-mode representation
        self.nmF = np.zeros_like(self.F)
        self.nmdFdX = np.zeros(self.replica_layout+
                               (self.nbeads,)+
                               self.PES.rpxshape)
        self.nmFMF = np.zeros(self.xshape)
        self.nmfext = np.zeros_like(self.fnm)
        self._oe_paths = dict()
        # The index specification here is not very readable - refactor/add better docs later.
        # Store summation indices for coupling-force calculations
        #                        n   n'  l 
        dfdx_indices = list(range(self.dFdX.ndim))
        max_idx = max(dfdx_indices)
        coupling_mat_indices = [max_idx+1, max_idx+2, max_idx+3]
        dfdx_indices[self.nmtrans.axis] = coupling_mat_indices[-1]
        nmdfdx_indices = dfdx_indices[:self.nmtrans.axis] + coupling_mat_indices[:2] + dfdx_indices[self.nmtrans.axis+1:]
        #
        _alpha_indices = coupling_mat_indices[1:2]
        nmf_indices = dfdx_indices[:self.nmtrans.axis] + coupling_mat_indices[1:2]
        nmfmf_indices = dfdx_indices.copy()
        nmfmf_indices[self.nmtrans.axis] = coupling_mat_indices[0]
        self._oe_indices = [
            coupling_mat_indices, dfdx_indices, nmdfdx_indices,
            _alpha_indices, nmf_indices, nmdfdx_indices, nmfmf_indices
        ]
        
    def to_mode(self, attr):
        try:
            super().to_mode(attr)
        except RuntimeError:
            if attr in {'F', 'fext'}:
                cart = getattr(self, attr)
                nm = getattr(self, 'nm{:s}'.format(attr))
                nm[:] = self.nmtrans.cart2mats(cart)
            else:
                raise RuntimeError("Trying to convert unknown attribute '{:s} to normal mode coordinates".format(attr))
            
    def to_bead(self, attr):
        try:
            super().to_bead(attr)
        except RuntimeError:
            if attr in {'F', 'fext'}:
                cart = getattr(self, attr)
                nm = getattr(self, 'nm{:s}'.format(attr))
                nm[:] = self.nmtrans.mats2cart(cart)
            else:
                raise RuntimeError("Trying to convert unknown attribute '{:s} to bead coordinates".format(attr))
            
    def force_update(self):
        # external forces
        super().force_update()
        # extras for the potential of mean field
        self.F[:], self.dFdX[:] = self.SB.coupling.both(self.x)
        self.to_mode('F')
        self.PMF[:] = np.sum(self._alpha * self.nmF**2, axis=-1) * self.nbeads/2
        self.V += self.PMF
        # Compute d F^{(n')} / d X^{(n)}
        self._contract(
            self.coupling_grad_mat, self._oe_indices[0],
            self.dFdX, self._oe_indices[1],
            self._oe_indices[2], out=self.nmdFdX)
        # Compute Sum[ _alpha^{(n')} * F^{(n')} * d F^{(n')} / d X^{(n)}, n' ]
        self._contract(
            self._alpha, self._oe_indices[3],
            self.nmF, self._oe_indices[4],
            self.nmdFdX, self._oe_indices[5],
            self._oe_indices[6], out=self.nmFMF
        )
        self.fnm -= self.nmFMF
        self.to_bead("f")

    def O(self):
        """Propagate the dissipative dynamics of the auxiliary variables and the centroid.
        """
        if self.tau is not None:
            self.ethermo += self.kinetic_energy()
            self.pnm *= self.pile_coeffs[0]
            self.pnm += self.rng.normal(scale=self.pile_coeffs[1], size=self.pnm.shape)
            self.fix_momenta()
            self.ethermo -= self.kinetic_energy()

    def B(self):
        """Propagate the momenta under the influence of the external potential, as well as
        the s2 auxiliary variables where applicable
        """
        self.pnm += self.fnm * self.dt/2
        self.fix_momenta() # also updates bead momenta

    def step0(self, **kwargs):
        """First half of the velocity Verlet algorithm.
        """
        self.ethermo -= RingNM.econs(self)
        self.O() 
        self.B() 
        RingNM.RESPA_propa(self, self.pnm, self.xnm) # free-ring-polymer NM propagation
        self.fix_positions() # also updates bead positions
        self.fix_momenta()   # also updates bead momenta
        self.force_update()
        
    def step1(self, **kwargs):
        """Second half of the velocity Verlet algorithm.
        """
        self.B()
        self.O()
        self.ethermo += RingNM.econs(self)


class SepGLEaux(SepGLEPILE):

    def __init__(self, rpPES, SB, dt, xshape, rng, beta, aux, *args, **kwargs):
        
        """Propagator for a ring-polymer system coupled to a harmonic bath. This propagator uses auxiliary dynamical variables to propagate the GLE.
        
        Args:
            rpPES (Ring): ring-polymerised external potential
            SB (BasePES): a Caldeira--Leggett representation of the system+dissipative environment.
            dt (float): propagation time step
            xshape (ndarray): shape of the array to propagate
            rng (int or Generator): seed for random number generator
            beta (float): reciprocal temperature, 1/kB*T
            aux (list[Dict]): a list of dictionaries for specifying the auxiliary variables parametrisation (see notes below)

        Notes:
            The length of `aux` must be equal to nbeads.
            Each element of aux must be a dictionary with entries `tauD`, `cD`, `tauO`, `omegaO`, `cO`.
            
        """
        
        super().__init__(rpPES, SB, dt, xshape, rng, beta, tau=None, *args, **kwargs)
        self.build_aux(aux)

    def build_aux(self, aux):
        if (naux := len(aux)) != self.nbeads:
            raise ValueError(f"The list of auxiliary variable specs must have {self.nbeads} items, instead got {naux}. Aborting...")
        
        # TODO: auxvars are not ordered in block-diagonal form at the moment - revise
        
        # drift coefficient (exponential prefactor of T-matrix)
        self.ak = []
        # matrix factor in the T-matrix
        self.expA = []
        # fluctuation coefficient (S-matrix)
        self.bk = []
        # auxiliary variable fluctuation frequencies
        self.wk = []
        # list of arrays of auxvars attached to each ring-polymer normal mode
        self.saux = []
        # coupling coefficients
        self.caux = []
        pre_shape = self.replica_layout
        # iterate over the normal modes
        for n,d in enumerate(aux):
            # Load parameters for non-oscillatory (Debye) auxvars
            tauD = _squeeze1d(d.get('tauD', np.array([])))
            cD = _squeeze1d(d.get('cD', np.array([])))
            if (n_debye := len(tauD)) != len(cD):
                raise RuntimeError(f"Inconsistent number of parameters for Debye auxiliaries of mode {n}.")
            # Load params for oscillatory auxvars
            tauO = _squeeze1d(d.get('tauO', np.array([])))
            cO = _squeeze1d(d.get('cO', np.array([])))
            omegaO = _squeeze1d(d.get('omegaO', np.array([])))
            if (n_osc := len(tauO)) != len(cO) or len(cO) != len(omegaO):
                raise RuntimeError(f"Inconsistent number of parameters for oscillatory auxiliaries of mode {n}.")
            self.wk.append(np.asarray(omegaO))
            try:
                self.caux.append(np.concatenate([cD, cO]))
            except ValueError:
                print(f"{cD = }")
                print(f"{cO = }")
                raise
            # Allocate drift coefficients and matrices
            a = np.zeros(pre_shape+(n_debye+2*n_osc,))
            propa = np.zeros(2*(n_debye + 2*n_osc,))
            for i in range(n_debye):
                a[...,i] = np.exp(-self.dt/(2*tauD[i]))
                propa[i,i] = 1
            for i in range(n_osc):
                # s1 variables
                k = i+n_debye
                a[...,k] = np.exp(-self.dt/(2*tauO[i]))
                l = k+n_osc
                a[...,l] = a[...,k]
                w = self.wk[n][i]
                c = np.cos(w*self.dt/2)
                s = np.sin(w*self.dt/2)
                propa[k,k] = c
                propa[k,l] = s
                propa[l,l] = c
                propa[l,k] = -s
            # s1-type variables (all of Debye and s1 oscillatory)
            s1 = np.zeros(pre_shape + (n_debye+n_osc,))
            # s2-type variables - only for oscillatory auxiliaries
            s2 = np.zeros(pre_shape + (n_osc,))
            self.ak.append(a)
            self.expA.append(propa)
            self.bk.append(np.sqrt((1-a**2)/self.beta))
            self.saux.append(np.concatenate([s1,s2], axis=-1))

    def set_pnm(self, pnm):
        super().set_pnm(pnm)
        for s in self.saux:
            s[:] = self.rng.normal(
                scale=np.sqrt(1/self.beta),
                size=s.shape)

    def aux_kinetic_energy(self):
        ans = np.zeros(self.saux[0].shape[:-1])
        for s in self.saux:
            ans += np.sum(s**2, axis=-1) 
        return ans*self.nbeads/2
    
    def O(self):
        """Propagate the Langevin dynamics of the auxiliary variables
        """
        self.ethermo += self.aux_kinetic_energy()
        for s, ak, bk, eA in zip(self.saux, self.ak, self.bk, self.expA):
            self._contract(eA, [0,1], s, [...,1], [...,0], out=s)
            s *= ak
            s += bk * self.rng.normal(size=s.shape)
        self.ethermo -= self.aux_kinetic_energy()

    def B(self):
        """Evolve the momenta under coupling to auxvars and the external force
        """
        
        s1vec = np.zeros(self.replica_layout+(self.nbeads,))
        for n, (s, cvec) in enumerate(zip(self.saux, self.caux)):
            naux = len(cvec)
            s1vec[...,n] = np.sum(s[...,:naux] * cvec, axis=-1)
        Pn = np.reshape(self.pnm, self.replica_layout+(self.nbeads, -1))
        dFdX = np.reshape(self.nmdFdX, self.replica_layout+(self.nbeads, self.nbeads, -1))
        Pn += self.dt/2 * self._contract(dFdX, [...,0,1,2], s1vec, [...,1], [...,0,2]) 
        super().B()
       
    def A(self):
        """Free ring-polymer propagation + auxvar update under coupling
        """
        RingNM.RESPA_propa(self, self.pnm, self.xnm)
        self.fix_positions() # also updates bead positions
        self.fix_momenta()   # also updates bead momenta
        self.force_update()
        # update the s1 under coupling
        Pn = np.reshape(self.pnm/self.m3, self.replica_layout+(self.nbeads, -1))
        dFdX = np.reshape(self.nmdFdX, self.replica_layout+(self.nbeads, self.nbeads, -1))
        arr = self._contract(dFdX, [...,0,1,2], Pn, [...,1,2], [...,0])
        for n, (s, cvec) in enumerate(zip(self.saux, self.caux)):
            naux = len(cvec)
            s[...,:naux] -= self.dt * cvec * arr[...,n,None]

    def step0(self, **kwargs):
        self.ethermo -= RingNM.econs(self) + self.aux_kinetic_energy()
        self.O()
        self.B()
        self.A()

    def step1(self, **kwargs):
        self.B()
        self.O()
        self.ethermo += RingNM.econs(self) + self.aux_kinetic_energy()