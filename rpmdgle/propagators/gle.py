#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   gle.py
@Time    :   2023/11/30 14:35:08
@Author  :   George Trenins
@Desc    :   Propagators for generalized Langevin dynamics


Key attributes of the SepGLE classes:

  * replica_layout: a tuple describing the layout of independent system realisations; the shape of the configuration array, xshape, is constructed as
  replica_layout + (nbeads,) + PES.extPES.xshape, where the latter is the shape of a single bead configuration expected by the external potential object.

  * F: system-bath coupling function, first appears in Eq. (5) of https://doi.org/10.1103/PhysRevLett.134.226201 ; shape replica_layout + (nbeads,)

  * nmF: linear combinations of F constructed using the ring-polymer normal-mode transformation, Eq. (S13) in https://doi.org/10.1103/PhysRevLett.134.226201 ; shape replica_layout + (nbeads,)

  * dFdX: gradient of the coupling function with respect to system coordinates; shape replica_layout + (nbeads,) + PES.extPES.xshape

  * nmdFdX: derivatives of nmF with respect to normal-mode coordinates; shape replica_layout + (nbeads,nbeads) + PES.extPES.xshape; the storage convention is such that nmdFdX[...,n',n,...] = d nmF^{(n')} / d X^{(n)}, and the array is symmetric in the n,n' indices, see curly braces in Eq. (S20) of https://doi.org/10.1103/PhysRevLett.134.226201

  * PMF: potential of mean field due to coupling to the bath, Eq. (8) of https://doi.org/10.1103/PhysRevLett.134.226201 ; shape = replica_layout

  * nmFMF: force of mean-field, in normal mode coordinates; shape = replica_layout + (nbeads,) + PES.extPES.xshape; defined in Eq. (S20) of https://doi.org/10.1103/PhysRevLett.134.226201   
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from string import ascii_lowercase
from rpmdgle.propagators.verlet import RingNM
from rpmdgle.propagators.langevin import RingPILE
from rpmdgle.utils.nmtrans import MatMulNormalModes
import numpy.typing as npt
from rpmdgle.utils.ou import check_matrix
from scipy.linalg import expm, cholesky

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
            aux (list[Dict] or list[list]): parametrisation of the `Ap` drift matrix 
                (see https://doi.org/10.1021/ct900563s for definitions). Each element 
                in the list corresponds to a different normal-mode index, in the order
                0, -1, +1, -2, +2, ... . For backwards compatibility, the elements may be
                dictionaries (see below). Otherwise, expecting the Ap coefficient matrix, 
                for propagation in mass-weighted coordinates.

        Notes:
            The length of `aux` must be equal to `nbeads`.
            If an element of `aux` is a dictionary, its entries may be `tauD`, `cD`, `tauO`, `omegaO`, or `cO`.
            
        """
        
        super().__init__(rpPES, SB, dt, xshape, rng, beta, tau=None, *args, **kwargs)
        self.build_aux(aux)

    def build_aux(self, aux):
        """Compute the coefficients for propagating GLE dynamics according to Eqs (S38-S42) of https://doi.org/10.1103/PhysRevLett.134.226201 . 

        Args:
            aux (list[Dict] or list[list]): parametrisation of the `Ap` drift matrix 
                (see https://doi.org/10.1021/ct900563s for definitions). Each element 
                in the list corresponds to a different normal-mode index, in the order
                0, -1, +1, -2, +2, ... . 
        """
        if (naux := len(aux)) != self.nbeads:
            raise ValueError(f"The list of auxiliary variable specs must have {self.nbeads} items, instead got {naux}. Aborting...")
        self.Amat = []
        self.theta = []
        self.Tmat = []
        self.Smat = []
        self.saux = []
        for params in aux:
            if isinstance(params, dict):
                # Build Ap matrix from dictionary parameters
                Ap = build_Ap_kantorovich(
                    tauD = _squeeze1d(params.get('tauD', np.array([]))),
                    cD = _squeeze1d(params.get('cD', np.array([]))),
                    tauO = _squeeze1d(params.get('tauO', np.array([]))),
                    omegaO = _squeeze1d(params.get('omegaO', np.array([]))),
                    cO = _squeeze1d(params.get('cO', np.array([])))
                )
            else:
                Ap = np.asarray(params, dtype=float)
                check_matrix(Ap, 'Ap')
                if Ap[0,0] != 0.0:
                    raise NotImplementedError("Expecting zero in the (0,0) position of the Ap matrix. Propagation of dissipative dynamics including a Markovian component is not yet implemented.")
            self.theta.append(np.copy(Ap[0,1:]))
            A = np.copy(Ap[1:,1:])
            self.Amat.append(A)
            T, S = compute_TS_matrices(A, self.dt/2, self.beta)
            self.Tmat.append(T)
            self.Smat.append(S)
            self.saux.append(np.zeros(self.replica_layout+(A.shape[0],)))

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
        """Propagate the OU dynamics of the auxiliary variables, 
        Eq (S38) of https://doi.org/10.1103/PhysRevLett.134.226201.
        """
        self.ethermo += self.aux_kinetic_energy()
        for s, T, S in zip(self.saux, self.Tmat, self.Smat):
            self._contract(T, [0,1], s, [...,1], [...,0], out=s) # drift
            xi = self.rng.normal(size=s.shape)
            s += self._contract(S, [0,1], xi, [...,1], [...,0])  # diffusion
        self.ethermo -= self.aux_kinetic_energy()

    def B(self):
        """Evolve the momenta under coupling to the auxiliary variables and the external force,
        Eqs (S39-S40) of https://doi.org/10.1103/PhysRevLett.134.226201.
        """

        #--- update the momenta under coupling to auxiliary variables, Eq. (S40)
        # mass weighting
        pnm = self.pnm / self.sqm3
        # self.nmdFdX.shape = replica_layout + (nbeads, nbeads, ...), but 
        # self.sqm3.shape = replica_layout + (nbeads, ...), so need to expand dims
        sqm3_ = np.expand_dims(self.sqm3, axis=len(self.replica_layout)+1)
        fnm = self.nmdFdX / sqm3_
        # shape arrays for einsum
        pnm_ = np.reshape(pnm, self.replica_layout+(self.nbeads, -1))
        fnm_ = np.reshape(fnm, self.replica_layout+(self.nbeads, self.nbeads, -1))
        # compute θ^(n) @ s^(n) for all normal modes n
        theta_s = np.zeros(self.replica_layout+(self.nbeads,))
        for n, (s, theta) in enumerate(zip(self.saux, self.theta)):
            theta_s[...,n] = np.sum(s * theta, axis=-1)
        # update momenta under coupling to auxvars
        pnm_ -= self.dt/2 * self._contract(
            fnm_, [...,0,1,2],
            theta_s, [...,1],
            [...,0,2])
        # undo mass weighting
        self.pnm[:] = pnm * self.sqm3
        #--- update momenta under external and mean-field forces
        super().B()
       
    def A(self):
        """Free ring-polymer propagation + auxvar update under coupling
        """

        #--- free ring-polymer normal-mode propagation, Eq. (S41) of https://doi.org/10.1103/PhysRevLett.134.226201
        RingNM.RESPA_propa(self, self.pnm, self.xnm)
        self.fix_positions() # update _bead_ positions
        self.fix_momenta()   # update _bead_ momenta
        self.force_update()  
        #--- propagate auxvars under coupling to system momenta, Eq. (S42)
        # mass weighting
        pnm = self.pnm / self.sqm3
        # self.nmdFdX.shape = replica_layout + (nbeads, nbeads, ...), but 
        # self.sqm3.shape = replica_layout + (nbeads, ...), so need to expand dims
        sqm3_ = np.expand_dims(self.sqm3, axis=len(self.replica_layout)+1)
        fnm = self.nmdFdX / sqm3_
        # shape arrays for einsum
        pnm_ = np.reshape(pnm, self.replica_layout+(self.nbeads, -1))
        fnm_ = np.reshape(fnm, self.replica_layout+(self.nbeads, self.nbeads, -1))
        # compute SUM[ d nmF^{(n)} / d X^{(n')} ⋅ pnm^{(n')} , n' ] 
        dFdX_p = self._contract(
            fnm_, [...,0,1,2],
            pnm_, [...,1,2],
            [...,0])
        # update auxiliary variables under coupling to system momenta
        for n, (s, theta) in enumerate(zip(self.saux, self.theta)):
            s += self.dt * theta * dFdX_p[...,n,None]


    def step0(self, **kwargs):
        self.ethermo -= RingNM.econs(self) + self.aux_kinetic_energy()
        self.O()
        self.B()
        self.A()

    def step1(self, **kwargs):
        self.B()
        self.O()
        self.ethermo += RingNM.econs(self) + self.aux_kinetic_energy()


def build_Ap_kantorovich(
    tauD: npt.ArrayLike, 
    cD: npt.ArrayLike,
    tauO: npt.ArrayLike, 
    omegaO: npt.ArrayLike, 
    cO: npt.ArrayLike
) -> np.ndarray:
    """Build the Ap drift matrix for auxiliary variables according to the Kantorovich parametrisation
    (see 10.1103/PhysRevB.89.134303 for definitions).

    Args:
        tauD (ndarray): relaxation times for Debye (non-oscillatory) auxiliaries
        cD (ndarray): coupling coefficients for Debye auxiliaries
        tauO (ndarray): relaxation times for oscillatory auxiliaries
        omegaO (ndarray): frequencies for oscillatory auxiliaries
        cO (ndarray): coupling coefficients for oscillatory auxiliaries

    Returns:
        ndarray: Ap drift matrix
    """

    tauD = np.asarray(tauD, dtype=float)
    cD = np.asarray(cD, dtype=float)
    if not tauD.ndim == cD.ndim == 1 or tauD.shape != cD.shape:
        raise ValueError("Inconsistent shapes for tauD and cD arrays.")
    tauO = np.asarray(tauO, dtype=float)
    omegaO = np.asarray(omegaO, dtype=float)
    cO = np.asarray(cO, dtype=float)
    if not (tauO.ndim == omegaO.ndim == cO.ndim == 1 and 
            tauO.shape == omegaO.shape == cO.shape):
        raise ValueError("Inconsistent shapes for tauO, omegaO, and cO arrays.")
    n_debye = len(tauD)
    n_osc = len(tauO)
    naux = n_debye + 2 * n_osc
    Ap = np.zeros((naux + 1, naux + 1)) 
    A = Ap[1:, 1:]
    theta = Ap[0, 1:]
    for i in range(n_debye):
        A[i, i] = 1 / tauD[i]
        theta[i] = cD[i]
    for i in range(n_osc):
        i1 = n_debye + 2*i
        i2 = i1 + 1
        A[i1, i1] = 1 / tauO[i]
        A[i2, i2] = 1 / tauO[i]
        A[i1, i2] = omegaO[i]
        A[i2, i1] = -omegaO[i]
        theta[i1] = cO[i]
    Ap[1:,0] = -theta
    return Ap

def compute_TS_matrices(
    A: npt.ArrayLike, 
    dt: float, 
    beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the T and S matrices for propagating auxiliary variables according to 
    Eq (S38) of https://doi.org/10.1103/PhysRevLett.134.226201.

    Args:
        A (ndarray): drift matrix for auxiliary variables, shape (naux, naux)
        dt (float): time step for propagating the decoupled OU dynamics of the auxiliary variables
        beta (float): reciprocal temperature, 1/kB*T
    Returns:
        tuple[ndarray, ndarray]: T and S matrices for the OU propagation step
    """
    A = np.asarray(A, dtype=float)
    check_matrix(A, 'A')
    T = expm(-A*dt)
    S = compute_S_from_T(T, beta)
    return T, S

def compute_S_from_T(
    T: npt.ArrayLike,
    beta: float, *, rtol: float = 1e-12
) -> np.ndarray:
    """
    Compute S such that S @ S.T = (1/beta) * (I - T @ T.T).

    Parameters
    ----------
    T : (d, d) array_like
        The matrix T = exp(-A τ).
    beta : float
        Reciprocal temperature, 1/kB*T.
    rtol : float
        Relative tolerance used for eigenvalue clipping in the PSD fallback.

    Returns
    -------
    S : (d, d) ndarray
        A factor satisfying S @ S.T = (1/beta) * (I - T @ T.T).
    """
    T = np.asarray(T, dtype=float)
    check_matrix(T, 'T')
    naux = T.shape[0]
    C = (1/beta) * (np.eye(naux) - T @ T.T)
    try:
        # may fail if positive semi-definite (PSD)
        return cholesky(C, lower=True, check_finite=False)
    except:
        # PSD fallback: eigen-decomposition with clipping
        w, V = np.linalg.eigh(C)
        # clip small negative eigenvalues (roundoff) to zero
        w_clip = np.clip(w, 0.0, np.max(w) * rtol if np.max(w) > 0 else 0.0)
        # construct S = V @ sqrt(Λ) @ V.T, so that S @ S.T = V @ Λ @ V.T = C
        return V @ (np.sqrt(w_clip)[:, None] * V.T)
