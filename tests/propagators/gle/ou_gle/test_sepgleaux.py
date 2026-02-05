import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rpmdgle.pes.harmonic import PES
from rpmdgle.pes.pi import Ring
from rpmdgle.propagators.gle import SepGLEaux
from rpmdgle.sysbath.coupling.linear import Coupling as LinearCoupling
from rpmdgle.sysbath.spectral.expohmic import Density as ExpOhmicBath
from rpmdgle.utils import tcfs
from rpmdgle.utils.ou import gle_cxx


@pytest.fixture
def kantorovich_params(tmp_path):
    """Fixture: Write Kantorovich parametrization and return its path.
    
    Specifies one Debye mode and two oscillatory modes for auxiliary variables.
    """
    aux = [{
        "tauD": [1.0],
        "cD": [0.8],
        "tauO": [0.5, 0.3],
        "omegaO": [0.5, 1.5],
        "cO": [0.5, 0.3]
    }]
    aux_path = tmp_path / "aux.json"
    with open(aux_path, "w") as f:
        json.dump(aux, f)
    return aux_path


@pytest.fixture
def system_params(tmp_path):
    """Fixture: Write system parameters for a 1D harmonic oscillator.
    
    Returns the path to a JSON file with mass, frequency, inverse temperature,
    time step, and number of integration steps.
    """
    params = {
        "mass": 1.5,
        "omega": 0.7,
        "beta": 2.0,
        "dt": 0.05,
        "nsteps": 10000
    }
    sys_path = tmp_path / "system.json"
    with open(sys_path, "w") as f:
        json.dump(params, f)
    return sys_path


@pytest.fixture
def direct_matrix_params(tmp_path):
    """Fixture: Generate a positive-definite Ap matrix and serialize to JSON.
    
    Creates a 4×4 Ap matrix (3 auxiliary variables) with guaranteed positive-definite
    A block via eigendecomposition.
    """
    naux = 3
    Ap = build_positive_definite_A(naux, seed=12345)
    
    params = {"aux": [Ap.tolist()]}
    matrix_path = tmp_path / "Ap_direct.json"
    with open(matrix_path, "w") as f:
        json.dump(params, f)
    return matrix_path


def build_positive_definite_A(naux: int, seed: int = 42) -> np.ndarray:
    r"""Build a positive-definite A matrix and construct the full Ap matrix.
    
    Constructs Ap via eigendecomposition: $A = Q \Lambda Q^T$ where $\Lambda$ has
    positive diagonal entries and $Q$ is orthogonal.
    
    Args:
        naux (int): Size of the A matrix (number of auxiliary variables).
        seed (int): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        ndarray: Ap drift matrix of shape (naux+1, naux+1) with structure:
                 - Ap[0,0] = 0 (required for non-Markovian dynamics)
                 - Ap[1:,1:] = positive-definite A matrix
                 - Ap[0,1:] = random coupling vector
                 - Ap[1:,0] = -Ap[0,1:]
    """
    rng = np.random.default_rng(seed)
    
    # Generate random orthogonal matrix via QR decomposition
    G = rng.standard_normal(size=(naux, naux))
    Q, _ = np.linalg.qr(G)
    
    # Diagonal matrix with positive eigenvalues
    Lambda = np.diag(0.5 + rng.uniform(size=naux))
    # Construct A = Q @ Lambda @ Q.T
    A = Q @ Lambda @ Q.T
    
    # Build the full Ap matrix
    Ap = np.zeros((naux + 1, naux + 1))
    Ap[1:, 1:] = A
    Ap[0, 1:] = rng.uniform(size=naux, low=-1.0, high=1.0)
    Ap[1:, 0] = -Ap[0, 1:]
    
    return Ap


def build_Ap_s1s2(
        tauD: list, 
        cD: list, 
        tauO: list, 
        omegaO: list, 
        cO: list, 
        mass: float) -> np.ndarray:
    """Build the Ap drift matrix from Kantorovich parameters in s1/s2 ordering.
    
    Constructs the full (1+naux) × (1+naux) Ap matrix where naux = len(tauD) + 2*len(tauO).
    The structure matches the convention used by [`SepGLEaux.build_aux`](rpmdgle/propagators/gle.py).
    
    Args:
        tauD (list): Debye relaxation times.
        cD (list): Debye coupling coefficients.
        tauO (list): Oscillatory relaxation times.
        omegaO (list): Oscillatory frequencies.
        cO (list): Oscillatory coupling coefficients.
        mass (float): System mass (for scaling).
        
    Returns:
        ndarray: Ap drift matrix with mass-weighted coupling terms.
    """
    n_debye = len(tauD)
    n_osc = len(tauO)
    naux = n_debye + 2 * n_osc
    
    Ap = np.zeros((1 + naux, 1 + naux))
    A = Ap[1:, 1:]
    theta = Ap[0, 1:]
    
    # Debye (diagonal) modes
    for i in range(n_debye):
        A[i, i] = 1 / tauD[i]
        theta[i] = cD[i]
    
    # Oscillatory (2×2 block) modes
    for i in range(n_osc):
        i1 = n_debye + 2 * i
        i2 = i1 + 1
        A[i1, i1] = 1 / tauO[i]
        A[i2, i2] = 1 / tauO[i]
        A[i1, i2] = omegaO[i]
        A[i2, i1] = -omegaO[i]
        theta[i1] = cO[i]
    
    # Mass-weighting for coupling to system-bath gradient
    Ap[0, 1:] /= np.sqrt(mass)
    Ap[1:, 0] = -Ap[0, 1:]
    
    return Ap


def plot_cross_correlations(
        tvec: np.ndarray, 
        corr: np.ndarray, 
        ref: np.ndarray, 
        filename: str = "cross_correlation_debug.eps") -> None:
    """Plot numerical and analytical cross-correlation functions.
    
    Creates a figure with one subplot per variable, displaying both numerical
    (from trajectory) and analytical (from theory) autocorrelations.
    
    Args:
        tvec (ndarray): Time points, shape (max_lag+1,).
        corr (ndarray): Numerical cross-correlations, shape (max_lag+1, nvars).
        ref (ndarray): Analytical cross-correlations, shape (max_lag+1, nvars).
        filename (str): Output filename. Defaults to "cross_correlation_debug.eps".
    """
    nvars = corr.shape[1]
    fig, axes = plt.subplots(
        nvars, 1, 
        figsize=(7, 2.5 * nvars), 
        sharex=True)
    
    if nvars == 1:
        axes = [axes]
    
    for i in range(nvars):
        axes[i].plot(tvec, corr[:, i], label="Numerical", color="C0")
        axes[i].plot(tvec, ref[:, i], label="Analytical", color="C1", linestyle="--")
        
        if i == 0:
            axes[i].set_ylabel("$C_{xx}(t)$")
        elif i == 1:
            axes[i].set_ylabel("$C_{pp}(t)$")
        else:
            axes[i].set_ylabel(f"$C_{{s_{i-2}s_{i-2}}}(t)$")
        
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(filename, format="eps")
    plt.close(fig)


def test_sepgleaux_kantorovich(kantorovich_params, system_params):
    """Test GLE propagation using Kantorovich parametrization.
    
    Validates that the propagator correctly samples the canonical ensemble
    and produces cross-correlations that match analytical predictions from
    generalized Langevin equation theory.
    """
    # --- 1. Load parameters ---
    with open(kantorovich_params, "r") as f:
        aux = json.load(f)
    with open(system_params, "r") as f:
        sys = json.load(f)
    
    mass = sys["mass"]
    omega = sys["omega"]
    beta = sys["beta"]
    dt = sys["dt"]
    nsteps = sys["nsteps"]
    nbeads = 1
    nrep = 400
    rng = np.random.default_rng(42)
    xshape = (nrep, nbeads, 1)

    # --- 2. System and propagator setup ---
    pes = PES(mass=mass, hess=[[mass * omega**2]])
    Nmodes = 1
    eta = 1.0
    omega_cut = 5.0
    
    bath = ExpOhmicBath(pes, Nmodes, eta, omega_cut, coupling=LinearCoupling())
    rpPES = Ring(Nmodes, (1,), pes, beta)
    propa = SepGLEaux(rpPES, bath, dt, xshape, rng, beta, aux)

    # --- 3. Initialize from thermal distribution ---
    propa.set_x(rng.normal(
        loc=0.0,
        scale=np.sqrt(1 / (beta * mass * omega**2)),
        size=propa.x.shape
    ))
    propa.set_p(rng.normal(
        loc=0.0,
        scale=np.sqrt(mass / beta),
        size=propa.p.shape
    ))
    for s in propa.saux:
        s[:] = rng.normal(
            loc=0.0,
            scale=np.sqrt(1 / beta),
            size=s.shape
        )

    naux = len(aux[0]["tauD"]) + 2 * len(aux[0]["tauO"])
    nvars = 2 + naux

    # --- 4. Simulate trajectory and write to temp file ---
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        for i in range(nsteps):
            arr = np.empty((nrep, nvars), dtype=np.float32)
            arr[:, 0] = propa.x[:, 0, 0]
            arr[:, 1] = propa.p[:, 0, 0]
            arr[:, 2:] = propa.saux[0]
            arr.tofile(tf)
            propa.step()
        tf.flush()
        
        # --- 5. Compute numerical cross-correlation ---
        max_lag = 500
        with open(tf.name, "rb") as f:
            mm = np.memmap(
                f, 
                dtype=np.float32, 
                mode="r", 
                shape=(nsteps, nrep, nvars))
            corr_ = tcfs.time_averaged_correlation(mm, mm, max_lag, axis=0)
            corr = np.mean(corr_, axis=1)  # average over replicas
            del mm

    # --- 6. Analytical cross-correlation ---
    tauD = aux[0]["tauD"]
    cD = aux[0]["cD"]
    tauO = aux[0]["tauO"]
    cO = aux[0]["cO"]
    omegaO = aux[0]["omegaO"]
    Ap_full = build_Ap_s1s2(tauD, cD, tauO, omegaO, cO, mass)
    
    tvec = np.arange(max_lag + 1) * dt
    ref = np.array([
        np.diag(gle_cxx([t], omega, Ap_full, beta=beta, mass=mass)) 
        for t in tvec
    ])

    # --- 7. Compare numerical and analytical results ---
    plot_cross_correlations(tvec, corr, ref, filename="cross_correlation_kantorovich.eps")
    np.testing.assert_allclose(corr, ref, rtol=0.01, atol=0.01)


def test_sepgleaux_direct_matrix(direct_matrix_params, system_params):
    """Test GLE propagation using direct matrix specification.
    
    Validates that the propagator correctly handles arbitrary positive-definite
    A matrices provided as direct input (not via Kantorovich parametrization).
    Cross-correlations are compared against analytical predictions.
    """
    # --- 1. Load parameters ---
    with open(direct_matrix_params, "r") as f:
        data = json.load(f)
    with open(system_params, "r") as f:
        sys = json.load(f)
    
    aux = data["aux"]
    mass = sys["mass"]
    omega = sys["omega"]
    beta = sys["beta"]
    dt = sys["dt"]
    nsteps = sys["nsteps"]
    nbeads = 1
    nrep = 400
    rng = np.random.default_rng(42)
    xshape = (nrep, nbeads, 1)

    # --- 2. System and propagator setup ---
    pes = PES(mass=mass, hess=[[mass * omega**2]])
    Nmodes = 1
    eta = 1.0
    omega_cut = 5.0
    
    bath = ExpOhmicBath(pes, Nmodes, eta, omega_cut, coupling=LinearCoupling())
    rpPES = Ring(Nmodes, (1,), pes, beta)
    propa = SepGLEaux(rpPES, bath, dt, xshape, rng, beta, aux)

    # --- 3. Initialize from thermal distribution ---
    propa.set_x(rng.normal(
        loc=0.0,
        scale=np.sqrt(1 / (beta * mass * omega**2)),
        size=propa.x.shape
    ))
    propa.set_p(rng.normal(
        loc=0.0,
        scale=np.sqrt(mass / beta),
        size=propa.p.shape
    ))
    for s in propa.saux:
        s[:] = rng.normal(
            loc=0.0,
            scale=np.sqrt(1 / beta),
            size=s.shape
        )

    # Number of auxiliary variables from the Ap matrix
    naux = np.array(aux[0]).shape[0] - 1
    nvars = 2 + naux

    # --- 4. Simulate trajectory and write to temp file ---
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        for i in range(nsteps):
            arr = np.empty((nrep, nvars), dtype=np.float32)
            arr[:, 0] = propa.x[:, 0, 0]
            arr[:, 1] = propa.p[:, 0, 0]
            arr[:, 2:] = propa.saux[0]
            arr.tofile(tf)
            propa.step()
        tf.flush()
        
        # --- 5. Compute numerical cross-correlation ---
        max_lag = 500
        with open(tf.name, "rb") as f:
            mm = np.memmap(
                f, 
                dtype=np.float32, 
                mode="r", 
                shape=(nsteps, nrep, nvars))
            corr_ = tcfs.time_averaged_correlation(mm, mm, max_lag, axis=0)
            corr = np.mean(corr_, axis=1)  # average over replicas
            del mm

    # --- 6. Analytical cross-correlation ---
    Ap_full = np.array(aux[0], dtype=float)
    Ap_full[0, 1:] /= np.sqrt(mass)  # ensure mass-weighting is consistent with SepGLEaux
    Ap_full[1:, 0] /= np.sqrt(mass)
    
    tvec = np.arange(max_lag + 1) * dt
    ref = np.array([
        np.diag(gle_cxx([t], omega, Ap_full, beta=beta, mass=mass)) 
        for t in tvec
    ])

    # --- 7. Compare numerical and analytical results ---
    plot_cross_correlations(tvec, corr, ref, filename="cross_correlation_direct_matrix.eps")
    np.testing.assert_allclose(corr, ref, rtol=0.01, atol=0.01)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
    # from pathlib import Path
    # root = Path(__file__).parent
    # kantorovich_params = root / "kantorovich.json"
    # system_params = root / "system.json"
    # test_sepgleaux_kantorovich(kantorovich_params, system_params)