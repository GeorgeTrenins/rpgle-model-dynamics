import numpy as np
import json
import pytest
import tempfile

from rpmdgle.pes.harmonic import PES
from rpmdgle.propagators.gle import SepGLEaux
from rpmdgle.pes.pi import Ring
from rpmdgle.utils.ou import gle_cxx
from rpmdgle.sysbath.coupling.linear import Coupling as LinearCoupling
from rpmdgle.sysbath.spectral.expohmic import Density as ExpOhmicBath
from rpmdgle.utils import tcfs

@pytest.fixture
def aux_params(tmp_path):
    """Fixture: Write a Debye + 2 oscillatory auxvar parameter file and return its path."""
    aux = [{
        "tauD": [1.0],
        "cD": [0.8],
        "tauO": [0.5, 0.3],
        "omegaO": [2.0, 3.5],
        "cO": [0.5, 0.3]
    }]
    aux_path = tmp_path / "aux.json"
    with open(aux_path, "w") as f:
        json.dump(aux, f)
    return aux_path

@pytest.fixture
def system_params(tmp_path):
    """Fixture: Write system parameters for a 1D harmonic oscillator and return its path."""
    params = {
        "mass": 1.5,
        "omega": 0.7,
        "beta": 2.0,
        "dt": 0.02,
        "nsteps": 10000
    }
    sys_path = tmp_path / "system.json"
    with open(sys_path, "w") as f:
        json.dump(params, f)
    return sys_path

def build_Ap_s1s2(tauD, cD, tauO, omegaO, cO, mass):
    """Build the drift matrix Ap in s1/s2 ordering to match SepGLEaux."""
    n_debye = len(tauD)
    n_osc = len(tauO)
    naux = n_debye + 2 * n_osc
    Ap = np.zeros((1 + naux, 1 + naux))  # [p, s1..., s2...]

    # Debye (s1)
    for i in range(n_debye):
        Ap[1 + i, 1 + i] = 1 / tauD[i]
    # Oscillatory (s1/s2 pairs)
    for i in range(n_osc):
        s1_idx = n_debye + i
        s2_idx = n_debye + n_osc + i
        Ap[1 + s1_idx, 1 + s1_idx] = 1 / tauO[i]
        Ap[1 + s2_idx, 1 + s2_idx] = 1 / tauO[i]
        Ap[1 + s1_idx, 1 + s2_idx] = omegaO[i]
        Ap[1 + s2_idx, 1 + s1_idx] = -omegaO[i]

    # Coupling vector (cD, cO) in s1/s2 ordering
    cvec = np.zeros(naux)
    for i in range(n_debye):
        cvec[i] = cD[i]
    for i in range(n_osc):
        cvec[n_debye + i] = cO[i]

    Ap_full = np.zeros((1 + naux, 1 + naux))
    Ap_full[0, 1:] = cvec / np.sqrt(mass)
    Ap_full[1:, 0] = -Ap_full[0, 1:]
    Ap_full[1:, 1:] = Ap[1:, 1:]
    return Ap_full

def test_sepgleaux_ou_regression(aux_params, system_params):
    # --- 1. Load parameters ---
    with open(aux_params, "r") as f:
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
    # These are not invoked in the dynamics
    eta = 1.0
    omega_cut = 5.0
    # 
    bath = ExpOhmicBath(pes, Nmodes, eta, omega_cut, coupling=LinearCoupling())
    rpPES = Ring(Nmodes, (1,), pes, beta)
    propa = SepGLEaux(rpPES, bath, dt, xshape, rng, beta, aux)

    # --- 3. Initialize from thermal distribution ---
    propa.set_x(rng.normal(
        loc=0.0,
        scale=np.sqrt(1/(beta * mass * omega**2)),
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
            scale=np.sqrt(1/beta),
            size=s.shape
        )

    naux = len(aux[0]["tauD"]) + 2*len(aux[0]["tauO"])
    nvars = 2 + naux

    # --- 4. Simulate trajectory and write to temp file ---
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        for i in range(nsteps):
            arr = np.empty((nrep, nvars), dtype=np.float32)
            arr[:, 0] = propa.x[:,0,0]
            arr[:, 1] = propa.p[:,0,0]
            arr[:, 2:] = propa.saux[0]
            arr.tofile(tf)
            propa.step()
        tf.flush()
        # --- 5. Compute numerical cross-correlation ---
        max_lag = 500
        with open(tf.name, "rb") as f:
            mm = np.memmap(f, dtype=np.float32, mode="r", shape=(nsteps, nrep, nvars))
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
    tvec = np.arange(max_lag+1) * dt
    ref = np.array([
        np.diag(gle_cxx([t], omega, Ap_full, beta=beta, mass=mass)) for t in tvec
    ])

    # --- 7. Compare numerical and analytical results ---
    np.testing.assert_allclose(corr, ref, rtol=0.01, atol=0.01)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])