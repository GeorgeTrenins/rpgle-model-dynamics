import numpy as np
import pytest
from scipy.integrate import quad

from rpmdgle.pes.harmonic import PES
from rpmdgle.system import get_bath
from rpmdgle.sysbath.bath import CaldeiraLeggett
from rpmdgle.sysbath.spectral.brownian import Density as Brownian


MASS = 1.7
ETA = 1.3
OMEGA0 = 0.8
NMODES = 32
# underdamped, critically damped and overdamped oscillator modes
GAMMAS = [0.3*OMEGA0, 2.0*OMEGA0, 5.0*OMEGA0]
# just below and just above critical damping
NEAR_CRITICAL = [2*OMEGA0*(1-1e-9), 2*OMEGA0*(1+1e-9)]


def make_density(gamma):
    return Brownian(eta=ETA, omega0=OMEGA0, gamma=gamma)


def make_pes():
    return PES(mass=MASS, hess=[[MASS * 0.5**2]])


@pytest.fixture(
        params=GAMMAS+NEAR_CRITICAL,
        ids=["underdamped", "critical", "overdamped", "near-critical-under", "near-critical-over"])
def density(request):
    return make_density(request.param)


def test_regimes():
    assert [make_density(g)._regime for g in GAMMAS] == ["underdamped", "critical", "overdamped"]


def test_lambda_and_J(density):
    omega = np.linspace(0, 5*OMEGA0, 11)
    assert density.Lambda(0.0) == pytest.approx(ETA, rel=1e-14, abs=0)
    np.testing.assert_allclose(density.J(omega), omega * density.Lambda(omega), rtol=1e-14)
    np.testing.assert_allclose(
        density.Lambda(omega), [density.Lambda(w) for w in omega], rtol=1e-14)


def test_kernel_cosine_transform(density):
    norm = quad(density.Lambda, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    assert density.K(0.0) == pytest.approx(2/np.pi * norm, rel=1e-10, abs=0)
    times = np.array([0.1, 0.5, 2.0, 7.0]) / OMEGA0
    ref = [2/np.pi * quad(density.Lambda, 0, np.inf, weight='cos', wvar=t)[0] for t in times]
    np.testing.assert_allclose(density.K(times), ref, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(density.K(-times), density.K(times), rtol=1e-14)


def test_kernel_static_friction(density):
    assert quad(density.K, 0, np.inf, limit=500)[0] == pytest.approx(ETA, rel=1e-9, abs=0)
    tlong = np.array([1.0e2, 1.0e4, 1.0e6]) / OMEGA0
    Klong = density.K(tlong)
    assert np.all(np.isfinite(Klong))
    if density._regime != "underdamped":
        # no oscillations, so the kernel stays positive
        assert np.all(Klong >= 0)


def test_cdf(density):
    norm = quad(density.Lambda, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    omega = np.array([0.05, 0.4, 0.9, 1.0, 1.1, 2.0, 6.0]) * OMEGA0
    ref = [quad(density.Lambda, 0, w, epsabs=1e-13, epsrel=1e-12)[0] / norm for w in omega]
    np.testing.assert_allclose(density.cdf(omega), ref, rtol=1e-10, atol=1e-14)
    assert density.cdf(0.0) == 0.0
    assert density.cdf(density.wplot) == pytest.approx(0.95, rel=1e-12, abs=0)


@pytest.mark.parametrize("delta", [1e-12, -1e-12])
def test_near_critical_continuity(delta):
    # the physical change is O(delta * (omega0 t)^2) in K and O(delta) in F, well below rtol
    critical = make_density(2*OMEGA0)
    nearby = make_density(2*OMEGA0*(1+delta))
    assert nearby._regime == ("overdamped" if delta > 0 else "underdamped")
    times = np.linspace(0, 20/OMEGA0, 41)
    omega = np.linspace(0, 6*OMEGA0, 41)
    np.testing.assert_allclose(nearby.K(times), critical.K(times), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(nearby.cdf(omega), critical.cdf(omega), rtol=1e-8, atol=1e-14)


def test_discretisation(density):
    bath = CaldeiraLeggett(make_pes(), [(density, NMODES)])
    w = bath.w
    N = bath.Nmodes
    assert N == NMODES
    assert w.shape == (N,)
    assert np.all(np.diff(w) > 0)
    np.testing.assert_allclose(N * density.cdf(w), np.arange(N) + 1/2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        bath.c, w * OMEGA0 * np.sqrt(ETA * MASS / (density.gamma * N)), rtol=1e-13)
    np.testing.assert_allclose(bath.bath_mass, MASS)
    lam = 2 * ETA * OMEGA0**2 / density.gamma
    assert density.exact_reorganisation() == pytest.approx(lam, rel=1e-14, abs=0)
    assert bath.reorganisation_energy() == pytest.approx(lam, rel=1e-12, abs=0)


def test_get_bath():
    bath_data = {"name": "brownian", "Nmodes": NMODES, "eta": ETA, "omega0": OMEGA0, "gamma": 0.5}
    SB = get_bath(make_pes(), bath_data, None)
    assert isinstance(SB, CaldeiraLeggett)
    assert len(SB.densities) == 1
    assert isinstance(SB.densities[0], Brownian)
    assert SB.mass.shape == (1+NMODES,)


@pytest.mark.parametrize("param", ["eta", "omega0", "gamma"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_invalid_parameters(param, value):
    kwargs = dict(eta=ETA, omega0=OMEGA0, gamma=0.5)
    kwargs[param] = value
    with pytest.raises(ValueError):
        Brownian(**kwargs)
