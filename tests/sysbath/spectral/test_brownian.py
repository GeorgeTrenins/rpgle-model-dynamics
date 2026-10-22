import numpy as np
import pytest
from scipy.integrate import quad

from rpmdgle.pes.harmonic import PES
from rpmdgle.system import get_bath
from rpmdgle.sysbath.spectral.brownian import Density as BrownianBath


MASS = 1.7
ETA = 1.3
OMEGA0 = 0.8
NMODES = 32
# underdamped, critically damped and overdamped oscillator modes
GAMMAS = [0.3*OMEGA0, 2.0*OMEGA0, 5.0*OMEGA0]
# just below and just above critical damping
NEAR_CRITICAL = [2*OMEGA0*(1-1e-9), 2*OMEGA0*(1+1e-9)]


def make_bath(gamma, Nmodes=NMODES):
    pes = PES(mass=MASS, hess=[[MASS * 0.5**2]])
    return BrownianBath(pes, Nmodes, eta=ETA, omega0=OMEGA0, gamma=gamma)


@pytest.fixture(
        params=GAMMAS+NEAR_CRITICAL,
        ids=["underdamped", "critical", "overdamped", "near-critical-under", "near-critical-over"])
def bath(request):
    return make_bath(request.param)


def test_regimes():
    assert [make_bath(g)._regime for g in GAMMAS] == ["underdamped", "critical", "overdamped"]


def test_lambda_and_J(bath):
    omega = np.linspace(0, 5*OMEGA0, 11)
    assert bath.Lambda(0.0) == pytest.approx(ETA, rel=1e-14, abs=0)
    np.testing.assert_allclose(bath.J(omega), omega * bath.Lambda(omega), rtol=1e-14)
    np.testing.assert_allclose(
        bath.Lambda(omega), [bath.Lambda(w) for w in omega], rtol=1e-14)


def test_kernel_cosine_transform(bath):
    norm = quad(bath.Lambda, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    assert bath.K(0.0) == pytest.approx(2/np.pi * norm, rel=1e-10, abs=0)
    times = np.array([0.1, 0.5, 2.0, 7.0]) / OMEGA0
    ref = [2/np.pi * quad(bath.Lambda, 0, np.inf, weight='cos', wvar=t)[0] for t in times]
    np.testing.assert_allclose(bath.K(times), ref, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(bath.K(-times), bath.K(times), rtol=1e-14)


def test_kernel_static_friction(bath):
    assert quad(bath.K, 0, np.inf, limit=500)[0] == pytest.approx(ETA, rel=1e-9, abs=0)
    tlong = np.array([1.0e2, 1.0e4, 1.0e6]) / OMEGA0
    Klong = bath.K(tlong)
    assert np.all(np.isfinite(Klong))
    if bath._regime != "underdamped":
        # no oscillations, so the kernel stays positive
        assert np.all(Klong >= 0)


def test_cdf(bath):
    norm = quad(bath.Lambda, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    omega = np.array([0.05, 0.4, 0.9, 1.0, 1.1, 2.0, 6.0]) * OMEGA0
    ref = [quad(bath.Lambda, 0, w, epsabs=1e-13, epsrel=1e-12)[0] / norm for w in omega]
    np.testing.assert_allclose(bath.cdf(omega), ref, rtol=1e-10, atol=1e-14)
    assert bath.cdf(0.0) == 0.0
    assert bath.cdf(bath.wmax) == pytest.approx(0.95, rel=1e-12, abs=0)


@pytest.mark.parametrize("delta", [1e-12, -1e-12])
def test_near_critical_continuity(delta):
    # the physical change is O(delta * (omega0 t)^2) in K and O(delta) in F, well below rtol
    critical = make_bath(2*OMEGA0)
    nearby = make_bath(2*OMEGA0*(1+delta))
    assert nearby._regime == ("overdamped" if delta > 0 else "underdamped")
    times = np.linspace(0, 20/OMEGA0, 41)
    omega = np.linspace(0, 6*OMEGA0, 41)
    np.testing.assert_allclose(nearby.K(times), critical.K(times), rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(nearby.cdf(omega), critical.cdf(omega), rtol=1e-8, atol=1e-14)


def test_discretisation(bath):
    w = bath.w
    N = bath.Nmodes
    assert w.shape == (N,)
    assert np.all(np.diff(w) > 0)
    np.testing.assert_allclose(N * bath.cdf(w), np.arange(N) + 1/2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        bath.c, w * OMEGA0 * np.sqrt(ETA * MASS / (bath.gamma * N)), rtol=1e-13)
    np.testing.assert_allclose(bath.bath_mass, MASS)
    lam = 2 * ETA * OMEGA0**2 / bath.gamma
    assert bath.exact_reorganisation() == pytest.approx(lam, rel=1e-14, abs=0)
    assert bath.reorganisation_energy() == pytest.approx(lam, rel=1e-12, abs=0)


def test_get_bath():
    pes = PES(mass=MASS, hess=[[MASS * 0.5**2]])
    bath_data = {"name": "brownian", "Nmodes": NMODES, "eta": ETA, "omega0": OMEGA0, "gamma": 0.5}
    SB = get_bath(pes, bath_data, None)
    assert isinstance(SB, BrownianBath)
    assert SB.mass.shape == (1+NMODES,)


@pytest.mark.parametrize("param", ["eta", "omega0", "gamma"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_invalid_parameters(param, value):
    pes = PES(mass=MASS, hess=[[MASS * 0.5**2]])
    kwargs = dict(eta=ETA, omega0=OMEGA0, gamma=0.5)
    kwargs[param] = value
    with pytest.raises(ValueError):
        BrownianBath(pes, NMODES, **kwargs)
