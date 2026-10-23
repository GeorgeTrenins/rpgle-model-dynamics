import numpy as np
import pytest

from rpmdgle import numderiv
from rpmdgle.pes.harmonic import PES
from rpmdgle.pes.pi import Ring
from rpmdgle.propagators.gle import SepGLEPILE
from rpmdgle.sysbath.bath import CaldeiraLeggett
from rpmdgle.sysbath.coupling.exptanh import Coupling as ExpTanhCoupling
from rpmdgle.sysbath.spectral import brownian, debye, expohmic, splined


MASS = 1.7
XGRID = np.linspace(0.0, 5.0, 51)

# Bath frequencies w and coupling coefficients c for six modes, computed before the
# spectral densities were separated from the bath (commit 3c664b4), with the
# parameters in make_density
REFERENCE = {
    "expohmic": (
        [0.06960910159170385, 0.23014565796142472, 0.4311972005861495, 0.70037498988312,
         1.1090354888959124, 1.9879253198304003],
        [0.030148908603690825, 0.0996800741390727, 0.18675898256656343, 0.3033445494261869,
         0.4803424958577174, 0.8610049175763621]),
    "debye": (
        [0.10532199806991667, 0.3313708498984761, 0.6138615903831683, 1.0425802982729644,
         1.9313708498984756, 6.076603290180115],
        [0.0571721592422295, 0.17987872757647536, 0.33322376370771906, 0.5659460314843627,
         1.0484100549715591, 3.2985752010459186]),
    "brownian": (
        [0.1637762798829499, 0.42585727460751427, 0.5954770379421017, 0.7204615374832165,
         0.8491534966657069, 1.1151370760270003],
        [0.11245441185815519, 0.2924082131168892, 0.4088749611645488, 0.4946936059483173,
         0.5830579197004903, 0.7656913694428977]),
    "splined": (
        [0.08639885562956806, 0.28543847249311677, 0.5341950900077855, 0.8660796944847616,
         1.3662819797599624, 2.413407216589022],
        [0.04169658481870881, 0.13775424908242417, 0.25780562390486805, 0.4179750435084069,
         0.6593755442732764, 1.1647242081543203]),
}


def make_pes():
    return PES(mass=MASS, hess=[[MASS * 0.5**2]])


def make_density(name):
    if name == "expohmic":
        return expohmic.Density(eta=1.3, omega_cut=0.8)
    elif name == "debye":
        return debye.Density(eta=1.3, omega_cut=0.8)
    elif name == "brownian":
        return brownian.Density(eta=1.3, omega0=0.8, gamma=0.5)
    elif name == "splined":
        return splined.Density(eta=1.3, eps=1e-10, x=XGRID, y=np.exp(-XGRID))
    raise ValueError(name)


@pytest.fixture
def terms():
    """Three spectral densities with distinct frequency scales and their numbers of modes"""
    return [
        (expohmic.Density(eta=1.3, omega_cut=0.8), 5),
        (brownian.Density(eta=0.4, omega0=1.5, gamma=0.3), 3),
        (debye.Density(eta=0.7, omega_cut=0.2), 4),
    ]


@pytest.fixture
def bath(terms):
    return CaldeiraLeggett(make_pes(), terms)


@pytest.mark.parametrize("name", list(REFERENCE))
def test_single_term_regression(name):
    bath = CaldeiraLeggett(make_pes(), [(make_density(name), 6)])
    w, c = REFERENCE[name]
    np.testing.assert_allclose(bath.w, w, rtol=1e-14)
    np.testing.assert_allclose(bath.c, c, rtol=1e-14)
    assert bath.Nmodes == 6
    assert bath.term_slices == (slice(0, 6),)
    assert bath.reorganisation_energy() == pytest.approx(
        bath.exact_reorganisation(), rel=1e-12, abs=0)


def test_sum_layout(bath, terms):
    N = 5 + 3 + 4
    assert bath.Nmodes == N
    assert bath.term_slices == (slice(0, 5), slice(5, 8), slice(8, 12))
    for arr in (bath.c, bath.w, bath.bath_mass, bath.ww, bath.mww):
        assert arr.shape == (N,)
    assert bath.mass.shape == (1+N,)
    np.testing.assert_allclose(bath.bath_mass, MASS)
    # each block is the discretisation of its own term
    for term, sl in zip(terms, bath.term_slices):
        single = CaldeiraLeggett(make_pes(), [term])
        np.testing.assert_array_equal(bath.w[sl], single.w)
        np.testing.assert_array_equal(bath.c[sl], single.c)
        lam, nmodes = term[0].exact_reorganisation(), term[1]
        np.testing.assert_allclose(
            bath.c[sl]**2 / bath.mww[sl], lam / (2*nmodes), rtol=1e-13)


def test_sum_spectrum(bath, terms):
    densities = [density for density, _ in terms]
    omega = np.linspace(0.0, 4.0, 17)
    times = np.linspace(0.0, 10.0, 11)
    for method, x in (("J", omega), ("Lambda", omega), ("K", times)):
        expected = sum(getattr(density, method)(x) for density in densities)
        np.testing.assert_allclose(getattr(bath, method)(x), expected, rtol=1e-14)
        # scalars, as passed by scipy.integrate.quad
        assert np.isscalar(getattr(bath, method)(x[3]))
    lam = sum(density.exact_reorganisation() for density in densities)
    assert bath.exact_reorganisation() == pytest.approx(lam, rel=1e-14, abs=0)
    assert bath.Lambda(0.0) == pytest.approx(1.3 + 0.4 + 0.7, rel=1e-14, abs=0)
    assert bath.K(0.0) == pytest.approx(lam/2, rel=1e-14, abs=0)
    assert bath.wplot == max(density.wplot for density in densities)


def test_sum_quadrature(bath):
    lam = bath.exact_reorganisation()
    assert bath.reorganisation_energy() == pytest.approx(lam, rel=1e-13, abs=0)
    assert bath.l_quadrature(np.ones(bath.Nmodes)) == pytest.approx(np.pi*lam/4, rel=1e-13, abs=0)


def test_potential_and_gradient(terms):
    """A sum of terms is the same as the system coupled to independent baths"""
    pes = make_pes()
    F = ExpTanhCoupling(eps1=-0.5, eps2=0.3, delta=1.2)
    bath = CaldeiraLeggett(pes, terms, coupling=F)
    singles = [CaldeiraLeggett(pes, [term], coupling=F) for term in terms]
    rng = np.random.default_rng(31415)
    x = rng.normal(size=(4, 3, 1+bath.Nmodes))
    q = x[..., :1]
    Vsys = pes.potential(q)
    expected = Vsys + sum(
        single.potential(np.concatenate([q, x[..., 1:][..., sl]], axis=-1)) - Vsys
        for single, sl in zip(singles, bath.term_slices))
    np.testing.assert_allclose(bath.potential(x), expected, rtol=1e-13)
    pot, grad = bath.both(x)
    np.testing.assert_allclose(pot, bath.potential(x), rtol=1e-14)
    np.testing.assert_allclose(grad, bath.gradient(x), rtol=1e-14)
    x0 = x[0, 0]
    fd = numderiv.gradient(bath.potential, x0, order=4)
    np.testing.assert_allclose(bath.gradient(x0), fd, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(
        "coupling", [None, ExpTanhCoupling(eps1=-0.5, eps2=0.3, delta=1.2)], ids=["linear", "exptanh"])
def test_unbatched_configuration(terms, coupling):
    """A single configuration gives the same results as the same configuration in a batch"""
    bath = CaldeiraLeggett(make_pes(), terms, coupling=coupling)
    rng = np.random.default_rng(2718)
    x = rng.normal(size=(4, 3, 1+bath.Nmodes))
    x0 = x[1, 2]
    pot = bath.potential(x0)
    assert np.ndim(pot) == 0
    np.testing.assert_allclose(pot, bath.potential(x)[1, 2], rtol=1e-14)
    grad = bath.gradient(x0)
    assert grad.shape == (1+bath.Nmodes,)
    np.testing.assert_allclose(grad, bath.gradient(x)[1, 2], rtol=1e-14)
    pot_both, grad_both = bath.both(x0)
    assert np.ndim(pot_both) == 0
    np.testing.assert_allclose(pot_both, pot, rtol=1e-14)
    np.testing.assert_allclose(grad_both, grad, rtol=1e-14)
    r_eq = bath.bath_eq(x0[:1])
    assert r_eq.shape == (bath.Nmodes,)
    np.testing.assert_allclose(r_eq, bath.bath_eq(x[..., :1])[1, 2], rtol=1e-14)


def test_pmf_additivity(terms):
    """The potential of mean field is linear in the squared couplings of the bath modes"""
    pes = make_pes()
    beta = 2.0
    nbeads = 4

    def alpha(bath):
        rpPES = Ring(nbeads, (1,), pes, beta)
        propa = SepGLEPILE(rpPES, bath, 0.1, (2, nbeads, 1), rng=0, beta=beta)
        return propa._alpha

    total = alpha(CaldeiraLeggett(pes, terms))
    parts = [alpha(CaldeiraLeggett(pes, [term])) for term in terms]
    assert total.shape == (nbeads,)
    np.testing.assert_allclose(total, sum(parts), rtol=1e-13)


@pytest.mark.parametrize("nmodes", [0, -2, 2.0, True, "4"])
def test_invalid_nmodes(nmodes):
    with pytest.raises(ValueError):
        CaldeiraLeggett(make_pes(), [(make_density("expohmic"), 4), (make_density("debye"), nmodes)])


def test_no_terms():
    with pytest.raises(ValueError):
        CaldeiraLeggett(make_pes(), [])


def test_coupling_units():
    from rpmdgle.sysbath.coupling.linear import Coupling as LinearCoupling
    with pytest.raises(RuntimeError):
        CaldeiraLeggett(make_pes(), [(make_density("expohmic"), 4)], coupling=LinearCoupling(UNITS="kcalAfs"))
