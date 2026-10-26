import numpy as np
import pytest
from scipy.integrate import quad
from rpmdgle.propagators.gle import build_Ap_kantorovich
from rpmdgle.utils.ou import get_friction_kernel, get_friction_spectrum

OMEGA = np.array([0.0, 1e-3, 0.3, 0.7, 2.0, 10.0, 1e3])


def make_Ap(theta, A):
    """Ap with zero Markovian friction and antisymmetric coupling θ to the auxiliary block A."""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    Ap = np.zeros((n+1, n+1))
    Ap[0,1:] = theta
    Ap[1:,0] = -theta
    Ap[1:,1:] = A
    return Ap


def jordan(a, n):
    """n x n Jordan block with eigenvalue a and ones on the superdiagonal."""
    return a*np.eye(n) + np.eye(n, k=1)


def jordan_spectrum(omega, a, theta):
    """Re θᵀ (J + iω)⁻¹ θ for a Jordan block J = a + N, from (z + N)⁻¹ = Σ_k (-N)^k / z^(k+1), with z = a + iω."""
    z = a + 1j*np.asarray(omega)
    n = len(theta)
    ans = 0
    for k in range(n):
        ans = ans + (-1)**k * np.dot(theta[:n-k], theta[k:]) / z**(k+1)
    return np.real(ans)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_spectrum_jordan_closed_form(n):
    a = 0.7
    theta = np.linspace(1.3, -0.4, n)
    Ap = make_Ap(theta, jordan(a, n))
    np.testing.assert_allclose(get_friction_spectrum(OMEGA, Ap), jordan_spectrum(OMEGA, a, theta), rtol=1e-12, atol=0)

def test_spectrum_nearly_defective():
    # a perturbation of 1e-14 makes A diagonalisable, but with a nearly singular eigenvector matrix
    a = 0.7
    theta = np.array([1.3, -0.4])
    A = jordan(a, 2)
    A[1,0] = 1e-14
    np.testing.assert_allclose(get_friction_spectrum(OMEGA, make_Ap(theta, A)), jordan_spectrum(OMEGA, a, theta), rtol=1e-12, atol=0)

@pytest.mark.parametrize("w", [0.2, 1.0, 3.0])
def test_spectrum_is_cosine_transform_of_kernel(w):
    # independent route through expm: Λ(ω) = ∫_0^∞ K(t) cos(ωt) dt
    Ap = make_Ap([1.0, 0.2, -0.5], jordan(0.5, 3))
    ref, _ = quad(lambda t: get_friction_kernel(t, Ap), 0, np.inf, weight='cos', wvar=w)
    assert get_friction_spectrum(w, Ap) == pytest.approx(ref, rel=1e-8)

def test_spectrum_kantorovich():
    tauD, cD = np.array([2.0, 0.3]), np.array([0.5, 1.1])
    tauO, omegaO, cO = np.array([1.0, 4.0]), np.array([3.0, 0.5]), np.array([0.8, 0.2])
    Ap = build_Ap_kantorovich(tauD, cD, tauO, omegaO, cO)
    # Debye terms are Lorentzians; an oscillatory term is c² Re[(z, ω₀; -ω₀, z)⁻¹]_11 = c² Re z/(z² + ω₀²), with z = 1/τ + iω
    ref = np.sum(cD**2 * (1/tauD) / ((1/tauD)**2 + OMEGA[:,None]**2), axis=-1)
    z = 1/tauO + 1j*OMEGA[:,None]
    ref += np.sum(cO**2 * np.real(z / (z**2 + omegaO**2)), axis=-1)
    np.testing.assert_allclose(get_friction_spectrum(OMEGA, Ap), ref, rtol=1e-12, atol=0)

def test_spectrum_undamped_zero_mode():
    # the second auxiliary variable does not relax, so A is singular and Λ diverges at ω = 0
    a = 2.0
    theta = np.array([1.0, 0.5])
    Ap = make_Ap(theta, np.diag([a, 0.0]))
    omega = np.array([0.0, 1.0, 3.0])
    spectrum = get_friction_spectrum(omega, Ap)
    assert spectrum[0] == np.inf
    # away from the pole, the undamped mode contributes θ² Re 1/(iω) = 0
    np.testing.assert_allclose(spectrum[1:], theta[0]**2 * a / (a**2 + omega[1:]**2), rtol=1e-14, atol=0)
    assert get_friction_spectrum(0.0, Ap) == np.inf

def test_spectrum_undamped_oscillator():
    w0 = 1.5
    Ap = make_Ap([1.0, 0.0], [[0.0, w0], [-w0, 0.0]])
    spectrum = get_friction_spectrum(np.array([0.5, w0, 2.0]), Ap)
    assert spectrum[1] == np.inf
    np.testing.assert_allclose(spectrum[[0,2]], 0.0, rtol=0, atol=1e-15)

def test_spectrum_shapes():
    Ap = make_Ap([1.3, -0.4], jordan(0.7, 2))
    assert isinstance(get_friction_spectrum(0.5, Ap), float)
    assert get_friction_spectrum(OMEGA, Ap).shape == OMEGA.shape
    with pytest.raises(RuntimeError):
        get_friction_spectrum(OMEGA[None,:], Ap)
