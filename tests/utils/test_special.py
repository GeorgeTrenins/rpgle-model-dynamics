import functools
import math
from fractions import Fraction
import mpmath
import numpy as np
import pytest
from scipy.special import spherical_jn
from rpmdgle.utils.special import sinc, _sinc_coeffs

NMAX = 30
XREF = (0.0,) + tuple(s*v for v in (1e-8, 1e-3, 0.1, 1.0, 5.0, 20.0, 100.0, 1000.0) for s in (1, -1))
ZREF = tuple(complex(x, y) for x in (0.0, 1e-3, 1.0, 5.0, 20.0, 1000.0)
             for y in (1e-8, 0.1, 1.0, 5.0, 20.0, 100.0, 700.0))
ZREF += (-5+20j, -0.1-20j, 100-1j, -1000-700j, -1e-3+1e-8j, -20-0.1j)


@functools.lru_cache(maxsize=None)
def mp_sinc_derivative(n, x, dps=50):
    """n'th derivative of sin(x)/x from mpmath at `dps` digits, rounded to float or complex."""
    with mpmath.workdps(dps):
        d = mpmath.diff(mpmath.sinc, mpmath.mpmathify(x), n)
    return complex(d) if isinstance(x, complex) else float(d)


def taylor_sinc_derivative(n, x, nterms=12):
    """n'th derivative of the Taylor series sin(x)/x = sum_k (-1)^k x^(2k) / (2k+1)!."""
    ans = 0.0
    kmin = (n+1)//2
    for k in reversed(range(kmin, kmin+nterms)):
        c = Fraction((-1)**k * math.factorial(2*k),
                     math.factorial(2*k-n) * math.factorial(2*k+1))
        ans = ans + float(c) * x**(2*k-n)
    return ans


def test_coeffs_low_order():
    assert _sinc_coeffs(0) == (Fraction(-1),)
    assert _sinc_coeffs(1) == (Fraction(3, 5), Fraction(-2, 5))
    assert _sinc_coeffs(2) == (Fraction(-3, 7), Fraction(4, 9), Fraction(-8, 63))

def test_coeffs_sum_to_one():
    for m in range(201):
        coeffs = _sinc_coeffs(m)
        assert len(coeffs) == m+1
        assert sum(abs(c) for c in coeffs) == 1
        assert all(math.isfinite(float(c)) for c in coeffs)

def test_explicit_expressions():
    x = np.array([-150.0, -7.3, -0.4, -1e-3, 1e-3, 0.5, 2.0, 11.0, 150.0])
    j1, j3, j5 = (spherical_jn(l, x) for l in (1, 3, 5))
    np.testing.assert_allclose(sinc(x, 1), -j1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sinc(x, 3), -2/5*j3 + 3/5*j1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(sinc(x, 5), -8/63*j5 + 4/9*j3 - 3/7*j1, rtol=0, atol=1e-15)

@pytest.mark.parametrize("n", range(NMAX+1))
def test_against_mpmath(n):
    x = np.array(XREF)
    ref = np.array([mp_sinc_derivative(n, xi) for xi in XREF])
    err = np.abs(sinc(x, n) - ref)
    tol = 1e-14 * (n+1)
    assert np.all(err <= tol), f"max error {err.max():.3e} at x = {x[np.argmax(err)]} exceeds {tol:.1e}"

@pytest.mark.parametrize("n", range(NMAX+1))
def test_against_mpmath_complex(n):
    z = np.array(ZREF)
    ref = np.array([mp_sinc_derivative(n, zi) for zi in ZREF])
    # |j_l|, |j'_l| and the result are all bounded by sinh|y|/|y| with y = Im z;
    # scipy's complex j_l loses about two digits for 5 <= |y| <= 20
    y = np.abs(z.imag)
    err = np.abs(sinc(z, n) - ref) / (np.sinh(y) / y)
    tol = 5e-14 * (n+1)
    assert np.all(err <= tol), f"max scaled error {err.max():.3e} at z = {z[np.argmax(err)]} exceeds {tol:.1e}"

@pytest.mark.parametrize("n", range(NMAX+1))
def test_small_x_taylor(n):
    r = np.geomspace(1e-12, 9e-3, 31)
    x = np.concatenate([-r[::-1], r])
    z = np.outer(r, np.exp(1j*np.linspace(0, 2*np.pi, 8, endpoint=False))).ravel()
    # relative tolerance: for odd n the derivative vanishes linearly at x = 0
    for arg in (x, z):
        np.testing.assert_allclose(sinc(arg, n), taylor_sinc_derivative(n, arg), rtol=1e-13, atol=0)

@pytest.mark.parametrize("n", range(NMAX+1))
def test_complex_on_real_axis(n):
    # conj(x + 0j) has Im = -0.0, where scipy alone negates j_l for x < 0
    x = np.array([-1000.0, -20.0, -2.0, -0.1, -1e-8, 1e-8, 0.1, 2.0, 20.0, 1000.0])
    expected = sinc(x, n)
    for z in (x + 0j, np.conj(x + 0j)):
        out = sinc(z, n)
        assert np.all(out.imag == 0)
        np.testing.assert_allclose(out.real, expected, rtol=0, atol=1e-14*(n+1))

@pytest.mark.parametrize("n", range(NMAX+1))
def test_at_zero(n):
    expected = 0.0 if n % 2 else (-1)**(n//2) / (n+1)
    assert sinc(0.0, n) == expected
    assert sinc(0, n) == expected
    assert sinc(0j, n) == expected
    v = sinc(np.array([[-1.0, 0.0], [0.0, 2.0]]), n)
    assert v[0, 1] == expected and v[1, 0] == expected
    assert np.all(np.isfinite(v))
    v = sinc(np.array([-1j, 0j, complex(-0.0, -0.0)]), n)
    assert v[1] == expected and v[2] == expected
    assert np.all(np.isfinite(v))

@pytest.mark.parametrize("n", range(NMAX+1))
def test_symmetries(n):
    x = np.geomspace(1e-8, 1e3, 67)
    np.testing.assert_array_equal(sinc(-x, n), (-1)**n * sinc(x, n))
    z = np.outer(x[::6], np.exp(1j*np.linspace(0, 2*np.pi, 12, endpoint=False))).ravel()
    z = np.concatenate([x + 0j, 1j*x, z])
    z = z[np.abs(z.imag) <= 700]
    np.testing.assert_array_equal(sinc(-z, n), (-1)**n * sinc(z, n))
    np.testing.assert_array_equal(sinc(z.conj(), n), sinc(z, n).conj())

@pytest.mark.parametrize("n", [0, 1, 2, 7])
def test_shape(n):
    assert isinstance(sinc(1.3, n), float)
    assert isinstance(sinc(1.3+0.2j, n), complex)
    assert np.shape(sinc(np.float64(1.3), n)) == ()
    for shape in [(0,), (5,), (3, 4), (2, 1, 3)]:
        x = np.linspace(-4.0, 4.0, math.prod(shape)).reshape(shape)
        for arg in (x, x*(1+0.5j)):
            out = sinc(arg, n)
            assert out.shape == shape
            np.testing.assert_array_equal(out.ravel(), [sinc(a, n) for a in arg.ravel()])
    np.testing.assert_array_equal(sinc([-1, 0, 2], n), sinc(np.array([-1.0, 0.0, 2.0]), n))

@pytest.mark.parametrize("x, dtype", [(1.5, np.float64), (2, np.float64), (np.float32(1.5), np.float64),
                                      (1.5+0j, np.complex128), (np.complex64(1.5+1j), np.complex128)])
def test_result_type(x, dtype):
    for n in (0, 1, 2):
        out = sinc(x, n)
        assert isinstance(out, dtype)
        assert out == sinc(dtype(x), n)
        assert sinc(np.array([x, x]), n).dtype == dtype

def test_integer_types_for_order():
    assert sinc(1.7, np.int64(3)) == sinc(1.7, 3)

@pytest.mark.parametrize("nu", [-1, 1.5, 2.0, True, "2", None])
def test_invalid_order(nu):
    with pytest.raises(ValueError):
        sinc(1.0, nu)
