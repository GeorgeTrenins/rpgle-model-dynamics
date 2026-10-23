import numpy as np
import pytest

from rpmdgle.pes._base import BasePES
from rpmdgle.sysbath.coupling.exptanh import Coupling as ExpTanhCoupling
from rpmdgle.sysbath.coupling.linear import Coupling as LinearCoupling


X = np.random.default_rng(161803).normal(size=(5, 4, 2))


class PotentialOnly(BasePES):
    """V(x) = x0⁴ + x0 x1 + sin(x1); gradient and Hessian come from the fallbacks"""

    def potential(self, x):
        return x[..., 0]**4 + x[..., 0]*x[..., 1] + np.sin(x[..., 1])


class AnalyticGradient(PotentialOnly):
    """Same potential with an analytic gradient; only the Hessian comes from the fallback"""

    def gradient(self, x):
        return np.stack([4*x[..., 0]**3 + x[..., 1], x[..., 0] + np.cos(x[..., 1])], axis=-1)


def hess(x):
    H = np.zeros(x.shape + (2,))
    H[..., 0, 0] = 12*x[..., 0]**2
    H[..., 0, 1] = H[..., 1, 0] = 1.0
    H[..., 1, 1] = -np.sin(x[..., 1])
    return H


def exptanh_d2F(q, eps1, eps2, delta):
    """Second derivative of the exptanh coupling F(q) = q (1 + ε1 exp(-y²/2) + ε2 tanh y), y = q/δ"""
    y = q/delta
    return (eps1*np.exp(-y**2/2)*y*(y**2 - 3) + 2*eps2*(1 - y*np.tanh(y))/np.cosh(y)**2)/delta


def assert_close(actual, expected, tol):
    """Compare to within tol relative to the largest element of expected"""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol*np.max(np.abs(expected)))


def test_gradient_fallback():
    pes = PotentialOnly()
    g = pes.gradient(X)
    assert g.shape == X.shape
    assert_close(g, AnalyticGradient().gradient(X), 1e-11)
    np.testing.assert_allclose(pes.gradient(X[2, 3]), g[2, 3], rtol=1e-14)
    np.testing.assert_array_equal(pes.force(X), -g)


@pytest.mark.parametrize("cls, tol", [(PotentialOnly, 1e-7), (AnalyticGradient, 1e-11)],
                         ids=["potential-only", "analytic-gradient"])
def test_hessian_fallback(cls, tol):
    pes = cls()
    H = pes.hessian(X)
    assert H.shape == X.shape + (2,)
    np.testing.assert_array_equal(H, np.swapaxes(H, -1, -2))
    assert_close(H, hess(X), tol)
    np.testing.assert_allclose(pes.hessian(X[2, 3]), H[2, 3], rtol=1e-14)
    V, G, H2 = pes.all(X)
    np.testing.assert_array_equal(V, pes.potential(X))
    np.testing.assert_array_equal(G, pes.gradient(X))
    np.testing.assert_array_equal(H2, H)


def test_coupling_hessians():
    q = np.random.default_rng(2).normal(size=(6, 1))
    H = LinearCoupling().hessian(q)
    assert H.shape == (6, 1, 1)
    np.testing.assert_array_equal(H, 0.0)
    eps1, eps2, delta = -0.5, 0.3, 1.2
    F = ExpTanhCoupling(eps1=eps1, eps2=eps2, delta=delta)
    H = F.hessian(q)
    assert H.shape == (6, 1, 1)
    np.testing.assert_allclose(H[..., 0], exptanh_d2F(q, eps1, eps2, delta), rtol=0, atol=1e-11)
    assert F.hessian(q[0]).shape == (1, 1)
