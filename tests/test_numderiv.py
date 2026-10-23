import numpy as np
import pytest

from rpmdgle import numderiv


X = np.random.default_rng(271828).normal(size=(4, 3, 2))


def fun(x):
    """f(x) = x0⁴ + x0 x1 + sin(x1), evaluated for each configuration x[..., :]"""
    return x[..., 0]**4 + x[..., 0]*x[..., 1] + np.sin(x[..., 1])


def grad(x):
    return np.stack([4*x[..., 0]**3 + x[..., 1], x[..., 0] + np.cos(x[..., 1])], axis=-1)


def hess(x):
    H = np.zeros(x.shape + (2,))
    H[..., 0, 0] = 12*x[..., 0]**2
    H[..., 0, 1] = H[..., 1, 0] = 1.0
    H[..., 1, 1] = -np.sin(x[..., 1])
    return H


def assert_close(actual, expected, tol):
    """Compare to within tol relative to the largest element of expected"""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol*np.max(np.abs(expected)))


@pytest.mark.parametrize("order, h, tol", [(2, 1e-4, 1e-8), (4, 1e-3, 1e-11)])
def test_scalar_function(order, h, tol):
    x = X.copy()
    jac = numderiv.batch_jacobian(fun, x, order=order, h=h)
    assert jac.shape == X.shape
    assert_close(jac, grad(X), tol)
    np.testing.assert_array_equal(x, X)


@pytest.mark.parametrize("order, h, tol", [(2, 1e-4, 1e-8), (4, 1e-3, 1e-11)])
def test_vector_function(order, h, tol):
    jac = numderiv.batch_jacobian(grad, X, order=order, h=h)
    assert jac.shape == X.shape + (2,)
    assert_close(jac, hess(X), tol)


def test_single_configuration():
    jac = numderiv.batch_jacobian(fun, X, order=4, h=1e-3)
    x0 = X[1, 2]
    np.testing.assert_allclose(numderiv.batch_jacobian(fun, x0, order=4, h=1e-3), jac[1, 2], rtol=1e-14)
    # a scalar is a configuration with one coordinate
    jac0 = numderiv.batch_jacobian(lambda x: np.sin(x[..., 0]), 0.3, order=4, h=1e-3)
    assert jac0.shape == (1,)
    assert jac0[0] == pytest.approx(np.cos(0.3), rel=1e-11, abs=0)


def test_matches_gradient():
    """Same result as numderiv.gradient applied to one configuration at a time"""
    jac = numderiv.batch_jacobian(fun, X, order=4, h=1e-3)
    for idx in np.ndindex(X.shape[:-1]):
        np.testing.assert_allclose(
            jac[idx], numderiv.gradient(fun, X[idx], order=4, h=1e-3), rtol=1e-12)


@pytest.mark.parametrize("order", [1, 3, 6])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        numderiv.batch_jacobian(fun, X, order=order)
