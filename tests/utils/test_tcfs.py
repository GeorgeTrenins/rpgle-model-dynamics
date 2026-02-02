import numpy as np
import pytest
from rpmdgle.utils import tcfs

def test_correlate_equal_size():
    rng = np.random.default_rng(123)
    A = rng.normal(size=200)
    B = rng.normal(size=200)
    N = len(A)
    ref = np.zeros(N)
    for lag in range(N):
        ref[lag] = np.sum(A[:N-lag] * B[lag:]) / (N-lag)
    out = tcfs.correlate(A, B)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_correlate_A_longer_than_B():
    rng = np.random.default_rng(456)
    A = rng.normal(size=250)
    B = rng.normal(size=200)
    N = len(B)
    ref = np.zeros(N)
    for lag in range(N):
        ref[lag] = np.sum(A[:N-lag] * B[lag:]) / (N-lag)
    out = tcfs.correlate(A, B)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_correlate_A_shorter_than_B():
    rng = np.random.default_rng(789)
    A = rng.normal(size=150)
    B = rng.normal(size=200)
    N = len(B)
    ref = np.zeros(N)
    for lag in range(N):
        n_overlap = min(len(A), N-lag)
        if n_overlap > 0:
            ref[lag] = np.sum(A[:n_overlap] * B[lag:lag+n_overlap]) / n_overlap
        else:
            ref[lag] = 0.0
    out = tcfs.correlate(A, B)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_correlate_multidim_axis0():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(100, 3))
    B = rng.normal(size=(100, 3))
    N = A.shape[0]
    ref = np.zeros((N, 3))
    for lag in range(N):
        ref[lag] = np.sum(A[:N-lag, :] * B[lag:, :], axis=0) / (N-lag)
    out = tcfs.correlate(A, B, axis=0)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_correlate_multidim_axis1():
    rng = np.random.default_rng(43)
    A = rng.normal(size=(3, 100))
    B = rng.normal(size=(3, 100))
    N = A.shape[1]
    ref = np.zeros((3, N))
    for lag in range(N):
        ref[:, lag] = np.sum(A[:, :N-lag] * B[:, lag:], axis=1) / (N-lag)
    out = tcfs.correlate(A, B, axis=1)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_time_averaged_correlation():
    rng = np.random.default_rng(321)
    A = rng.normal(size=50)
    B = rng.normal(size=50)
    lag = 5
    # Reference: explicit time-averaged correlation
    nwin = 0
    ref = np.zeros(lag+1)
    stride = lag + 1
    while True:
        a_start = nwin * stride
        b_start = nwin * stride
        a_end = a_start + stride
        b_end = b_start + stride + lag
        if a_end > len(A) or b_end > len(B):
            break
        a_wk = A[a_start:a_end]
        b_wk = B[b_start:b_end]
        for t in range(lag+1):
            ref[t] += np.sum(a_wk * b_wk[t:t+stride]) / stride
        nwin += 1
    ref /= nwin
    out = tcfs.time_averaged_correlation(A, B, lag)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_time_averaged_correlation_multidim():
    rng = np.random.default_rng(322)
    A = rng.normal(size=(30, 2))
    B = rng.normal(size=(30, 2))
    lag = 4
    nwin = 0
    ref = np.zeros((lag+1, 2))
    stride = lag + 1
    while True:
        a_start = nwin * stride
        b_start = nwin * stride
        a_end = a_start + stride
        b_end = b_start + stride + lag
        if a_end > A.shape[0] or b_end > B.shape[0]:
            break
        a_wk = A[a_start:a_end, :]
        b_wk = B[b_start:b_end, :]
        for t in range(lag+1):
            ref[t] += np.sum(a_wk * b_wk[t:t+stride, :], axis=0) / stride
        nwin += 1
    ref /= nwin
    out = tcfs.time_averaged_correlation(A, B, lag, axis=0)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

def test_time_averaged_correlation_multidim_axis1():
    rng = np.random.default_rng(323)
    A = rng.normal(size=(2, 30))
    B = rng.normal(size=(2, 30))
    lag = 4
    nwin = 0
    ref = np.zeros((2, lag+1))
    stride = lag + 1
    while True:
        a_start = nwin * stride
        b_start = nwin * stride
        a_end = a_start + stride
        b_end = b_start + stride + lag
        if a_end > A.shape[1] or b_end > B.shape[1]:
            break
        a_wk = A[:, a_start:a_end]
        b_wk = B[:, b_start:b_end]
        for t in range(lag+1):
            ref[:, t] += np.sum(a_wk * b_wk[:, t:t+stride], axis=1) / stride
        nwin += 1
    ref /= nwin
    out = tcfs.time_averaged_correlation(A, B, lag, axis=1)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
