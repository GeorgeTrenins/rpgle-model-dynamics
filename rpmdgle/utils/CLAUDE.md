# CLAUDE.md — `rpmdgle/utils`

Self-contained numerical helpers. When you add a submodule, add its name to both `__all__` and the `TYPE_CHECKING` import block.

| Module | Contents | Used by |
|---|---|---|
| `arrays` | `slice_along_axis`, `idx_along_axis`, `append_dims`, `string_to_slice`, `index_in_slice` | `pes/pi.py`, `propagators/verlet.py`, `langevin.py`, `properties.py`, `rfft`, `tcfs`, `visualisation/lineplot.py` |
| `rfft` | `RealFFT`, a wrapper for real FFTs along one axis | `nmtrans` |
| `nmtrans` | `FFTNormalModes`, `MatMulNormalModes` | `propagators/verlet.py` (FFT), `propagators/gle.py` (MatMul) |
| `tcfs` | `correlate`, `time_averaged_correlation` | tests, `notebooks/thermal_tcfs.ipynb` |
| `ou` | analytical GLE/OU covariances, friction kernel and spectrum of an `Ap` matrix | `propagators/gle.py` (`check_matrix`), `visualisation/auxplot.py`, `test_sepgleaux.py`, notebooks |
| `kernel_transforms` | friction kernels of non-centroid modes; DCT kernel ↔ spectrum helpers | `sysbath/spectral/splined.py` (`shift_lambda_to_kernel` with `n=0` only) |
| `calculus` | `GL_quad`, `FFTFourierIntegral` | `rates/rpmdrate.py` (`GL_quad`) |
| `kde` | kernel density estimate | `rates/analyse.py` |
| `special` | `sinc` and its derivatives, overflow-safe `logcosh` and `logsinh` | nothing in the package |

## Normal modes (`nmtrans`, `rfft`)

- `forward_matrix[k, j]` for bead `j = 0..N-1`, in the mode order `0, -1, +1, -2, +2, ...`:
  - mode $0$: $1/N$
  - mode $-k$: $(\sqrt{2}/N)\cos(2\pi jk/N)$
  - mode $+k$: $(\sqrt{2}/N)\sin(2\pi jk/N)$
  - for even $N$, the last mode is $-N/2$, with entries $(-1)^j/N$

  `backward_matrix = N * forward_matrix.T`. The forward transform therefore gives the centroid as mode 0, and the backward transform inverts it exactly.
- `nm_freqs = 2 sin(π|k|/N)` is dimensionless. Multiply it by $\omega_N = N/(\beta\hbar)$, as `RingNM.parse_mscale` does.
- `FFTNormalModes` calls `scipy.fftpack.rfft`, whose packed output `[y0, Re y1, Im y1, Re y2, ...]` lines up with the mode order. `_norm` supplies the signs and the factor $\sqrt{2}/N$. `RealFFT.__init__` is never called. `RealFFT` reads the bead axis from `self.axis`, which `BaseNMTransform` sets, so renaming that attribute in either class breaks the FFT path.
- With `nmodes < nbeads`, `cart2mats` keeps the first `nmodes` modes in this order, which are the lowest frequencies, and `mats2cart` pads the rest with zeros. The round trip is exact only when `nmodes == nbeads`.
- `cart2mats(cart, mats=None)` and `mats2cart(mats, cart=None)` write into the second argument when it is given. Use this to fill preallocated propagator arrays in place. `mats2cart(..., overwrite_mats=True)` divides its input in place and leaves it rescaled.
- `bead_ax` may be positive or negative. The propagators pass `len(xshape) - len(rpxshape)`.
- `MatMulNormalModes` exposes `forward_matrix` with shape `(nmodes, nbeads)` and `backward_matrix` with shape `(nbeads, nmodes)`. `gle.py` builds `coupling_grad_mat` from both.
- No pytest test covers this module. `python rpmdgle/utils/nmtrans.py` runs the `__main__` self-check, which asserts that the two classes agree for odd, even and truncated cases.

## Time-correlation functions (`tcfs`)

- `correlate(A, B, axis=0)` returns $C[k] = \langle A[t]\,B[t+k]\rangle_t$. Each lag is divided by its number of overlapping pairs, so the estimate is unbiased. The output has `len(B)` points along `axis`. If `A` is longer, it is truncated to `len(B)`. If `A` is shorter, the divisor is capped at `len(A)`.
  - The default axis is **0**, unlike most of the package, which defaults to `-1`.
  - `A` and `B` must have the same `ndim`.
  - The inputs must be real; nothing is conjugated.
- `time_averaged_correlation(A, B, lag, axis=0)` returns `lag + 1` points. It splits the time origins into consecutive non-overlapping blocks of `lag + 1`, correlates each block against a window of `B` with `2*lag + 1` points, and averages the blocks. Every origin in a complete block contributes to every lag, and tail points that do not complete a block are dropped.
  - It needs at least `2*lag + 1` points along `axis`. With fewer, it returns all NaN, because the block count is zero.
  - The result is allocated as a real array.
- `tests/utils/test_tcfs.py` checks both functions against explicit loops.

## GLE and OU reference results (`ou`)

The notation follows Ceriotti et al. (doi:10.1021/ct900563s). `Ap` is the drift matrix of $(p, \mathbf{s})$ with shape `(1+naux, 1+naux)`. Below, $\theta$ = `Ap[0, 1:]` and $A$ = `Ap[1:, 1:]`.

- `get_friction_kernel(t, Ap)` $= \theta^\mathsf{T} e^{-|t|A}\theta$.
- `get_friction_spectrum(ω, Ap)` $= \theta^\mathsf{T} A (A^2+\omega^2)^{-1}\theta = \int_0^\infty K(t)\cos\omega t\,\mathrm{d}t$.

  These two functions use the same normalisation as `Density.K` and `Density.Lambda`, so the results compare directly with a bath, as `auxplot.py` does. Each takes a scalar, which returns a float, or a 1-D array. They read only `Ap[0, 1:]` and `Ap[1:, 1:]`, so they require `Ap[0, 0] = 0` and `Ap[1:, 0] = -θ`, which `build_Ap_kantorovich` and the test fixtures produce. `check_Ap_structure` enforces this in both functions, and also in `SepGLEaux.build_aux`, which likewise reads only θ and A. A non-zero `Ap[0, 0]` (Markovian friction) raises `NotImplementedError`; the test is exact. A coupling that is not antisymmetric raises `ValueError`, with relative tolerance 1e-12.
- **Two mass conventions for `Ap`.**
  - `SepGLEaux`, `parse_aux_spec`, and `auxplot.py`: $\theta^\mathsf{T} e^{-tA}\theta$ is the friction kernel *including* the mass, in the same units as `Density.K` (mass/time²).
  - `gle_cxx`: the dynamics run in mass-weighted coordinates, so $\theta$ must be per unit mass. `test_sepgleaux.py::build_Ap_s1s2` divides $\theta$ by $\sqrt{m}$ before calling `gle_cxx`. Do the same whenever you compare a propagator's `Ap` with `gle_cxx` at a mass other than 1.
- `gle_cxx(t, omega, Ap, Cp=None, C0=None, u=0, beta=None, mass=1.0)` returns $\langle v(t)\,v(u)^\mathsf{T}\rangle$ for $v = (q, p, \mathbf{s})$ of a harmonic oscillator.
  - Pass exactly one of `beta` and `Cp`. `beta` selects the canonical stationary state. `Cp` gives a general stationary covariance of $(p, \mathbf{s})$, and β is then taken as `1/Cp[0,0]`.
  - `C0` sets a non-stationary covariance at time 0. A 1×1 or 2×2 `C0` sets only the $(q, p)$ block, and the rest is filled with $\mathbb{1}/\beta$.
  - `t` is a scalar, a 1-D array, or a 2-D array whose columns correspond to the elements of a 1-D `u`. `u` is a scalar or a 1-D array.
  - The result has shape `S + (2+naux, 2+naux)`. If `t` and `u` are both 1-D, the result is their outer product, with `S = (len(t), len(u))`. Otherwise, `t` and `u` broadcast as NumPy arrays. In both cases `t` varies along the leading axis of `S` and `u` along the trailing axis.
  - Without `C0`, the process is stationary and `t` and `u` may take any real value. With `C0`, the process starts at time 0, so a negative time raises `RuntimeError`.
  - $q$ and $p$ are returned in physical units (rescaled by `mass`). The auxiliary components stay mass-weighted.
- `get_Cqp` solves $A_{qp}C + CA_{qp}^\mathsf{T} = D_{qp}$ with `scipy.linalg.solve_continuous_lyapunov` which stays accurate when $A_{qp}$ is defective. If two eigenvalues of $A_{qp}$ sum to nearly zero, there is no stationary state and `get_Cqp` raises a `ValueError`. It symmetrises the result, $(C + C^\mathsf{T})/2$.
- `wnle_cqq(t, beta, omega, tau, m=1)` gives $\langle q(t)q(0)\rangle$ under white-noise Langevin dynamics with friction $\gamma = 1/\tau$.
  - `omega` and `tau` must be scalars. All four parameters must be positive, or it raises `ValueError`.
  - `gle_cxx(t, omega, [[1/tau]], beta=beta, mass=m)[..., 0, 0]` gives the same function by an independent route, which makes it a useful check.

## Non-centroid friction kernels (`kernel_transforms`)

Normal mode $n$ sees the frequency-shifted bath $J_n(\omega) = J\big(\sqrt{\omega^2-\omega_n^2}\big)$ for $\omega > \omega_n$, and $0$ below $\omega_n$. Here $\omega_n = 2\omega_P\sin(\pi n/P)$ and $\omega_P = P/(\beta\hbar)$.

- `shift_lambda_to_kernel(Lambda, UNITS, T, P, n, wmax, eps, tvec)` returns `(ω_n, K_n(tvec))`, where $K_n(t) = \frac{2}{\pi}\int \Lambda_n(\omega)\cos\omega t\,\mathrm{d}\omega$.
  - `wmax` is the cutoff of the *unshifted* Λ, so the upper limit of the integral is $\sqrt{\omega_n^2+w_\mathrm{max}^2}$.
  - `eps` is the absolute tolerance passed to `quad`. `T` is in kelvin.
  - `quad` runs once per time point, so the function is slow on long time grids.
  - With `n = 0`, `UNITS`, `T` and `P` are not used. `splined.Density.K` passes `None` for all three.
  - For negative `n`, the returned ω_n is negative, but the kernel equals the one for |n|.
- `shift_spectral_to_kernel` does the same starting from J. Nothing calls it.
- The DCT helpers work on uniform grids that start at 0. They apply the trapezoid rule through a type-I DCT, which matches the `Density` convention $K = \frac{2}{\pi}\int\Lambda\cos$, $\Lambda = \int K\cos$.
  - `kernel_to_lambda(kvec, tvec, hh)` and `kernel_to_spectral(...)` return values on the grid `timefreqs(tvec)`, which is $\pi k / t_\mathrm{max}$.
  - `lambda_to_kernel(lvec, wvec)` and `spectral_to_kernel(jvec, wvec)` return K on `timefreqs(wvec)`.
  - Truncation at the end of the grid causes errors of the order of the missing tail.
  - `shift_spectral` evaluates $J_n$ on a grid.
  - None of these helpers is used in the package.

## Smaller helpers

- `calculus.GL_quad(n, lo, hi)` returns Gauss–Legendre nodes and weights on `[lo, hi]`. `rpmdrate.py --ti` uses it and writes `GL_quad.csv`.
- `calculus.FFTFourierIntegral(lo, hi, npts, pad)` implements the endpoint-corrected FFT of Numerical Recipes §13.9 for $I(k) = \int_a^b e^{-2\pi i k x} f(x)\,\mathrm{d}x$.
  - `k` is in cycles per unit x, not angular.
  - The data must be `npts` equally spaced points that include both endpoints.
  - The results correspond to `self.freqs[axis]`: ascending, centred on 0, with `npts` values spaced by `1/(pad·npts·h)`.
  - The constructor's per-axis parameters are indexed by the axis numbers passed in `axes`.
  - Nothing in the package uses it and no test covers it. A manual check on $x^2$ over $[0, 1]$ agreed with `quad` to 1e-14.
- `kde.kde(data, grid, bw, kernel="gaussian", axis=-1)` averages `kernel(data, grid, bw)` over `axis`, so the caller has to broadcast the arguments, as in `kde(data[..., None], xgrid, bw, axis=0)`.
  - `kde()` looks the kernel up by name in `KERNELS`. The `@register` decorator fills that dictionary, under the function name minus its `_kernel` suffix. The registered kernels are `gaussian`, `tophat`, `triangular`, `epanechnikov` and `tricube`. `analyse.py` uses `epanechnikov`.
  - To add a kernel, define `<name>_kernel(x, mu, sigma)` and decorate it with `@register`; `kde()` itself needs no change. An unknown name raises `ValueError`, and the message lists the registered kernels.
- `special.sinc(x, nu=0)` is the *unnormalised* $\sin x / x$; `scipy.special.sinc` is $\sin \pi x/\pi x$. It returns the derivative of any order `nu` ≥ 0 in terms of scipy's `spherical_jn`
  - Points x whose real part has its sign bit set (i.e., real part is negative) are evaluated at −x and multiplied by $(-1)^\nu$. This is needed because scipy's 
    ```py
    spherical_jn(n, -(a+0j)) = -spherical_jn(n, -a-0j) \
                            = -spherical_jn(n, -np.conj(a+0j))
    ```
    With the fix, $f(-x) = (-1)^\nu f(x)$ and $f(\bar x) = \overline{f(x)}$ hold bitwise for all x.
- `special.logcosh` and `special.logsinh` take real input. `logsinh` always returns a complex array, adding $i\pi$ for $x < 0$, so take `.real` for $x > 0$. `logsinh(0)` is `-inf`.
- `arrays.slice_along_axis` and `arrays.idx_along_axis` use basic indexing and return views. Writes through them reach the original array, which `pes/pi.py` relies on.
- `arrays.append_dims(arr, n)` appends `n` size-1 axes, so that per-mode or per-bead arrays broadcast against configurations.
- `arrays.string_to_slice` returns an `int` (`'1'`), a `list` of ints (`'0,1,3'`) or a `slice` (`'2:-1:2'`, `':'`), so callers must handle all three.
- `arrays.index_in_slice(slice_obj, index, length=None)` tells whether `seq[index]` is among the elements of `seq[slice_obj]`.
  - Without `length` it reads the slice literally as `range(start, stop, step)`. This is valid only for non-negative index, `start` and `stop` and a positive step, and anything negative raises `ValueError`. The rule is conservative: it also rejects some cases whose answer does not depend on the length, such as `slice(None)` with index −1.
  - A zero step raises `ValueError` in both modes, as `seq[::0]` does.


## Testing

Only `tcfs`, `special` and `arrays.index_in_slice` have direct tests. `test_sepgleaux.py` uses `ou.gle_cxx` as a reference, so that function is also exercised. Put new tests in `tests/utils/test_<module>.py` and keep the file name unique across `tests/` (see `tests/CLAUDE.md`).