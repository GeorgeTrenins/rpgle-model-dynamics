# CLAUDE.md — `rpmdgle/utils`

Self-contained numerical helpers. `__init__.py` imports submodules lazily: a module-level `__getattr__` (PEP 562) loads a submodule on first access, so after `import rpmdgle.utils` the expression `rpmdgle.utils.ou.gle_cxx` works, but only the submodules actually used are imported. `__all__` lists the submodules and nothing else; it does not re-export functions or classes. When you add a submodule, add its name to both `__all__` and the `TYPE_CHECKING` import block, or it stays invisible to `__getattr__`, star-imports and IDEs. Explicit imports (`from rpmdgle.utils import tcfs`, `from rpmdgle.utils.ou import gle_cxx`) behave as they did before. Within the package, `rfft` and `tcfs` import `arrays`, `nmtrans` imports `rfft`, and `ou` imports `special`. Nothing here imports from the rest of `rpmdgle`, except for a lazy `rpmdgle.units` import inside `fitaux/expohmic.main`. Keep it that way: `pes`, `propagators` and `sysbath` import from `utils`, so importing them back would create cycles.

| Module | Contents | Used by |
|---|---|---|
| `arrays` | `slice_along_axis`, `idx_along_axis`, `append_dims`, `string_to_slice`, `index_in_slice` | `pes/pi.py`, `propagators/verlet.py`, `langevin.py`, `properties.py`, `rfft`, `tcfs` |
| `rfft` | `RealFFT`, a wrapper for real FFTs along one axis | `nmtrans` |
| `nmtrans` | `FFTNormalModes`, `MatMulNormalModes` | `propagators/verlet.py` (FFT), `propagators/gle.py` (MatMul) |
| `tcfs` | `correlate`, `time_averaged_correlation` | tests, `notebooks/thermal_tcfs.ipynb` |
| `ou` | analytical GLE/OU covariances, friction kernel and spectrum of an `Ap` matrix | `propagators/gle.py` (`check_matrix`), `visualisation/auxplot.py`, `test_sepgleaux.py`, notebooks |
| `kernel_transforms` | friction kernels of non-centroid modes; DCT kernel ↔ spectrum helpers | `sysbath/spectral/splined.py` (`shift_lambda_to_kernel` with `n=0` only) |
| `calculus` | `GL_quad`, `FFTFourierIntegral` | `rates/rpmdrate.py` (`GL_quad`) |
| `kde` | kernel density estimate | `rates/analyse.py` |
| `special` | `sinc` and its derivatives, overflow-safe `logcosh` and `logsinh` | `ou.wnle_cqq` |
| `fitaux/` | interpolation and scaling of pre-fitted auxiliary-variable parameters; CLI for `expohmic` | standalone script |

## Normal modes (`nmtrans`, `rfft`)

- `forward_matrix[k, j]` for bead `j = 0..N-1`, in the mode order `0, -1, +1, -2, +2, ...`:
  - mode $0$: $1/N$
  - mode $-k$: $(\sqrt{2}/N)\cos(2\pi jk/N)$
  - mode $+k$: $(\sqrt{2}/N)\sin(2\pi jk/N)$
  - for even $N$, the last mode is $-N/2$, with entries $(-1)^j/N$

  `backward_matrix = N * forward_matrix.T`. The forward transform therefore gives the centroid as mode 0, and the backward transform inverts it exactly.
- `nm_freqs = 2 sin(π|k|/N)` is dimensionless. Multiply it by $\omega_N = N/(\beta\hbar)$, as `RingNM.parse_mscale` does. `mats_freqs = 2|k|` is not used anywhere.
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
- `tests/utils/test_tcfs.py` checks both functions against explicit loops. No other module in `utils` has its own tests.

## GLE and OU reference results (`ou`)

The notation follows Ceriotti et al. (doi:10.1021/ct900563s). `Ap` is the drift matrix of $(p, \mathbf{s})$ with shape `(1+naux, 1+naux)`. Below, $\theta$ = `Ap[0, 1:]` and $A$ = `Ap[1:, 1:]`.

- `get_friction_kernel(t, Ap)` $= \theta^\mathsf{T} e^{-|t|A}\theta$.
- `get_friction_spectrum(ω, Ap)` $= \theta^\mathsf{T} A (A^2+\omega^2)^{-1}\theta = \int_0^\infty K(t)\cos\omega t\,\mathrm{d}t$.

  These two functions use the same normalisation as `Density.K` and `Density.Lambda`, so the results compare directly with a bath, as `auxplot.py` does. Each takes a scalar, which returns a float, or a 1-D array. They read only `Ap[0, 1:]` and `Ap[1:, 1:]`, and so assume `Ap[0, 0] = 0` and `Ap[1:, 0] = -θ`. `build_Ap_kantorovich` and the test fixtures produce this structure. A Markovian `Ap[0, 0]` term, or a coupling that is not antisymmetric, is ignored without warning.
- **Two mass conventions for `Ap`.**
  - `SepGLEaux`, `parse_aux_spec`, `fitaux` and `auxplot.py`: $\theta^\mathsf{T} e^{-tA}\theta$ is the friction kernel *including* the mass, in the same units as `Density.K` (mass/time²).
  - `gle_cxx`: the dynamics run in mass-weighted coordinates (`Aqp[0,1] = -1`, `Aqp[1,0] = ω²`), so $\theta$ must be per unit mass. `test_sepgleaux.py::build_Ap_s1s2` divides $\theta$ by $\sqrt{m}$ before calling `gle_cxx`. Do the same whenever you compare a propagator's `Ap` with `gle_cxx` at a mass other than 1.
- `gle_cxx(t, omega, Ap, Cp=None, C0=None, u=0, beta=None, mass=1.0)` returns $\langle v(t)\,v(u)^\mathsf{T}\rangle$ for $v = (q, p, \mathbf{s})$ of a harmonic oscillator.
  - Pass exactly one of `beta` and `Cp`. `beta` selects the canonical stationary state. `Cp` gives a general stationary covariance of $(p, \mathbf{s})$, and β is then taken as `1/Cp[0,0]`.
  - `C0` sets a non-stationary covariance at time 0. A 1×1 or 2×2 `C0` sets only the $(q, p)$ block, and the rest is filled with $\mathbb{1}/\beta$.
  - `t` must be a 1-D (or 2-D) array. **A scalar `t` raises `RuntimeError`**, so pass `[t]`. Times must be non-negative.
  - With 1-D `t` and scalar `u`, the result has shape `(len(t), 2+naux, 2+naux)`. It is squeezed, so a length-1 `t` loses that axis.
  - $q$ and $p$ are returned in physical units (rescaled by `mass`). The auxiliary components stay mass-weighted.
- `get_Cqp` solves $A_{qp}C + CA_{qp}^\mathsf{T} = D_{qp}$ by eigendecomposition, in an O(n⁴) loop. That is fine for a few auxiliary variables.
- `wnle_cqq(t, beta, omega, tau, m)` gives $\langle q(t)q(0)\rangle$ under white-noise Langevin dynamics with friction $\gamma = 1/\tau$. `omega` and `tau` must be scalars. **Known bug:** the branch condition is `elif test < eps` instead of `< -eps`. As a result, the critical-damping branch is unreachable, and $\omega\tau = 1/2$ exactly returns NaN.

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
- `__all__` lists only two of the functions, so import functions by name.

## Smaller helpers

- `calculus.GL_quad(n, lo, hi)` returns Gauss–Legendre nodes and weights on `[lo, hi]`. `rpmdrate.py --ti` uses it and writes `GL_quad.csv`.
- `calculus.FFTFourierIntegral(lo, hi, npts, pad)` implements the endpoint-corrected FFT of Numerical Recipes §13.9 for $I(k) = \int_a^b e^{-2\pi i k x} f(x)\,\mathrm{d}x$.
  - `k` is in cycles per unit x, not angular.
  - The data must be `npts` equally spaced points that include both endpoints.
  - The results correspond to `self.freqs[axis]`: ascending, centred on 0, with `npts` values spaced by `1/(pad·npts·h)`.
  - The constructor's per-axis parameters are indexed by the axis numbers passed in `axes`.
  - Nothing in the package uses it and no test covers it. A manual check on $x^2$ over $[0, 1]$ agreed with `quad` to 1e-14.
- `kde.kde(data, grid, bw, kernel="gaussian", axis=-1)` averages `kernel(data, grid, bw)` over `axis`, so the caller has to broadcast the arguments, as in `kde(data[..., None], xgrid, bw, axis=0)`.
  - The accepted kernels are `gaussian`, `tophat`, `triangular` and `epanechnikov`. `analyse.py` uses `epanechnikov`.
  - `tricube_kernel` exists, but `kde()` raises `ValueError` for `"tricube"`.
- `special.sinc(x, nu=0)` is the *unnormalised* $\sin x / x$; `scipy.special.sinc` is $\sin \pi x/\pi x$. With `nu = 1` or `2` it returns the first or second derivative, and switches to a Taylor series for $|x| < 10^{-4}\pi$.
- `special.logcosh` and `special.logsinh` take real input. `logsinh` always returns a complex array, adding $i\pi$ for $x < 0$, so take `.real` for $x > 0$. `logsinh(0)` is `-inf`.
- `arrays.slice_along_axis` and `arrays.idx_along_axis` use basic indexing and return views. Writes through them reach the original array, which `pes/pi.py` relies on.
- `arrays.append_dims(arr, n)` appends `n` size-1 axes, so that per-mode or per-bead arrays broadcast against configurations.
- `arrays.string_to_slice` returns an `int`, a `list` or a `slice`, whatever its annotation says. `visualisation/lineplot.py` has its own copy.
- `arrays.index_in_slice` handles only non-negative `start` and `stop`. For example, `slice(None, -1)` contains nothing.

## `fitaux`: pre-fitted auxiliary-variable parameters

This package does **not** fit anything. It takes Kantorovich parameters (`tauD`, `cD`, `tauO`, `cO`, `omegaO`) that were fitted elsewhere, as functions of the reduced normal-mode frequency $\omega_n/\omega_c$, interpolates them with splines, and scales them to physical units for an `expohmic` bath. The data files are not in the repository.

- **CSV layout.** The first header row holds the parameter names (`nmfreq`, `tauD`, `cD`, ...). The second header row holds the auxiliary-variable index (0, 1, ...). Each following row is one reduced frequency. `MultiAuxParams` reads both header rows as a `MultiIndex`. `BaseOneAuxParam` reads the first row as the header and drops the second (`.iloc[1:]`), so both classes expect the same layout.
- `MultiAuxParams` fits one weighted `UnivariateSpline` per auxiliary variable. It raises `RuntimeError` for frequencies above the tabulated range. With `sort="ascending"` or `"descending"`, it sorts the values across auxiliary variables above `sorting_cutoff`, which keeps the spline branches from crossing.
- `BaseOneAuxParam` handles a single auxiliary variable, and subclasses implement `extrapolate(w)` for frequencies beyond the data. In reduced units, `expohmic.OneAuxParam` uses `omegaO → w`, `tauO → 0.4415·w` and `cO → √K_n(0)`, where `zero_time_kernel` gives K_n(0) analytically.
- `get_nm_freqs(beta, hbar, N, n=None)` returns the physical frequencies $\frac{2N}{\beta\hbar}\sin(\pi|n|/N)$, by default in the order `0, -1, +1, ...`. It tests `if n:`, so `n=0` returns *all* modes. Pass `[0]` to get the centroid alone.
- `expohmic.py` is a CLI that is not linked in `bin/`:

  ```bash
  python -m rpmdgle.utils.fitaux.expohmic fits.csv atomic --wc 200 --wb 1000 --mass "1 amu" -T 300 -N 32 [--eta0 1] [-n 0 -1 1] [--oneaux]
  ```

  - `--wc` and `--wb` are in cm⁻¹ and `-T` is in kelvin.
  - The static friction is `eta0·m·ω_b`, which corresponds to the `eta` of `expohmic`.
  - Times are divided by ω_c, frequencies are multiplied by ω_c, and couplings are multiplied by $\sqrt{\eta\,\omega_c}$. The couplings therefore include the mass, following the propagator convention described under `ou`.
  - The script prints comma-separated JSON objects, one per normal mode, with no enclosing brackets. Run it without `-n` to get all `N` modes in the order that `SepGLEaux` expects, then wrap the output in `[...]` to form the `aux` list.
  - Columns missing from the CSV are skipped silently (by a bare `except`). This can produce empty `{}` entries.

## Testing

Only `tcfs` has direct tests. `test_sepgleaux.py` uses `ou.gle_cxx` as a reference, so that function is also exercised. Put new tests in `tests/utils/test_<module>.py` and keep the file name unique across `tests/` (see `tests/CLAUDE.md`). Good candidates are:

- that `FFTNormalModes` and `MatMulNormalModes` agree (move the `nmtrans` `__main__` block into a test);
- that `get_friction_spectrum` matches a `quad` cosine transform of `get_friction_kernel`;
- the `kernel_transforms` DCT round trips on $K = e^{-t}$, $\Lambda = 1/(1+\omega^2)$.
