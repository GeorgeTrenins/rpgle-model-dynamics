# CLAUDE.md — `rpmdgle/sysbath/spectral`

This package holds the spectral densities that define the harmonic bath. Each submodule exposes one class, `Density`, which is a `BasePES` for the full Caldeira–Leggett potential: the system PES plus `Nmodes` discretised bath modes. The bath JSON selects a submodule by its `"name"` key.

| Module | `J(ω)` | `K(t)` | 
|---|---|---|
| `expohmic` | $\eta \omega \exp(-\omega/\omega_c)$ | $ \dfrac{2 \eta \omega_c / \pi}{(1 + \omega_c^2 t^2)}$ | 
| `debye` | $\eta \omega \dfrac{\omega_c^2}{\omega^2+\omega_c^2}$ | $\eta\omega_ce^{-\omega_c t}$ | 
| `brownian` | $\eta \omega \dfrac{\omega_0^4}{(\omega^2-\omega_0^2)^2+\gamma^2\omega^2}$ | $\dfrac{\eta\omega_0^2}{\gamma}e^{-\gamma t/2}\left[\cos\bar\omega t + \dfrac{\gamma}{2\bar\omega}\sin\bar\omega t\right]$, $\bar\omega = \sqrt{\omega_0^2-\gamma^2/4}$ | 
| `splined` | $\eta \omega\cdot{}$ `spline(ω)` | numerical cosine transform | 

For `brownian` with $\gamma > 2\omega_0$, $\cos$ and $\sin$ become $\cosh$ and $\sinh$ of $\kappa t$, where $\kappa = \sqrt{\gamma^2/4-\omega_0^2}$. At $\gamma = 2\omega_0$ the bracket is $1 + \gamma t/2$. The code chooses the regime from the sign of $\omega_0^2 - \gamma^2/4$, and each branch stays accurate close to critical damping. The overdamped branch is evaluated as a sum of decaying exponentials, so it does not overflow at long times.

## Conventions

These definitions hold for every density in the package, and `_base.py` relies on them. $N$ is `Nmodes`.

$$
V(q, \mathbf{r}) = V_\mathrm{sys}(q) + \sum_{\nu=1}^{N} \frac{1}{2} m_\nu \omega_\nu^2 \left( r_\nu - \frac{c_\nu F(q)}{m_\nu \omega_\nu^2} \right)^2
$$

$$
J(\omega) = \frac{\pi}{2} \sum_{\nu=1}^{N} \frac{c_\nu^2}{m_\nu \omega_\nu} \, \delta(\omega - \omega_\nu),
\qquad
\Lambda(\omega) = \frac{J(\omega)}{\omega}
$$

$$
K(t) = \frac{2}{\pi} \int_0^\infty \Lambda(\omega) \cos(\omega t) \, \mathrm{d}\omega,
\qquad
\lambda = \frac{4}{\pi} \int_0^\infty \Lambda(\omega) \, \mathrm{d}\omega = 2 K(0)
$$

- $\eta = \Lambda(0) = \int_0^\infty K(t) \, \mathrm{d}t$ is the static friction. $J$, $\Lambda$, $K$ and $\eta$ all include the system mass, so $\eta$ has units of mass/time. Divide by `PES.mass` to get a rate, as `visualisation/bathplot.py` and `auxplot.py` do. For `splined`, the static friction is $\eta \, y_0$, where $y_0$ is `y[0]`; normalise `y` so that $y_0 = 1$ if $\eta$ should keep this meaning.
- $F(q)$ is the coupling object (`sysbath/coupling`, linear by default). Every `Density` stores it as `self.coupling`.

## Discretisation (`_base.BaseSpectralDensity.bath_params`)

- Every bath mass equals the system mass: $m_\nu = m$, where $m$ is `PES.mass`.
- The frequencies $\omega_\nu$ satisfy the following condition, from doi:10.1002/jcc.24527:

  $$
  \int_0^{\omega_\nu} \Lambda(\omega) \, \mathrm{d}\omega
  = \frac{\nu - \frac{1}{2}}{N} \int_0^\infty \Lambda(\omega) \, \mathrm{d}\omega
  = \frac{\pi \lambda}{4} \, \frac{\nu - \frac{1}{2}}{N},
  \qquad \nu = 1, \dots, N
  $$

  Each mode therefore carries an equal share of $\Lambda$, and the couplings are

  $$
  c_\nu = \sqrt{\frac{\lambda}{2N}} \, \sqrt{m} \, \omega_\nu
  \quad \Longleftrightarrow \quad
  \frac{c_\nu^2}{m_\nu \omega_\nu^2} = \frac{\lambda}{2N}
  $$

  `quadpoints()` must return the $N$ frequencies in ascending order.
- `quadrature(f)` and `l_quadrature(f)` take `f` as an array of the values $f(\omega_\nu)$ and use the discrete modes to approximate

  $$
  \begin{aligned}
  \int_0^\infty J(\omega) f(\omega) \, \mathrm{d}\omega & \approx \frac{\pi}{2} \sum_{\nu=1}^{N} \frac{c_\nu^2}{m_\nu \omega_\nu} f(\omega_\nu), \\
  \int_0^\infty \Lambda(\omega) f(\omega) \, \mathrm{d}\omega & \approx \frac{\pi \lambda}{4N} \sum_{\nu=1}^{N} f(\omega_\nu)
  \end{aligned}
  $$

- `reorganisation_energy()` evaluates $\frac{4}{\pi} \int_0^\infty \Lambda \, \mathrm{d}\omega$ with this quadrature, so it equals `exact_reorganisation()` by construction. That makes it a consistency check, not a test of convergence in $N$.

## What other code reads from a `Density`

- Arrays: `c`, `w`, `bath_mass`, `ww = w²` and `mww = m w²`, each of shape `(Nmodes,)`. `mass` has shape `(1+Nmodes,)`, with the system first.
- Scalars and objects: `Nmodes`, `PES`, `UNITS`, `coupling`.
- `propagators/gle.py::build_pmf` builds the potential of mean field from `c`, `w` and `bath_mass`, and evaluates `F` with `coupling.both`. Even the implicit-bath propagators (`SepGLE*`) construct a discrete bath, so `Nmodes` sets the accuracy of the PMF.
- `rates/qtst.py::sample_bath_modes` uses `ww`, `mass[..., 1:]` and `bath_eq(q)`, which returns the equilibrium bath positions `c F(q)/(m ω²)`.
- `visualisation/bathplot.py` and `auxplot.py` call `Lambda`, `K` and `w`. They pick the default plot range from a `wmax` or `omega_cut` attribute.
- `potential`, `gradient` and `both` are analytical. `hessian` falls back to `BasePES` finite differences.
- In the configuration's last axis, index 0 is the system coordinate `q` and indices `1..N` are the bath modes `r`. The code splits off `ndof − Nmodes` system coordinates, but the coupling reduces the last axis and `mass` is built from `np.atleast_1d(PES.mass)`. In practice only one-dimensional systems work.

## Construction and JSON input

`system.get_bath` pops `name` and `Nmodes` and passes the remaining keys as kwargs, together with `coupling=F`:

```json
{"name": "expohmic", "Nmodes": 64, "eta": 1.0, "omega_cut": 0.002, "UNITS": "atomic"}
{"name": "brownian", "Nmodes": 64, "eta": 1.0, "omega0": 0.001, "gamma": 0.0005}
{"name": "splined", "Nmodes": 64, "eta": 1.0, "eps": 1e-10, "x": [0.0, ...], "y": [1.0, ...]}
```

- `brownian` takes the static friction `eta`, the undamped frequency `omega0` of the oscillator mode and its damping rate `gamma`. All three must be positive, or the constructor raises `ValueError`. The reorganisation energy is $\lambda = 2\eta\omega_0^2/\gamma$.

- Parameter values must be bare numbers in the PES's internal units. Nothing here calls `str2base`, so strings such as `"200 cm^-1"` fail.
- `UNITS` defaults to `"atomic"` and is **not** taken from the PES. If the PES uses a different unit system, set `"UNITS"` in the bath JSON to the same class name, or the constructor raises `RuntimeError`. The default coupling built by `get_bath_coupling` does take its units from the PES.
- `splined` expects the frequency grid `x` and the shape of Λ, `y`; both are converted with `np.asarray`, so JSON lists work. The optional `axis`, `bc_type` and `extrapolate` keys go to `scipy.interpolate.CubicSpline`. Start `x` at `ω = 0`: `K`, `λ` and `quadpoints` all integrate from 0, so a grid that starts higher leaves Λ extrapolated near ω = 0. `K(t)` calls `quad` once per time point, so it is slow on long time arrays.

## Adding a spectral density

1. Create `<name>.py` containing `class Density(BaseSpectralDensity)`.
2. Store all parameters on `self` **before** calling `super().__init__(PES, Nmodes, *args, **kwargs)`. The base constructor calls `quadpoints()` and `exact_reorganisation()` immediately.
3. Implement `J`, `Lambda`, `K`, `quadpoints` and `exact_reorganisation`. `J`, `Lambda` and `K` must accept NumPy arrays and also scalars, because `scipy.integrate.quad` passes scalars.
4. If there is no closed form for the frequencies, subclass `BaseNumericalSpectralDensity(PES, Nmodes, wmax, eps, ...)` instead. It finds the frequencies by bisecting `quad(Lambda)` and sets `λ = 2K(0)`, so `K` must already work before `super().__init__`. No existing density uses this class yet. If the cumulative integral of Λ has a closed form, follow `brownian` instead. It subclasses `BaseSpectralDensity`, implements the normalised CDF `cdf(ω)` analytically, and finds the frequencies by solving `cdf(ω) = (j+½)/N` with `brentq`, without calling `quad`.
5. Add the module to `from . import ...` in `__init__.py`. `get_bath` looks densities up with `getattr(spectral, name)`.
6. Expose `omega_cut` or `wmax` so that `bathplot.py` picks a sensible plot range. In `brownian`, `wmax` is only a plotting hint: the frequency below which 95% of $\int_0^\infty \Lambda \, \mathrm{d}\omega$ lies. It is not an integration cutoff, unlike in `splined`.

`tests/sysbath/spectral/test_brownian.py` tests `brownian` against `quad` in all three damping regimes and close to critical damping. No other density has its own test; `tests/propagators/gle/ou_gle/test_sepgleaux.py` only exercises `expohmic`. To check a new density, compare `K(t)` with `(2/π) quad(Lambda, 0, ∞, weight='cos', wvar=t)`, and check that `∫₀^{w[j]} Λ / ∫₀^∞ Λ · N` equals `j + ½` for zero-based `j`. `test_brownian.py` can serve as a template.
