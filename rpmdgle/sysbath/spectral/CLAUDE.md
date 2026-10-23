# CLAUDE.md — `rpmdgle/sysbath/spectral`

This package holds the spectral densities that define the harmonic bath. Each submodule exposes one class, `Density`, which is the continuous spectral density alone: its parameters, `J`, `Lambda`, `K`, `exact_reorganisation` and the discretisation rule `quadpoints(N)`. It knows nothing about the system PES, the number of bath modes or the coupling. The bath JSON selects a submodule by its `"name"` key.

The Caldeira–Leggett potential is `rpmdgle/sysbath/bath.py::CaldeiraLeggett(PES, terms, coupling)`, a `BasePES`. `terms` is a list of `(Density, Nmodes)` pairs, and the bath's spectral density is their sum. A single density is a bath with one term. This file documents both modules.

| Module | `J(ω)` | `K(t)` | 
|---|---|---|
| `expohmic` | $\eta \omega \exp(-\omega/\omega_c)$ | $ \dfrac{2 \eta \omega_c / \pi}{(1 + \omega_c^2 t^2)}$ | 
| `debye` | $\eta \omega \dfrac{\omega_c^2}{\omega^2+\omega_c^2}$ | $\eta\omega_ce^{-\omega_c t}$ | 
| `brownian` | $\eta \omega \dfrac{\omega_0^4}{(\omega^2-\omega_0^2)^2+\gamma^2\omega^2}$ | $\dfrac{\eta\omega_0^2}{\gamma}e^{-\gamma t/2}\left[\cos\bar\omega t + \dfrac{\gamma}{2\bar\omega}\sin\bar\omega t\right]$, $\bar\omega = \sqrt{\omega_0^2-\gamma^2/4}$ | 
| `splined` | $\eta \omega\cdot{}$ `spline(ω)` | numerical cosine transform | 

For `brownian` with $\gamma > 2\omega_0$, $\cos$ and $\sin$ become $\cosh$ and $\sinh$ of $\kappa t$, where $\kappa = \sqrt{\gamma^2/4-\omega_0^2}$. At $\gamma = 2\omega_0$ the bracket is $1 + \gamma t/2$. The code chooses the regime from the sign of $\omega_0^2 - \gamma^2/4$, and each branch stays accurate close to critical damping. The overdamped branch is evaluated as a sum of decaying exponentials, so it does not overflow at long times.

## Conventions

These definitions hold for every density in the package, and `bath.py` relies on them. $N$ is the total number of bath modes, `CaldeiraLeggett.Nmodes`.

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
- For a bath with terms $k = 1, 2, \dots$, $J = \sum_k J_k$, and likewise $\Lambda$, $K$, $\lambda$ and $\eta$ are sums over the terms. `CaldeiraLeggett.J`, `Lambda`, `K` and `exact_reorganisation` return these sums.
- $F(q)$ is the coupling object (`sysbath/coupling`, linear by default). `CaldeiraLeggett` stores it as `self.coupling`, and every bath mode couples through it: all terms share one coupling.

## Discretisation (`bath.CaldeiraLeggett.bath_params`)

Each term is discretised on its own, as if the system were coupled to independent baths, and the blocks of modes are concatenated in the order of `terms`. Below, $N_k$, $\Lambda_k$ and $\lambda_k$ refer to term $k$.

- Every bath mass equals the system mass: $m_\nu = m$, where $m$ is `PES.mass`.
- The frequencies of term $k$ are `quadpoints(N_k)`. They satisfy the following condition, from doi:10.1002/jcc.24527:

  $$
  \int_0^{\omega_\nu} \Lambda_k(\omega) \, \mathrm{d}\omega
  = \frac{\nu - \frac{1}{2}}{N_k} \int_0^\infty \Lambda_k(\omega) \, \mathrm{d}\omega
  = \frac{\pi \lambda_k}{4} \, \frac{\nu - \frac{1}{2}}{N_k},
  \qquad \nu = 1, \dots, N_k
  $$

  Each mode of term $k$ therefore carries an equal share of $\Lambda_k$, and the couplings are

  $$
  c_\nu = \sqrt{\frac{\lambda_k}{2N_k}} \, \sqrt{m} \, \omega_\nu
  \quad \Longleftrightarrow \quad
  \frac{c_\nu^2}{m_\nu \omega_\nu^2} = \frac{\lambda_k}{2N_k}
  $$

  Modes of different terms carry different weights unless $\lambda_k / N_k$ is the same for all terms.
- `quadpoints(N)` must return $N$ frequencies in ascending order. Each block of `w` is therefore ascending, but the whole array is not sorted. `term_slices[k]` is the slice of the bath-mode arrays that holds term $k$.
- `quadrature(f)` and `l_quadrature(f)` take `f` as an array of the values $f(\omega_\nu)$ and use the discrete modes to approximate

  $$
  \begin{aligned}
  \int_0^\infty J(\omega) f(\omega) \, \mathrm{d}\omega & \approx \frac{\pi}{2} \sum_{\nu=1}^{N} \frac{c_\nu^2}{m_\nu \omega_\nu} f(\omega_\nu), \\
  \int_0^\infty \Lambda(\omega) f(\omega) \, \mathrm{d}\omega & \approx \frac{\pi}{2} \sum_{\nu=1}^{N} \frac{c_\nu^2}{m_\nu \omega_\nu^2} f(\omega_\nu)
  = \sum_k \frac{\pi \lambda_k}{4N_k} \sum_{\nu \in k} f(\omega_\nu)
  \end{aligned}
  $$

- `reorganisation_energy()` evaluates $\frac{4}{\pi} \int_0^\infty \Lambda \, \mathrm{d}\omega$ with this quadrature, so it equals `exact_reorganisation()` by construction. That makes it a consistency check, not a test of convergence in $N$.

## What other code reads from a `CaldeiraLeggett`

- Arrays: `c`, `w`, `bath_mass`, `ww = w²` and `mww = m w²`, each of shape `(Nmodes,)`. `mass` has shape `(1+Nmodes,)`, with the system first.
- Scalars and objects: `Nmodes`, `PES`, `UNITS` (the PES's unit object), `coupling`, `densities` (the `Density` of each term) and `term_slices`.
- `propagators/gle.py::build_pmf` builds the potential of mean field from `c`, `w` and `bath_mass`, and evaluates `F` with `coupling.both`. Even the implicit-bath propagators (`SepGLE*`) construct a discrete bath, so `Nmodes` sets the accuracy of the PMF. The PMF sums over all modes, so it is the sum of the PMFs of the terms.
- `rates/qtst.py::sample_bath_modes` uses `ww`, `mass[..., 1:]` and `bath_eq(q)`, which returns the equilibrium bath positions `c F(q)/(m ω²)`.
- `visualisation/bathplot.py` and `auxplot.py` call `Lambda`, `K` and `w`. `bathplot.py` takes the default plot range from `wplot`, the largest `wplot` of the terms, and with `--terms` it also plots each term from `densities` and colours the quadrature lines by `term_slices`.
- `potential`, `gradient` and `both` are analytical. `hessian` falls back to `BasePES` finite differences.
- In the configuration's last axis, index 0 is the system coordinate `q` and indices `1..N` are the bath modes `r`. The code splits off `ndof − Nmodes` system coordinates, but the coupling reduces the last axis and `mass` is built from `np.atleast_1d(PES.mass)`. In practice only one-dimensional systems work. The couplings return a Python float for an unbatched configuration of shape `(1+Nmodes,)`, which `potential` cannot index, so always pass at least one leading replica axis.

## Construction and JSON input

`system.get_bath(PES, bath_data, F_data)` builds the coupling with `get_bath_coupling`, turns `bath_data` into `(Density, Nmodes)` pairs with `get_bath_terms`, and returns `CaldeiraLeggett(PES, terms, coupling=F)`. Neither function modifies its input dictionaries. A single density is a flat object:

```json
{"name": "expohmic", "Nmodes": 64, "eta": 1.0, "omega_cut": 0.002, "UNITS": "atomic"}
{"name": "brownian", "Nmodes": 64, "eta": 1.0, "omega0": 0.001, "gamma": 0.0005}
{"name": "splined", "Nmodes": 64, "eta": 1.0, "eps": 1e-10, "x": [0.0, ...], "y": [1.0, ...]}
```

A sum of densities lists them under `terms`:

```json
{"UNITS": "atomic",
 "terms": [{"name": "expohmic", "Nmodes": 48, "eta": 1.0, "omega_cut": 0.002},
           {"name": "brownian", "Nmodes": 16, "eta": 0.2, "omega0": 0.001, "gamma": 0.0005}]}
```

- In each term, `name` and `Nmodes` (a positive integer) are required, and the remaining keys go to that module's `Density` constructor. The constructors take no `**kwargs`, so a misspelled or missing parameter raises `TypeError`.
- The only keys allowed beside `terms` are `UNITS`. A bath-level `Nmodes` raises `ValueError`: give `Nmodes` in each term. Other unknown keys, an empty `terms` list, nested `terms` and an unknown `name` also raise `ValueError`.
- `coupling` is reserved, in a term and beside `terms`, for couplings specific to a term. Both raise `NotImplementedError`. For now the coupling comes only from `--coupling` and is shared by all terms.
- An error raised while building term `i` carries the note `Raised while constructing bath term i ('name')`.
- `UNITS` is optional at both levels. It defaults to the PES's unit system, and if given it must match, or `RuntimeError` is raised. It is not passed to `Density`. The default coupling built by `get_bath_coupling` also takes its units from the PES, but a coupling specified in a JSON file defaults to `"atomic"` unless the file sets `"UNITS"`.
- `brownian` takes the static friction `eta`, the undamped frequency `omega0` of the oscillator mode and its damping rate `gamma`. All three must be positive, or the constructor raises `ValueError`. The reorganisation energy is $\lambda = 2\eta\omega_0^2/\gamma$.
- Parameter values must be bare numbers in the PES's internal units. Nothing here calls `str2base`, so strings such as `"200 cm^-1"` fail.
- `splined` expects the frequency grid `x` and the shape of Λ, `y`; both are converted with `np.asarray`, so JSON lists work. The optional `axis`, `bc_type` and `extrapolate` keys go to `scipy.interpolate.CubicSpline`. Start `x` at `ω = 0`: `K`, `λ` and `quadpoints` all integrate from 0, so a grid that starts higher leaves Λ extrapolated near ω = 0. `K(t)` calls `quad` once per time point, so it is slow on long time arrays.

In Python, build the bath directly:

```python
from rpmdgle.sysbath.bath import CaldeiraLeggett
from rpmdgle.sysbath.spectral import brownian, expohmic

SB = CaldeiraLeggett(PES, [(expohmic.Density(eta=1.0, omega_cut=0.002), 48),
                           (brownian.Density(eta=0.2, omega0=0.001, gamma=0.0005), 16)])
```

## Adding a spectral density

1. Create `<name>.py` containing `class Density(BaseSpectralDensity)`. Its constructor takes only the parameters of the density, as explicit arguments.
2. Implement `J`, `Lambda`, `K`, `quadpoints(Nmodes)` and `exact_reorganisation`. `J`, `Lambda` and `K` must accept NumPy arrays and also scalars, because `scipy.integrate.quad` passes scalars.
3. If there is no closed form for the frequencies, subclass `BaseNumericalSpectralDensity(wmax, eps)` instead. It finds the frequencies by bisecting `quad(Lambda)` and sets `λ = 2K(0)` in its constructor, so store the parameters on `self` before calling `super().__init__`. No existing density uses this class yet. If the cumulative integral of Λ has a closed form, follow `brownian` instead. It subclasses `BaseSpectralDensity`, implements the normalised CDF `cdf(ω)` analytically, and finds the frequencies by solving `cdf(ω) = (j+½)/N` with `brentq`, without calling `quad`.
4. Add the module to `from . import ...` in `__init__.py`. `get_bath_terms` accepts only the public submodules imported there.
5. Set `wplot`, the upper end of the default plot range in `bathplot.py`. `expohmic` and `debye` use $5\omega_c$. In `brownian` it is the frequency below which 95% of $\int_0^\infty \Lambda \, \mathrm{d}\omega$ lies. `splined` and `BaseNumericalSpectralDensity` use `wmax`, which in those classes is also the integration cutoff.

`tests/sysbath/spectral/test_brownian.py` tests `brownian` against `quad` in all three damping regimes and close to critical damping. `tests/sysbath/test_bath.py` checks every density's discretisation against values recorded before the densities were separated from the bath, and checks the identities for a sum of terms. `tests/test_system.py` tests the JSON parser. No other density has its own test; `tests/propagators/gle/ou_gle/test_sepgleaux.py` only exercises `expohmic`. To check a new density, compare `K(t)` with `(2/π) quad(Lambda, 0, ∞, weight='cos', wvar=t)`, and check that `∫₀^{w[j]} Λ / ∫₀^∞ Λ · N` equals `j + ½` for zero-based `j`. `test_brownian.py` can serve as a template.
