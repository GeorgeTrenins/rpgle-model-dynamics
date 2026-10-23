# CLAUDE.md — `tests/sysbath/spectral`

Tests for `rpmdgle/sysbath/spectral`. So far only `brownian` has tests, in `test_brownian.py`. They are deterministic, take under a second and write no files. Their reference values come from `scipy.integrate.quad` and from closed-form results such as Λ(0) = η. `test_discretisation` and `test_get_bath` build a one-term `CaldeiraLeggett` bath. The discretisation of every density, and baths that sum several densities, are tested one level up in `tests/sysbath/test_bath.py`.

## Running

```bash
pytest tests/sysbath/spectral -W error::RuntimeWarning
```

- `-W error::RuntimeWarning` turns NumPy overflow and invalid-value warnings into failures. The tests check that `K(t)` stays finite at long times and near critical damping, so use this flag after changing `brownian.py`. The tests pass with it.
- The `density` fixture, a `brownian.Density`, runs each test at five damping rates, with the IDs `underdamped`, `critical`, `overdamped`, `near-critical-under` and `near-critical-over`. Select cases by node ID or with `-k`:

  ```bash
  pytest "tests/sysbath/spectral/test_brownian.py::test_cdf[near-critical-over]"
  pytest tests/sysbath/spectral -k "critical and not near"    # exact critical damping only
  ```

  `-k critical` on its own also selects the near-critical cases and `test_near_critical_continuity`.

## Conventions in `test_brownian.py`

- Tight `pytest.approx` checks pass `abs=0`. Without it, the tolerance is `max(rel·|expected|, 1e-12)`, so a smaller `rel` is silently loosened. `np.testing.assert_allclose` defaults to `atol=0` and needs no such care.
- The tests read the private attribute `Density._regime`, which is `"underdamped"`, `"critical"` or `"overdamped"`. Update them if it is renamed.
- To test another density, add `test_<module>.py` here. `rpmdgle/sysbath/spectral/CLAUDE.md` lists the checks a new density should pass, and `test_brownian.py` can serve as a template.
