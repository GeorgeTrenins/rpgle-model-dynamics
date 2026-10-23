# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`rpmdgle` is a NumPy implementation of dissipative ring-polymer molecular dynamics (RPMD) for low-dimensional model systems coupled to harmonic (Caldeira–Leggett) baths. The central method is the RPMD generalized Langevin equation (RPMD-GLE) of https://doi.org/10.1103/PhysRevLett.134.226201; docstrings in `rpmdgle/propagators/gle.py` cite equations from that paper (main text and SI, e.g. "Eq. (S38)") by number.

## Environment and commands

There is no packaging (no `setup.py`/`pyproject.toml`). The package is put on the path by sourcing `env.sh`, which prepends the repo root to `PYTHONPATH` and `bin/` to `PATH`. The VS Code workspace uses the `tools` conda env (Python 3.11; the code needs ≥3.11, e.g. for `Exception.add_note`). Dependencies are in `requirements.txt`.

```bash
mamba activate tools && source env.sh

pytest tests                                                      # all tests (~10 s)
pytest tests/utils/test_tcfs.py::test_correlate_equal_size        # single test
```

`tests/propagators/gle/ou_gle/test_sepgleaux.py` writes `.eps` debug plots to the current working directory. Those files are gitignored only inside that test directory, so run it from there, or delete the plots afterwards. It is a statistical test: it propagates 400 replicas of a harmonic oscillator under `SepGLEaux` and compares the sampled correlation functions with the analytical ones from `rpmdgle.utils.ou.gle_cxx`, with 1% tolerance.

## Running simulations

The CLI drivers live in `rpmdgle/rates/`. Files in `bin/` are symlinks to the executable drivers and plotting scripts in `rpmdgle/rates/` and `rpmdgle/visualisation/`; edit the targets, not the symlinks.

- `rates/qtst.py`: thermal sampling and production trajectories. It also provides the setup helpers (`make_SB`, `make_RP`, `make_propa`, `equilibrate`, `production`) that the other drivers reuse.
- `rates/fluxcorr.py`: computes the RPMD flux-side correlation function and transmission coefficient.
- `rates/rpmdrate.py`: complete Bennett–Chandler workflow, with stages enabled by `--hist`, `--ti` and `--fs`.
- `rates/analyse.py`: post-processes the output of `rpmdrate.py` (histograms, thermodynamic integration, rate).

All parsers use `MyArgumentParser`, so `+args.txt` reads arguments from a file with one `--flag=value` per line.

### JSON input files

All inputs are JSON. The README examples contain `//` comments for explanation only; real input files must be plain JSON.

- **PES** (`-P`): `{"module": "rpmdgle.pes.polynomial", "name": "PES", ...constructor kwargs}`. `system.get_PES` imports the module dynamically.
- **Bath** (`--bath`): `{"name": "expohmic", "Nmodes": 64, ...}` for one spectral density, or `{"terms": [{"name": ..., "Nmodes": ..., ...}, ...]}` for a sum of them. `name` is a submodule of `rpmdgle.sysbath.spectral`, and the class used is always `Density`. Each term is discretised with its own `Nmodes`. `rpmdgle/sysbath/spectral/CLAUDE.md` gives the full rules.
- **Coupling** (`--coupling`): `{"name": "linear" | "exptanh" | "splined", ...}`. The class used is always `Coupling` in `rpmdgle.sysbath.coupling.<name>`. It defaults to linear when the file is omitted.
- **Propagator** (`--propa` etc.): `{"class": "<name exported from rpmdgle.propagators>", "dt": "0.25 fs", ...kwargs}`. The README documents each class's options, including the `aux` drift-matrix spec for `SepGLEaux`.
- **Properties** (`--properties`): a list of `{"name": <output file stem>, "A": <property id>, "stride": "<time>", "postA": "<python lambda string>"}`. `postA` is `eval`'d. Valid ids are listed in `rpmdgle/properties.py`.

## Architecture

### Units

Every PES carries a `UNITS` object, an instance of a class in `rpmdgle/units.py` (`atomic` by default; also `hartAng`, `kcalAfs`, `eVAamu`, and others). The class is built on SymPy units. Time and other dimensional inputs may be strings such as `"100 fs"`, which `UNITS.str2base` converts to internal units. Bare numbers are taken to be in internal units already. Couplings take `UNITS` as a class-name string, and the bath raises if it does not match the PES. A `UNITS` key in the bath JSON is optional and defaults to the PES's unit system.

### Array layout

Everything is vectorised over independent replicas. Configuration arrays have shape `replica_layout + (nbeads,) + xshape`, where `xshape` is what the physical PES expects for one bead (always `(1,)` or `(1+Nmodes,)` here). The bead axis is therefore `ndim - len(rpPES.rpxshape)`, and `nmtrans.axis` stores it. When an explicit bath is present, index 0 of the last axis is the system coordinate and the rest are bath modes.

### Normal modes

`utils/nmtrans.py` uses the Matsubara-paper normalisation (doi:10.1063/1.4916311) and orders the modes `0, -1, +1, -2, +2, ...`. Mode 0 is the centroid `xnm[..., 0, :]`, which equals the bead average because the forward transform divides by N. The same ordering governs the per-mode `aux` list for `SepGLEaux`, which must have exactly `nbeads` entries. `FFTNormalModes` (used by propagators) and `MatMulNormalModes` (used where explicit matrices are needed) are interchangeable.

### PES layers

- `pes/_base.BasePES` is the interface: `potential`, `gradient`, `hessian`, `force`, `both`, `all`. Gradients and Hessians fall back to finite differences.
- `pes/pi.Ring` wraps a physical PES into a ring polymer. `BasePI.__getattr__` synthesises `potential`, `gradient` and the other methods as `polymer_*` + `external_*`. Propagators often call `external_both` and `polymer_potential` separately.
- `sysbath/bath.CaldeiraLeggett` is a `BasePES`. It wraps the system PES plus `Nmodes` discretised harmonic bath modes into the full Caldeira–Leggett potential, and also exposes the bath parameters `c`, `w`, `bath_mass` and the coupling `F(q)`. Its spectral density is a sum of one or more terms, each a `sysbath/spectral/*.Density` (the continuous J(ω) alone) discretised into its own block of modes. All terms share the coupling.

### Two ways to represent dissipation

`qtst.make_propa` dispatches on the propagator class:

1. **Explicit bath**: `VelocityVerlet`, `RingNM`, `RingPILE`, `Langevin` and `SVR` propagate the full `CaldeiraLeggett` potential ring-polymerised with `xshape = (1+Nmodes,)`.
2. **Implicit bath**: `SepGLEPILE` and `SepGLEaux` propagate only the system ring polymer, `xshape = (1,)`. They take the `CaldeiraLeggett` object separately to build the potential of mean field (PMF) from coupling to the bath (`build_pmf`). `SepGLEPILE` adds only the PMF and thermostats with fictitious PILE friction, so it is for sampling. `SepGLEaux` adds the true memory friction and noise through auxiliary variables (`saux`), with one `Ap` drift matrix per normal mode, following Ceriotti's GLE conventions (doi:10.1021/ct900563s). `Ap` can come from a Kantorovich parametrisation, an inline matrix, or an external JSON file; see `parse_aux_spec`.

### Propagator conventions (`propagators/`)

- `x`, `p`, `f` and their normal-mode counterparts `xnm`, `pnm`, `fnm` are preallocated. Always update them in place (`arr[:] = ...`) and never rebind them.
- `set_x`/`set_p` take bead coordinates; `set_xnm`/`set_pnm` take normal-mode coordinates. Each call converts the input and refreshes forces. `to_mode(attr)` and `to_bead(attr)` sync one representation from the other.
- `step(s)` runs `step0` then `step1` (splitting such as OBABO), and `force_update()` recomputes `V` and `f`.
- `fixed` is a list of index tuples that are held constant, for example the system centroid during thermodynamic integration. It is enforced through `fix_positions` and `fix_momenta`, which also re-sync the bead arrays.
- In the `SepGLE*` propagators, `ethermo` accumulates each step's change in total energy (plus the auxiliary-variable kinetic energy for `SepGLEaux`) minus the energy injected by the O steps. It therefore measures integration error and should stay near zero.
- RNG seeding goes through `np.random.default_rng`. Drivers split one seed with `np.random.SeedSequence(seed).spawn(n)` so that each propagator gets an independent stream.

### Other utilities

- `utils/ou.py`: analytical Ornstein–Uhlenbeck/GLE correlation functions, friction kernels and spectra. Tests and notebooks use these as references.
- `utils/tcfs.py`: FFT-based time correlation functions.
- `utils/fitaux/`: fitting auxiliary-variable `Ap` matrices to a target friction kernel.
- `notebooks/` contains worked examples that use the Python API directly instead of the CLI.

## Code style

Source files start with the header docstring used throughout the repo (`@File`, `@Time`, `@Author`, `@Desc`), followed by `from __future__ import print_function, division, absolute_import`. Docstrings are Google style, with shapes and the physical meaning of each array spelled out.

## Editing rules

1. The AI assistant must never make commits to the git repository themselves.

1. They may propose commit messages, where the indication of which AI tool was employed must be indicated as "Parts of the code generated with {name of LLM}"
