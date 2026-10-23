# RPMD GLE

Python implementation of dissipative ring-polymer molecular dynamics for simple model systems.


## Input files


### Potential energy surface

### Harmonic bath

The bath file (`--bath`) specifies the spectral density of a harmonic (Caldeira–Leggett) bath. `name` selects a module in `rpmdgle.sysbath.spectral` (`expohmic`, `debye`, `brownian` or `splined`), `Nmodes` is the number of discrete bath modes, and the remaining keys are the parameters of the spectral density, as bare numbers in the internal units of the PES.

```json
{
    "name" : "expohmic",
    "Nmodes" : 64,
    "eta" : 1.0,          // static friction, including the system mass
    "omega_cut" : 0.002
}
```

A spectral density that is a sum of several terms lists them under `terms`. Each term is discretised separately with its own `Nmodes`, so the bath has 48 + 16 modes here.

```json
{
    "terms" : [
        {"name" : "expohmic", "Nmodes" : 48, "eta" : 1.0, "omega_cut" : 0.002},
        {"name" : "brownian", "Nmodes" : 16, "eta" : 0.2, "omega0" : 0.001, "gamma" : 0.0005}
    ]
}
```

All terms couple to the system through the same function, which is linear unless a coupling file is given with `--coupling`. An optional `"UNITS"` key must match the unit system of the PES.

### Propagator

#### Velocity Verlet

Classical energy-conserving dynamics.

```json
{
    "class" : "VelocityVerlet",
    "dt" : "0.25 fs"    // will be converted to PES internal units
}
```

#### (White-Noise) Langevin Equation Thermostat

```json
{
    "class" : "Langevin",
    "dt" : "0.25 fs",
    "tau" : "100 fs"
}
```

#### Stochastic Velocity-Rescaling Thermostat

```json
{
    "class" : "SVR",
    "dt" : "0.25 fs",
    "tau" : "100 fs"
}
```

#### RPMD

Normal-mode (RESPA) propagator for ring-polymer molecular dynamics.

```json
{
    "class" : "RingNM",
    "dt" : "0.25 fs"    // will be converted to PES internal units
}
```

#### TRPMD and PIMD

PILE-L thermostat with non-centroid damping coefficients for the modes $\tilde{q}^{(n)}$ with free ring-polymer frequencies $\tilde{\omega}^{(n)}$ set to $2 \lambda |\tilde{\omega}^{(n)}|$. For TRPMD, set the centroid thermostat constant to a large value.

```json
{
    "class" : "RingPILE",
    "dt" : "0.25 fs",
    "tau" : "100 fs", // centroid thermostat constant
    "lamda" : 1       // the default value is 0.5   
}
```

#### PIMD + dissipative bath

Like PILE-L, except the propagator takes care of calculating the potential of mean field from coupling to a dissipative environment and adds that to the external potential experienced by the ring-polymer normal modes. The dynamical friction forces in the underlying RPMD GLE equations do not affect the equilibrium distribution are **not** simulated here: the frictional and random forces are fictitious and are just there to thermalise the ring-polymer.

```json
{
    "class" : "SepGLEPILE",
    "dt" : "0.25 fs",
    "tau" : "100 fs", // centroid thermostat constant
    "lamda" : 1       // the default value is 0.5   
}
```

#### RPMD GLE

The full auxiliary-variable propagator for the RPMD GLE, which includes the mean-field correction to the external potential as well as the correct friction and stochastic forces, represented through an auxiliary-variable mapping.

```json
{
    "class" : "SepGLEaux",
    "dt" : "0.25 fs",
    "aux" :  [
        {...},  // List of dictionaries, one per normal mode
        {...},  // in the order 0, -1, +1, -2, +2, ...
        {...}
    ]
}
```

Each dictionary in `aux` looks as follows
```json
{
    "friction_scale"  : 0.1,    // optional, default 1
    "frequency_scale" : 2.0,    // optional, default 1
    // EITHER
    "external"        : "path/to/optimized_Ap_matrix.json",
    // OR
    "Ap"              : [
          [ 0.0000000e+00, 9.7161755e+00, 0.0000000e+00],
          [-9.7161755e+00, 7.9322533e+00, 2.4247703e+00],
          [-0.0000000e+00,-2.4247703e+00, 3.9845801e-03]
    ]
}
```
If an `external` file is specified for the drift matrix, its contents mirror the internal specification:
```json
// contents of "path/to/optimized_Ap_matrix.json"
{ 
    "Ap" : [
          [ 0.0000000e+00, 9.7161755e+00, 0.0000000e+00],
          [-9.7161755e+00, 7.9322533e+00, 2.4247703e+00],
          [-0.0000000e+00,-2.4247703e+00, 3.9845801e-03]
    ]
}
```

The optional **scaling** can be used to take drift matrices that were fitted to a friction kernel $K(t)$ and scale them to simulate a kernel $\alpha \beta \, K(\beta t)$, so that the **static friction** is scaled by a factor of $\alpha$ and the bath relaxation time-constant is scaled by a factor of $\beta^{-1}$ (equivalently, the cutoff frequency is scaled by $\beta$).