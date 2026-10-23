#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   system.py
@Time    :   2026/02/05 16:48:29
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES
from rpmdgle.sysbath import spectral, coupling
from rpmdgle.sysbath.bath import CaldeiraLeggett
from rpmdgle.sysbath.spectral._base import BaseSpectralDensity
from types import ModuleType
from typing import List, Optional, Tuple
import copy
import importlib


def get_PES(kwargs):
    """fetch the classical external potential"""
    pesmod = importlib.import_module(kwargs.pop("module"))
    pesname = kwargs.pop("name")
    PES: BasePES = getattr(pesmod, pesname)(**kwargs)
    return PES, PES.UNITS

def _check_units(PES: BasePES, UNITS: Optional[str], what: str) -> None:
    """Raise if a unit system named in the specification of `what` differs from that of the PES."""
    expected = PES.UNITS.__class__.__name__
    if UNITS is not None and UNITS != expected:
        raise RuntimeError(
            f"The {what} and the potential appear to have "
            f"different unit systems, {UNITS} and {expected}, respectively")

def _submodule_names(package: ModuleType) -> List[str]:
    """Names of the public submodules of `package` that can be selected by "name" in a specification."""
    return sorted(name for name, obj in vars(package).items()
                  if isinstance(obj, ModuleType) and not name.startswith("_"))

def get_bath_coupling(PES, coupling_data):
    """construct the system-bath coupling potential, if specified. If not specified,
    returns the default linear coupling. A coupling is specified by a dictionary such as

        {"name": "exptanh", "eps1": -1.0, "delta": "0.5 angstrom"}

    "name" selects a submodule of rpmdgle.sysbath.coupling, and the remaining keys are
    passed to the constructor of its Coupling. "UNITS" is optional and must match the
    unit system of the PES, which is passed to the constructor.

    Args:
        PES (BasePES): system potential
        coupling_data (dict or None): coupling specification; it is not modified
    """
    units_name = PES.UNITS.__class__.__name__
    if coupling_data is None:
        return coupling.linear.Coupling(UNITS=units_name)
    if not isinstance(coupling_data, dict):
        raise TypeError(f"Expected the coupling to be specified by a JSON object, instead got {type(coupling_data).__name__}")
    if "name" not in coupling_data:
        raise ValueError("A coupling requires the key 'name'")
    spec = dict(coupling_data)
    name = spec.pop("name")
    if name not in _submodule_names(coupling):
        raise ValueError(f"Unknown coupling {name!r}; expected one of {_submodule_names(coupling)}")
    _check_units(PES, spec.pop("UNITS", None), "coupling")
    try:
        return getattr(coupling, name).Coupling(UNITS=units_name, **spec)
    except Exception as e:
        e.add_note(f"Raised while constructing the coupling ({name!r})")
        raise

def _get_bath_term(PES: BasePES, term: dict) -> Tuple[BaseSpectralDensity, int]:
    """Construct one spectral density from its specification, which is consumed in the process."""
    if not isinstance(term, dict):
        raise TypeError(f"Expected a spectral density to be specified by a JSON object, instead got {type(term).__name__}")
    if "terms" in term:
        raise ValueError("Bath terms cannot be nested")
    if "coupling" in term:
        raise NotImplementedError(
            "Couplings for individual bath terms are not supported yet; "
            "specify a coupling shared by all terms with --coupling")
    for key in ("name", "Nmodes"):
        if key not in term:
            raise ValueError(f"A spectral density requires the key '{key}'")
    name = term.pop("name")
    Nmodes = term.pop("Nmodes")
    _check_units(PES, term.pop("UNITS", None), "spectral density")
    if name not in _submodule_names(spectral):
        raise ValueError(f"Unknown spectral density {name!r}; expected one of {_submodule_names(spectral)}")
    return getattr(spectral, name).Density(**term), Nmodes

def get_bath_terms(PES: BasePES, bath_data: dict) -> List[Tuple[BaseSpectralDensity, int]]:
    """Construct the spectral densities that make up the bath. A single spectral
    density is specified by a flat dictionary,

        {"name": "expohmic", "Nmodes": 64, "eta": 1.0, "omega_cut": 0.002}

    and a sum of spectral densities lists them under "terms",

        {"UNITS": "atomic",
         "terms": [{"name": "expohmic", "Nmodes": 48, "eta": 1.0, "omega_cut": 0.002},
                   {"name": "brownian", "Nmodes": 16, "eta": 0.2, "omega0": 0.001, "gamma": 0.0005}]}

    In each term, "name" selects a submodule of rpmdgle.sysbath.spectral, "Nmodes" is the
    number of bath modes used to discretise that term, and the remaining keys are passed to
    the constructor of its Density. "UNITS" is optional at either level and must match the
    unit system of the PES.

    Args:
        PES (BasePES): system potential
        bath_data (dict): bath specification; it is not modified

    Returns:
        list of (density, Nmodes) pairs
    """
    spec = copy.deepcopy(bath_data)
    if not isinstance(spec, dict):
        raise TypeError(f"Expected the bath to be specified by a JSON object, instead got {type(spec).__name__}")
    if "terms" in spec:
        term_specs = spec.pop("terms")
        if "coupling" in spec:
            raise NotImplementedError(
                "A coupling in the bath specification is not supported yet; "
                "specify a coupling shared by all terms with --coupling")
        if "Nmodes" in spec:
            raise ValueError("Specify 'Nmodes' in each of the bath 'terms', not for the bath as a whole")
        _check_units(PES, spec.pop("UNITS", None), "spectral density")
        if spec:
            raise ValueError(f"Unknown keys in the bath specification: {sorted(spec)}")
        if not isinstance(term_specs, list) or len(term_specs) == 0:
            raise ValueError("Expected 'terms' to be a non-empty list of spectral densities")
    else:
        term_specs = [spec]
    terms = []
    for i, term in enumerate(term_specs):
        name = term.get("name") if isinstance(term, dict) else None
        try:
            terms.append(_get_bath_term(PES, term))
        except Exception as e:
            e.add_note(f"Raised while constructing bath term {i}" +
                       ("" if name is None else f" ({name!r})"))
            raise
    return terms

def get_bath(PES, bath_data, F_data):
    """construct the Caldeira-Leggett model of the dissipative system
    given the classical external potential. See get_bath_terms for the
    format of bath_data.
    """
    if bath_data is None:
        # bare system, no bath
        return PES
    # get the interaciton potential
    F = get_bath_coupling(PES, F_data)
    # get the spectral densities
    terms = get_bath_terms(PES, bath_data)
    return CaldeiraLeggett(PES, terms, coupling=F)
