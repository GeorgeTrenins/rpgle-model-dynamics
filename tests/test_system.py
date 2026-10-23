import copy

import numpy as np
import pytest

from rpmdgle.pes.harmonic import PES
from rpmdgle.system import get_bath
from rpmdgle.sysbath.bath import CaldeiraLeggett
from rpmdgle.sysbath.coupling.exptanh import Coupling as ExpTanhCoupling
from rpmdgle.sysbath.coupling.linear import Coupling as LinearCoupling
from rpmdgle.sysbath.spectral import brownian, expohmic


MASS = 1.7
EXPOHMIC = {"name": "expohmic", "Nmodes": 8, "eta": 1.3, "omega_cut": 0.8}
BROWNIAN = {"name": "brownian", "Nmodes": 4, "eta": 0.4, "omega0": 1.5, "gamma": 0.3}


def make_pes(UNITS="atomic"):
    return PES(mass=MASS, hess=[[MASS * 0.5**2]], UNITS=UNITS)


def without(spec, key):
    return {k: v for k, v in spec.items() if k != key}


def test_no_bath():
    pes = make_pes()
    assert get_bath(pes, None, None) is pes


def test_single_density():
    pes = make_pes()
    SB = get_bath(pes, EXPOHMIC, None)
    assert isinstance(SB, CaldeiraLeggett)
    assert isinstance(SB.coupling, LinearCoupling)
    assert len(SB.densities) == 1
    density = SB.densities[0]
    assert isinstance(density, expohmic.Density)
    assert (density.eta, density.omega_cut) == (1.3, 0.8)
    ref = CaldeiraLeggett(pes, [(expohmic.Density(eta=1.3, omega_cut=0.8), 8)])
    np.testing.assert_array_equal(SB.w, ref.w)
    np.testing.assert_array_equal(SB.c, ref.c)


def test_sum_of_densities():
    SB = get_bath(make_pes(), {"terms": [EXPOHMIC, BROWNIAN]}, None)
    assert [type(density) for density in SB.densities] == [expohmic.Density, brownian.Density]
    assert SB.Nmodes == 12
    assert SB.term_slices == (slice(0, 8), slice(8, 12))
    single = get_bath(make_pes(), BROWNIAN, None)
    np.testing.assert_array_equal(SB.w[8:], single.w)
    np.testing.assert_array_equal(SB.c[8:], single.c)


def test_shared_coupling():
    F_data = {"name": "exptanh", "eps1": -0.5, "eps2": 0.3, "delta": 1.2}
    SB = get_bath(make_pes(), {"terms": [EXPOHMIC, BROWNIAN]}, F_data)
    assert isinstance(SB.coupling, ExpTanhCoupling)
    assert SB.coupling.eps1 == -0.5


def test_inputs_not_modified():
    bath_data = {"UNITS": "atomic", "terms": [EXPOHMIC, BROWNIAN]}
    F_data = {"name": "exptanh", "eps1": -0.5}
    bath_copy, F_copy = copy.deepcopy(bath_data), copy.deepcopy(F_data)
    for _ in range(2):
        get_bath(make_pes(), bath_data, F_data)
        get_bath(make_pes(), EXPOHMIC, None)
    assert bath_data == bath_copy
    assert F_data == F_copy
    assert EXPOHMIC == {"name": "expohmic", "Nmodes": 8, "eta": 1.3, "omega_cut": 0.8}


@pytest.mark.parametrize("bath_data", [
    EXPOHMIC,
    {**EXPOHMIC, "UNITS": "kcalAfs"},
    {"UNITS": "kcalAfs", "terms": [EXPOHMIC, {**BROWNIAN, "UNITS": "kcalAfs"}]},
], ids=["flat-default", "flat-explicit", "terms-explicit"])
def test_units_from_pes(bath_data):
    SB = get_bath(make_pes("kcalAfs"), bath_data, None)
    assert SB.UNITS.__class__.__name__ == "kcalAfs"
    assert SB.coupling.UNITS.__class__.__name__ == "kcalAfs"


@pytest.mark.parametrize("bath_data", [
    {**EXPOHMIC, "UNITS": "kcalAfs"},
    {"UNITS": "kcalAfs", "terms": [EXPOHMIC]},
    {"terms": [EXPOHMIC, {**BROWNIAN, "UNITS": "kcalAfs"}]},
], ids=["flat", "bath-level", "term-level"])
def test_units_mismatch(bath_data):
    with pytest.raises(RuntimeError):
        get_bath(make_pes(), bath_data, None)


@pytest.mark.parametrize("bath_data, error", [
    ({"terms": [EXPOHMIC], "Nmodes": 8}, ValueError),
    ({"terms": [EXPOHMIC], "Nmode": 8}, ValueError),
    ({"terms": []}, ValueError),
    ({"terms": EXPOHMIC}, ValueError),
    ({"terms": [{"terms": [EXPOHMIC]}]}, ValueError),
    ({"terms": [EXPOHMIC, "brownian"]}, TypeError),
    ([EXPOHMIC, BROWNIAN], TypeError),
    (without(EXPOHMIC, "name"), ValueError),
    (without(EXPOHMIC, "Nmodes"), ValueError),
    ({**EXPOHMIC, "name": "expohmc"}, ValueError),
    ({**EXPOHMIC, "name": "_base"}, ValueError),
    ({**EXPOHMIC, "Nmodes": 0}, ValueError),
    ({**EXPOHMIC, "Nmodes": 8.0}, ValueError),
    (without(EXPOHMIC, "omega_cut"), TypeError),
    ({**EXPOHMIC, "omega_cutoff": 0.8}, TypeError),
    ({"terms": [{**EXPOHMIC, "coupling": {"name": "linear"}}]}, NotImplementedError),
    ({"terms": [EXPOHMIC], "coupling": {"name": "linear"}}, NotImplementedError),
], ids=[
    "bath-level-Nmodes", "unknown-bath-key", "empty-terms", "terms-not-list", "nested-terms",
    "term-not-object", "bare-list", "missing-name", "missing-Nmodes", "unknown-name",
    "private-module", "zero-Nmodes", "float-Nmodes", "missing-parameter", "unknown-parameter",
    "term-coupling", "bath-coupling",
])
def test_invalid_spec(bath_data, error):
    with pytest.raises(error):
        get_bath(make_pes(), bath_data, None)


def test_error_names_term():
    bath_data = {"terms": [EXPOHMIC, without(BROWNIAN, "gamma")]}
    with pytest.raises(TypeError) as excinfo:
        get_bath(make_pes(), bath_data, None)
    assert "bath term 1 ('brownian')" in "\n".join(excinfo.value.__notes__)
