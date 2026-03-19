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
import importlib


def get_PES(kwargs):
    """fetch the classical external potential"""
    pesmod = importlib.import_module(kwargs.pop("module"))
    pesname = kwargs.pop("name")
    PES: BasePES = getattr(pesmod, pesname)(**kwargs)
    return PES, PES.UNITS

def get_bath_coupling(PES, coupling_data):
    """construct the system-bath coupling potential, if specified. If not specified, returns the default linear coupling."""
    if coupling_data is None:
        return coupling.linear.Coupling(UNITS=PES.UNITS.__class__.__name__)
    coupling_name = coupling_data.pop("name")
    F = getattr(coupling, coupling_name).Coupling(**coupling_data)
    return F

def get_bath(PES, bath_data, F_data):
    """construct the Caldeira-Leggett model of the dissipative system
    given the classical external potential.
    """
    if bath_data is None:
        # bare system, no bath
        return PES
    # get the interaciton potential
    F = get_bath_coupling(PES, F_data)
    # get the spectral density
    Jname = bath_data.pop("name")
    Nmodes = bath_data.pop("Nmodes")
    SB = getattr(spectral, Jname).Density(
        PES, Nmodes, coupling=F, **bath_data)
    return SB

