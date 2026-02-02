#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Contributed by George Trenins.

Some common unit systems for converting to and from SI. A unit system is defined in terms of seven *base units* that span the dimensions of length, mass, time, electric current, amount of substance, luminous intensity, and thermodynamic temperature.
'''


from __future__ import print_function, division, absolute_import
from scipy import constants as sc
import sympy.physics.units as u
import sys
from sympy import pi
import math


# A map from dimension names to SymPy Dimension objects
dimensions = {
    "energy" : u.energy,
    "length" : u.length,
    "time" : u.time,
    "mass" : u.mass,
    "charge" : u.charge,
    "luminous_intensity" : u.luminous_intensity,
    "amount" : u.amount,
    "current" : u.current,
    "temperature" : u.temperature,
    "action" : u.action,
    "angular_momentum" : u.action,
    "vacuum_permittivity" : u.capacitance / u.length,
}

# All systems here are dimensionally consistent with SI
_SI = u.systems.SI

# Define extra quantities missing in SymPy
aa = angstrom = angstroms = u.Quantity("angstrom", abbrev="AA")
_SI.set_quantity_dimension(angstrom, u.length)
_SI.set_quantity_scale_factor(angstrom, u.meter / 10**10)

me = electron_mass = u.Quantity("electron_mass", abbrev="me")
_SI.set_quantity_dimension(me, u.mass)
_SI.set_quantity_scale_factor(me, sc.m_e * u.kg)

mp = proton_mass = u.Quantity("proton_mass", abbrev="mp")
_SI.set_quantity_dimension(mp, u.mass)
_SI.set_quantity_scale_factor(mp, sc.m_p * u.kg) 

alpha = fine_structure_constant = sc.fine_structure

a0 = bohr_radius = bohr_radii = u.Quantity("bohr_radius", abbrev="a0")
_SI.set_quantity_dimension(a0, u.length)
_SI.set_quantity_scale_factor(a0, sc.value(u'Bohr radius')*u.meter)

Eh = hartree = hartrees = u.Quantity("hartree", abbrev="Eh")
_SI.set_quantity_dimension(hartree, u.energy)
_SI.set_quantity_scale_factor(hartree, sc.value(u'Hartree energy')*u.J)

kcal_mol = u.Quantity("kcal_mol")
_SI.set_quantity_dimension(kcal_mol, u.energy)
_SI.set_quantity_scale_factor(kcal_mol, sc.kilo*sc.calorie/sc.N_A * u.J)

fs = femtosecond = femtoseconds = u.Quantity("femtosecond", abbrev="fs")
_SI.set_quantity_dimension(fs, u.time)
_SI.set_quantity_scale_factor(fs, sc.femto*u.second)

coulomb_constant = u.Quantity("coulomb_constant", abbrev="ke")
_SI.set_quantity_dimension(coulomb_constant, u.length/u.capacitance)
_SI.set_quantity_scale_factor(coulomb_constant, 1/(4*pi*u.vacuum_permittivity))

wn = wavenumber = wavenumbers = u.Quantity("wavenumber", abbrev="wn")
_SI.set_quantity_dimension(wn, u.energy)
_SI.set_quantity_scale_factor(wn, u.planck*u.c/u.cm)


def _convert_to(quantity, base_units):
    # Takes a Quantity or a Dimension object. If Quantity, gets expressed
    # in the specified base_units. If Dimension, the appropriate combination
    # of base units is determined and a Quantity of 1.0*[combination of base units]
    # is returned
    try:
        dimension = quantity.dimension
        scale = 1.0*quantity
    except AttributeError:
        dimension = quantity
        scale = None
    ans = u.Quantity("derived")
    _SI.set_quantity_dimension(ans, dimension)
    if scale is not None:
        _SI.set_quantity_scale_factor(ans, scale)
        return u.convert_to(ans, base_units).n()
    else:
        ans = u.convert_to(1.0*ans, base_units).n()
        return 1.0*ans.as_two_terms()[1]


class SI(object):
    """Base units of metre, kilogram, second, ampere, mole, candela and Kelvin.
    """

    base_units = (u.meter, u.kilogram, u.second, u.ampere, u.mol, u.cd, u.K)

    # Physical constants
    @property
    def hbar(self):
        """
        Value of the reduced Planck constant in base units.
        """
        return float(self.in_base(u.hbar).as_two_terms()[0])

    @property
    def e(self):
        """
        Absolute value of the charge of the electron in base units.
        """
        return float(self.in_base(u.elementary_charge).as_two_terms()[0])

    @property
    def kb(self):
        """
        Boltzmann constant in base units.
        """
        return float(self.in_base(u.boltzmann_constant).as_two_terms()[0])

    @property
    def amu(self):
        """
        Atomic mass unit (Dalton) in base units.
        """
        return float(self.in_base(u.amu).as_two_terms()[0])

    @property
    def m_e(self):
        """
        Mass of the electron in base units.
        """
        return float(self.in_base(me).as_two_terms()[0])

    me = m_e

    @property
    def c(self):
        """
        Speed of light in vacuum, in base units.
        """
        return float(self.in_base(u.c).as_two_terms()[0])


    def in_base(self, quantity):
        """
        Convert a quantity to base units.

        :param quantity: a physical quantity: can be a unit of measure, a constant or a generic quantity.
        :type quantity: sympy.physics.units.quantities.Quantity
        :return: converted quantity
        :rtype: sympy.physics.units.quantities.Quantity
        """
        return _convert_to(quantity, self.base_units)

    def in_SI(self, quantity):
        """
        Convert a quantity to SI units.

        :param quantity: a physical quantity: unit of measure, constant or a generic quantity.
        :type quantity: sympy.physics.units.quantities.Quantity
        :return: converted quantity
        :rtype: sympy.physics.units.quantities.Quantity
        """
        return _convert_to(quantity, SI.base_units)
    
    def str2valunit(self, quantity):
        """
        Given a string representation of a quantity, such as '1 mm', return a 2-tuple containing the numerical value (in this case, `1`) and the unit object (in this case, `sympy.physics.units.millimeter`)

        :param quantity: a physical quantity in the format 'value unit'
        :type quantity: string
        :return: a value-unit tuple
        :rtype: tuple[float, sympy.physics.units.Quantity]
        """
        string_list = quantity.split()
        if len(string_list) == 1:
            value = string_list[0]
            uobj = None
        elif len(string_list) == 2:
            value, unit = string_list
            try:
                uobj = getattr(u, unit)
            except AttributeError:
                try:
                    uobj = getattr(sys.modules[__name__], unit)
                except AttributeError as e:
                    e.add_note(f'Unknown unit "{unit}"')
                    raise
            if not isinstance(uobj, u.Quantity):
                raise ValueError(f'"{unit}" is not a valid unit, {type(uobj) = }')
        else:
            raise ValueError("The input for string conversion must be of the form 'value' or 'value unit'")
        try:
            valnum = float(value)
        except ValueError as e:
            e.add_note(f"{value} is not a valid value")
            raise
        return valnum, uobj
    
    def str2base(self, string):
        """
        Given a string representation of a quantity, such as '1 mm', return the corresponding numerical value in base units. If not of type `str`, the input us returned unchanged. If the string contains no units, it is simply converted to a float.

        :param quantity: a physical quantity in the format 'value unit'
        :type quantity: string
        :return: converted quantity
        :rtype: float
        """
        if type(string) is str:
            value, unit = self.str2valunit(string)
            if unit is None:
                return value
            else:
                return value * float(self.in_base(unit).as_two_terms()[0])
        else:
            return string
        
    def str2SI(self, string):
        """
        Return the numerical value of the input string in SI units. See `str2base` for details.
        """
        if type(string) is str:
            value, unit = self.str2valunit(string)
            if unit is None:
                return value
            else:
                return value * float(self.in_SI(unit).as_two_terms()[0])
        else:
            return string

    def __getattr__(self, attr):
        try:
            dim = dimensions[attr]
        except KeyError:
            raise AttributeError("Trying to access an unknown property {:}".format(attr))
        else:
            SI_units = self.in_SI(dim).as_two_terms()[1]
            return float(
                u.convert_to(self.in_base(dim), SI_units).n().as_two_terms()[0])

    # Temperature conversion
    def betaTemp(self, beta):
        r"""Perform the conversion :math:`\beta = 1/k_B T`

        :param beta: thermodynamic temperature in base units
        :type quantity: float
        :return: :math:`\beta = 1/k_B T` in base units
        :rtype: float
        """
        return 1.0/(self.kb*beta)
    
    # Wavenumber conversion
    def energy2wn(self, E):
        r"""Convert from energy to wavenumbers.

        :param E: energy in base units
        :type quantity: float
        :return: wavenumber in :math:`\mathrm{cm}^{-1}`
        :rtype: float
        """
        return E * self.energy*self.time / (
            200*math.pi*self.c*self.hbar * self.length*self.action)
    
    def wn2energy(self, wn):
        r"""Convert from wavenumbers to energy.

        :param wn: wavenumber in :math:`\mathrm{cm}^{-1}`
        :type quantity: float
        :return: energy in base units
        :rtype: float
        """
        return 200*math.pi*self.c*self.hbar*wn * self.length*self.action / (
            self.energy*self.time)
    
    def omega2wn(self, w):
        r"""Convert from radial frequency to wavenumbers.

        :param w: radial frequency in units of rad per base unit of time
        :type quantity: float
        :return: wavenumber in :math:`\mathrm{cm}^{-1}`
        :rtype: float
        """
        return w / ( 200*math.pi * (self.c * self.length) )

    def wn2omega(self, wn):
        r"""Convert from wavenumbers to radial frequency .

        :param wn:  wavenumber in :math:`\mathrm{cm}^{-1}`
        :type quantity: float
        :return: radial frequency in units of rad per base unit of time
        :rtype: float
        """
        return wn * ( 200*math.pi * (self.c * self.length) )


class atomic(SI):
    r"""Base units of :math:`\hbar`, mass of electron, Bohr radius, and charge of electron.
    """
    base_units = (u.hbar, u.elementary_charge, a0, me, u.mol, u.cd, u.K)

class hartAng(SI):
    r"""Base units of :math:`\hbar`, Hartree, angstrom, and charge of electron.
    """
    base_units = (u.hbar, u.elementary_charge, aa, Eh, u.mol, u.cd, u.K)

class kcalAfs(SI):
    r"""Base units of :math:`\text{kcal}\cdot\text{mol}^{-1}`, angstrom, femtosecond, and :math:`1/4\pi\epsilon_0`
    """
    base_units = (kcal_mol, aa, fs, coulomb_constant, u.mol, u.cd, u.K)

class kcalAamu(SI):
    r"""Base units of :math:`\text{kcal}\cdot\text{mol}^{-1}`, angstrom, Dalton, and :math:`1/4\pi\epsilon_0`
    """
    base_units = (kcal_mol, aa, u.amu, coulomb_constant, u.mol, u.cd, u.K)

class eVAamu(SI):
    r"""Base units of electronvolt, angstrom, Dalton, and coulomb
    """
    base_units = (u.electronvolt, aa, u.amu, u.coulomb, u.mol, u.cd, u.K)
