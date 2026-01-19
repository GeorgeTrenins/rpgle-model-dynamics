#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   expohmic.py
@Time    :   2024/12/18 11:36:12
@Author  :   George Trenins
@Desc    :   Parse and scale auxiliary variable propagation parameters for the exponentially damped Ohmic spectral density and its ring-polymer normal mode transforms.
'''

from __future__ import print_function, division, absolute_import
from rpmdgle.utils.fitaux._base import BaseOneAuxParam, MultiAuxParams, get_nm_freqs
import numpy as np
import argparse
from scipy.special import sici

parser = argparse.ArgumentParser(description="Create input file for auxiliary-variable propagation, appropriately interpolating and scaling pre-fitted data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', help='Path to the data file with fitted coefficients.')
parser.add_argument('units', help='Name of unit system', choices=[
    'SI', 'atomic', 'hartAng', 'kcalAfs', 'kcalAamu', 'eVAamu', 'cmbohramu'
])
parser.add_argument('--oneaux', action='store_true', help='Use dedicated parsers for parameters fitted to a single auxiliary variable')
parser.add_argument('--wc', type=float, required=True, help='Value of the cut-off frequency in wavenumbers')
parser.add_argument('--mass', type=str, required=True, help='Particle mass; if no units are given, the value is assumed to be in base units.')
parser.add_argument('--wb', type=float, required=True, help='Characteristic frequency, in wavenumbers, for defining the unit of static friction')
parser.add_argument('--eta0', type=float, default=1, help='Static friction in units of mass*wb')
parser.add_argument('-T', type=float, required=True, help='Temperature in Kelvin')
parser.add_argument('-N', type=int, required=True, help='Number or ring-polymer beads')
parser.add_argument('-n', type=int, nargs="+", default=None, help='Indices of the normal modes for which to compute the coefficients (all possible indices are computed by default).')


def zero_time_kernel(ww_n):
    """Analytical expression for the zero-time value of the memory-friction
    kernel for a ring-polymer normal mode with frequency ww_n at zero time.
    """
    if ww_n == 0:
        ans = 2/np.pi
    else:
        si, ci = sici(ww_n)
        s, c = np.sin(ww_n), np.cos(ww_n)
        ans = 2/np.pi * (1 - ww_n * (
                ci*s - (si-np.pi/2)*c
            ))
    return ans

class OneAuxParam(BaseOneAuxParam):
    def extrapolate(self, w):
        if 'omega' in self.col:
            # extrapolating oscillator frequency
            return np.copy(w)
        elif 'tau' in self.col:
            # extrapolating decay timescale
            return 0.4415*w
        else:
            # extrapolating coupling coefficient
            return np.sqrt(np.asarray([zero_time_kernel(s) for s in w]))
        

def main(args: argparse.Namespace):

    from rpmdgle import units
    import json
    
    UNITS = getattr(units, args.units)()
    wc = UNITS.wn2omega(args.wc)
    wb = UNITS.wn2omega(args.wb)
    m = UNITS.str2base(args.mass)
    eta0 = args.eta0 * (m*wb)
    beta = UNITS.betaTemp(args.T)
    wn = get_nm_freqs(beta, UNITS.hbar, args.N, n=args.n)
    wn_reduced = wn/wc
    columns = ['tauD', 'cD', 'tauO', 'cO', 'omegaO']
    if args.oneaux:
        tau_fun = OneAuxParam(args.data, 'tauO', A=10, wc=25, b=1, smoothing=0.0001)
        c_fun = OneAuxParam(args.data, 'cO', A=10, wc=25, b=1, smoothing=0.0001)
        w_fun = OneAuxParam(args.data, 'omegaO', A=10, wc=25, b=1, smoothing=0.0001)
        calculators = [None, None, tau_fun, c_fun, w_fun]
    else:
        calculators = []
        for column in columns:
            try:
                if 'tau' in column:
                    calculator = MultiAuxParams(
                        args.data, column, A=10, wc=2, b=0.1, 
                        smoothing=0.005, sort='ascending', sorting_cutoff=2.0)
                else:
                    calculator = MultiAuxParams(
                        args.data, column, A=10, wc=50, b=1, 
                        smoothing=0.00015, sort='none')
            except:
                calculator = None
            calculators.append(calculator)
    for i,w in enumerate(wn_reduced):
        if i != 0:
            print(",")
        data = dict()
        for calculator, column in zip(calculators, columns):
            if calculator is None:
                continue
            value = calculator(w)
            # Scale to system units
            if 'tau' in column:
                value = value / wc
            elif 'omega' in column:
                value = value * wc
            else:
                value = value * np.sqrt(eta0 * wc)
            data[column] = np.atleast_1d(value).tolist()
        print(json.dumps(data, indent=4), end='')
    print()
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)