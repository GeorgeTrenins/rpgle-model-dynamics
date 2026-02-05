#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bathplot.py
@Time    :   2026/02/05 17:30:00
@Author  :   George Trenins
@Desc    :   Visualize spectral densities (normalized by frequency)
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from rpmdgle.system import get_PES, get_bath


def main(args):
    # Load the system PES
    with open(args.potential, 'r') as f:
        pes_data = json.load(f)
    PES, UNITS = get_PES(pes_data)
    
    # Load the bath spectral density
    if args.bath is None:
        raise ValueError("Bath specification is required for spectral density plotting")
    
    with open(args.bath, 'r') as f:
        bath_data = json.load(f)
    
    # Construct the spectral density object
    SB = get_bath(PES, bath_data, None)
    
    if SB is PES:
        raise ValueError("No bath coupling was specified")
    
    # Determine plotting range for frequency
    if hasattr(SB, 'wmax'):
        # For splined or numerical spectral densities
        xmin = args.xmin if args.xmin is not None else 0.0
        xmax = args.xmax if args.xmax is not None else SB.wmax*UNITS.hbar
    elif hasattr(SB, 'omega_cut'):
        # For exponentially damped or Debye spectral densities
        xmin = args.xmin if args.xmin is not None else 0.0
        xmax = args.xmax if args.xmax is not None else 5.0 * SB.omega_cut * UNITS.hbar
    else:
        # Default range
        xmin = args.xmin if args.xmin is not None else 0.0
        xmax = args.xmax if args.xmax is not None else 10.0
    
    # Generate frequency values and compute spectral density
    energy = np.linspace(xmin, xmax, args.npoints)
    omega = energy / UNITS.hbar  
    Lambda = SB.Lambda(omega)
    Lambda /= np.asarray(PES.mass).item()
    Lambda *= UNITS.str2base("1 ps")
    omega *= UNITS.hbar
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot Lambda(omega)
    
    ax.plot(energy, Lambda, 'b-', linewidth=1.5)
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel(r'$\Lambda(\omega) / \mathrm{m} \ [ \mathrm{time}^{-1} ]$', fontsize=12)
    ax.set_title(f'Friction Spectrum\nUnits: {UNITS.__class__.__name__}', fontsize=14)
    ax.set_xlim(xmin, xmax)

    spectral_span = Lambda.max()
    ypad = 0.05 * spectral_span
    ymin = args.ymin if args.ymin is not None else - ypad
    ymax = args.ymax if args.ymax is not None else Lambda.max() + ypad
    ax.set_ylim(ymin, ymax)
    
    # Mark the discrete quadrature frequencies if requested
    if args.show_quad:
        quad_freqs = SB.w
        quad_energies = quad_freqs * UNITS.hbar
        for energy in quad_energies:
            if xmin <= energy <= xmax:
                ax.axvline(energy, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        # Add legend entry
        ax.plot([], [], 'gray', linestyle='--', linewidth=0.8, 
                label=f'Quadrature points (N={SB.Nmodes})')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Display interactive window
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot spectral density from JSON input files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'potential', 
        type=str, 
        help="JSON file with PES parameters"
    )
    parser.add_argument(
        '--bath',
        type=str,
        required=True,
        help="JSON file with spectral density parameters"
    )
    parser.add_argument(
        '--xmin', 
        type=float, 
        default=None, 
        help="Minimum energy for plotting"
    )
    parser.add_argument(
        '--xmax', 
        type=float, 
        default=None, 
        help="Maximum energy for plotting"
    )
    parser.add_argument(
        '--ymin', 
        type=float, 
        default=None, 
        help="Minimum y value for plotting"
    )
    parser.add_argument(
        '--ymax', 
        type=float, 
        default=None, 
        help="Maximum y value for plotting"
    )
    parser.add_argument(
        '--npoints', 
        type=int, 
        default=500, 
        help="Number of points for plotting"
    )
    parser.add_argument(
        '--show-quad',
        action='store_true',
        help="Show discrete quadrature frequencies as vertical lines"
    )
    
    args = parser.parse_args()
    main(args)