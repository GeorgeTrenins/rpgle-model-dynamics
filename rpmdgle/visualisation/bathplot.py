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
    xmin = args.xmin if args.xmin is not None else 0.0
    if args.xmax is not None:
        xmax = args.xmax
    elif SB.wplot is not None:
        xmax = SB.wplot * UNITS.hbar
    else:
        # Default range
        xmax = 10.0

    # Generate frequency values and compute spectral density
    energy = np.linspace(xmin, xmax, args.npoints)
    omega = energy / UNITS.hbar
    # convert Λ(ω) to Λ(ω)/m in ps^-1
    Lambda_scale = UNITS.str2base("1 ps") / np.asarray(PES.mass).item()
    Lambda = SB.Lambda(omega) * Lambda_scale
    multiple = len(SB.densities) > 1
    term_names = [type(density).__module__.rsplit('.', 1)[-1] for density in SB.densities]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot Lambda(omega)

    ax.plot(energy / UNITS.str2base("1 eV"), Lambda, 'b-', linewidth=1.5,
            label='Total' if args.terms else None)
    if args.terms:
        for k, (density, name) in enumerate(zip(SB.densities, term_names)):
            ax.plot(energy / UNITS.str2base("1 eV"), density.Lambda(omega) * Lambda_scale,
                    color=f'C{k+1}', linewidth=1.0, label=f'Term {k}: {name}')
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel(r'$\Lambda(\omega) / \mathrm{m} \ [ \mathrm{ps}^{-1} ]$', fontsize=12)
    ax.set_title(f'Friction Spectrum\nUnits: {UNITS.__class__.__name__}', fontsize=14)
    ax.set_xlim(xmin / UNITS.str2base("1 eV"), xmax / UNITS.str2base("1 eV"))

    spectral_span = Lambda.max()
    ypad = 0.05 * spectral_span
    ymin = args.ymin if args.ymin is not None else - ypad
    ymax = args.ymax if args.ymax is not None else Lambda.max() + ypad
    ax.set_ylim(ymin, ymax)
    
    # Mark the discrete quadrature frequencies if requested, coloured by term
    if args.show_quad:
        for k, sl in enumerate(SB.term_slices):
            color = f'C{k+1}' if multiple else 'gray'
            quad_energies = SB.w[sl] * UNITS.hbar
            for quad_energy in quad_energies:
                if xmin <= quad_energy <= xmax:
                    ax.axvline(quad_energy / UNITS.str2base("1 eV"), color=color, linestyle='--', alpha=0.3, linewidth=0.8)
            # Add legend entry
            label = f'Quadrature points (N={SB.Nmodes})' if not multiple else \
                f'Quadrature points, term {k} (N={sl.stop - sl.start})'
            ax.plot([], [], color=color, linestyle='--', linewidth=0.8, label=label)
    if args.show_quad or args.terms:
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
    parser.add_argument(
        '--terms',
        action='store_true',
        help="Also plot each term of a bath that is a sum of spectral densities"
    )
    
    args = parser.parse_args()
    main(args)