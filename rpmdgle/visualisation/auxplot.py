#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   auxplot.py
@Time    :   2026/02/12 14:26:00
@Author  :   George Trenins
@Desc    :   Visualize auxiliary variable GLE friction kernels and spectra
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from rpmdgle.system import get_PES, get_bath
from rpmdgle.propagators.gle import parse_aux_spec
from rpmdgle.utils.ou import get_friction_kernel, get_friction_spectrum


def get_normal_mode_index(aux_idx):
    """Convert auxiliary variable list index to normal mode index.
    
    The ordering is: 
        0, -1, +1, ..., -(nbeads/2 - 1), +(nbeads/2 - 1), -nbeads/2 [for even nbeads]
        0, -1, +1, ..., -(nbeads - 1)/2, +(nbeads - 1)/2            [for odd nbeads]
    
    Args:
        aux_idx (int): index in the aux list
        
    Returns:
        int: normal mode index
    """
    if aux_idx == 0:
        return 0
    elif aux_idx % 2 == 1:  
        return -(aux_idx + 1) // 2
    else: 
        return aux_idx // 2


def main(args):
    # Load the system PES
    with open(args.potential, 'r') as f:
        pes_data = json.load(f)
    PES, UNITS = get_PES(pes_data)
    
    # Load the bath spectral density
    if args.bath is None:
        raise ValueError("Bath specification is required (--bath)")
    
    with open(args.bath, 'r') as f:
        bath_data = json.load(f)
    
    # Construct the spectral density object for reference
    SB = get_bath(PES, bath_data, None)
    
    if SB is PES:
        raise ValueError("No bath coupling was specified in the bath file")
    
    # Load propagator parameters
    if args.propagator is None:
        raise ValueError("Propagator specification is required (--propagator)")
    
    with open(args.propagator, 'r') as f:
        prop_data = json.load(f)
    
    if 'aux' not in prop_data:
        raise ValueError("Propagator file must contain an 'aux' key with auxiliary variable specifications")
    
    aux_list = prop_data['aux']
    nbeads = len(aux_list)
    
    # Determine which modes to plot
    if args.modes is None:
        mode_indices = list(range(nbeads))
    else:
        mode_indices = args.modes
        for idx in mode_indices:
            if idx < 0 or idx >= nbeads:
                raise ValueError(f"Mode index {idx} out of range [0, {nbeads-1}]")
    
    nmodes = len(mode_indices)
    
    # Parse auxiliary variable specifications and build Ap matrices
    Ap_matrices = []
    for idx in mode_indices:
        try:
            Ap = parse_aux_spec(aux_list[idx])
            Ap_matrices.append(Ap)
        except Exception as e:
            raise ValueError(f"Error parsing aux specification at index {idx}: {str(e)}")
    
    # Generate time grid for friction kernel
    tmax, time_unit = UNITS.str2valunit(args.tmax)
    tmax_u = UNITS.str2base(args.tmax)
    # Units of the command-line argument:
    t = np.linspace(0, tmax, args.npoints)
    # Units of the potential (used in all calculations)
    t_u = np.linspace(0, tmax_u, args.npoints)
    
    # Generate energy/frequency grid for friction spectrum
    emax, energy_unit = UNITS.str2valunit(args.emax)
    emax_u = UNITS.str2base(args.emax)
    energy = np.linspace(0, emax, args.npoints)
    energy_u = np.linspace(0, emax_u, args.npoints)
    omega = energy_u / UNITS.hbar
    
    # Compute reference curves
    K_ref = SB.K(t_u)
    K_ref /= np.asarray(PES.mass).item()
    K_ref *= UNITS.str2base(f"1 {time_unit}")**2 
    
    Lambda_ref = SB.Lambda(omega)
    Lambda_ref /= np.asarray(PES.mass).item()
    Lambda_ref *= UNITS.str2base(f"1 {time_unit}") # in ps^-1
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows=2, ncols=nmodes, sharex="row", figsize=(6, 6*nmodes))
    if nmodes == 1:
        axes = axes[:, np.newaxis]  # Ensure 2D array for consistency
    
    # Plot each mode
    for plot_idx, aux_idx in enumerate(mode_indices):
        Ap = Ap_matrices[plot_idx]
        nm_idx = get_normal_mode_index(aux_idx)
        
        # Compute auxiliary variable friction kernel and spectrum
        K_aux = get_friction_kernel(t_u, Ap)
        K_aux /= np.asarray(PES.mass).item()
        K_aux *= UNITS.str2base(f"1 {time_unit}")**2 
        
        Lambda_aux = get_friction_spectrum(omega, Ap)
        Lambda_aux /= np.asarray(PES.mass).item()
        Lambda_aux *= UNITS.str2base(f"1 {time_unit}") # in ps^-1
        
        # Plot friction kernel
        ax_K = axes[0, plot_idx]
        ax_K.plot(t, K_ref, 'k-', linewidth=2, label='Reference', alpha=0.7)
        ax_K.plot(t, K_aux, 'r-', linewidth=1.5, label='Auxiliary')
        ax_K.set_xlabel(f'Time ({time_unit})', fontsize=8)
        ax_K.set_ylabel(rf'$K(t) / \mathrm{{m}} \ [ \mathrm{{{time_unit}}}^{-2} ]$', fontsize=8)
        ax_K.set_title(f'Friction Kernel: Mode n={nm_idx}', fontsize=9)
        ax_K.grid(True, alpha=0.3)
        ax_K.legend()
        
        
        # Plot friction spectrum
        ax_Lambda = axes[1, plot_idx]
        ax_Lambda.plot(energy, Lambda_ref, 'k-', linewidth=2, label='Reference', alpha=0.7)
        ax_Lambda.plot(energy, Lambda_aux, 'r-', linewidth=1.5, label='Auxiliary')
        ax_Lambda.set_xlabel(f'Energy ({energy_unit})', fontsize=8)
        ax_Lambda.set_ylabel(rf'$\Lambda(\omega) / \mathrm{{m}} \ [ \mathrm{{{time_unit}}}^{-1} ]$', fontsize=8)
        ax_Lambda.set_title(f'Friction Spectrum: Mode n={nm_idx}', fontsize=8)
        ax_Lambda.grid(True, alpha=0.3)
        ax_Lambda.legend()
        
    
    fig.suptitle(f'GLE Auxiliary Variable Analysis\nUnits: {UNITS.__class__.__name__}', 
                 fontsize=10, y=0.995)
    plt.tight_layout()
    
    # Display interactive window
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot friction kernels and spectra for auxiliary variable GLE from JSON input files.",
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
        help="JSON file with bath spectral density parameters"
    )
    parser.add_argument(
        '--propagator',
        type=str,
        required=True,
        help="JSON file with propagator parameters (must contain 'aux' key)"
    )
    parser.add_argument(
        '--modes',
        type=int,
        nargs='+',
        default=None,
        help="Indices of aux list elements to plot (default: all modes)"
    )
    parser.add_argument(
        '--tmax',
        type=str,
        required=True,
        help="Maximum time for friction kernel plot as a string with units (e.g. '5 ps')"
    )
    parser.add_argument(
        '--emax',
        type=str,
        required=True,
        help="Maximum energy for friction spectrum plot as a string with units (e.g. '0.1 eV')"
    )
    parser.add_argument(
        '--npoints', 
        type=int, 
        default=500, 
        help="Number of points for plotting"
    )
    
    args = parser.parse_args()
    main(args)