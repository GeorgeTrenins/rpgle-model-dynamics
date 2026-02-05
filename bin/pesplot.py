#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pesplot.py
@Time    :   2026/02/05 16:49:26
@Author  :   George Trenins
@Desc    :   Visualize one-dimensional potential energy surfaces
'''


from __future__ import print_function, division, absolute_import
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from rpmdgle.system import get_PES


def main(args):
    with open(args.potential, 'r') as f:
        pes_data = json.load(f)
    PES, UNITS = get_PES(pes_data)
    # Determine plotting range
    amplitude = 5.0  # Default amplitude for plotting around the minimum
    if hasattr(PES, 'xgrid'):
        # For splined potentials
        xmin = args.xmin if args.xmin is not None else PES.xgrid.min()
        xmax = args.xmax if args.xmax is not None else PES.xgrid.max()
    elif hasattr(PES, 'shift'):
        # For harmonic or polynomial potentials with a shift
        shift = np.atleast_1d(PES.shift).item()
        xmin = args.xmin if args.xmin is not None else shift - amplitude
        xmax = args.xmax if args.xmax is not None else shift + amplitude
    else:
        # Default range
        xmin = args.xmin if args.xmin is not None else -amplitude
        xmax = args.xmax if args.xmax is not None else amplitude
    
    
    # Generate x values and compute potential
    x = np.linspace(xmin, xmax, args.npoints)
    x_reshaped = x[:, None]  # Shape: (npoints, 1)
    V = PES.potential(x_reshaped)
    potential_span = V.max() - V.min()
    ypad = 0.05 * potential_span
    ymin = args.ymin if args.ymin is not None else V.min() - ypad
    ymax = args.ymax if args.ymax is not None else V.max() + ypad
    
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, V, 'b-', linewidth=2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Potential Energy')
    ax.set_title(f'Potential Energy Surface\nUnits: {UNITS.__class__.__name__}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    # Display interactive window
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a one-dimensional potential energy surface from JSON input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'potential', 
        type=str, 
        help="JSON file with PES parameters"
    )
    parser.add_argument(
        '--xmin', 
        type=float, 
        default=None, 
        help="Minimum x value for plotting"
    )
    parser.add_argument(
        '--xmax', 
        type=float, 
        default=None, 
        help="Maximum x value for plotting"
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
    
    args = parser.parse_args()
    main(args)
