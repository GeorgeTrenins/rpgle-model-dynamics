#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pi_rates_1D.py
@Time    :   2024/06/13 17:55:25
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import argparse

parser = argparse.ArgumentParser(description="Run a complete Bennett-Chandler rate calculation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-T', type=float, default=300.0, help='Temperature in Kelvin')

parser.add_argument('-a', type=float, required=True, help='A reference value for the system coordinate around the reactant minimum.')
parser.add_argument('-d', type=float, default=0.0, help='System coordinate at the dividing surface.')
parser.add_argument('-N', '--nbead', type=int, default=1, help='Number of ring-polymer beads')
parser.add_argument('-n', '--nrep', type=int, default=1, help='Number of simulation replicas')
parser.add_argument('-P', '--potential', required=True, help="json file with the parameters needed to initialize the external potential")
parser.add_argument('--bath', default=None, help="json file with the parameters needed to initialize a spectral density")
parser.add_argument('--coupling', default=None, help="json file with the parameters needed to initialize a separable coupling")
parser.add_argument('--nvt-propa', help="json file with the parameters needed to initialize the propagator for thermostatted trajectories")
parser.add_argument('--nve-propa', help="json file with the parameters needed to initialize the propagator for unthermostatted trajectories")
parser.add_argument('--seed', default=31415, type=int, help="Integer seed for random number generator" )
parser.add_argument('--config', type=str, default=None, help='name of pkl with an input ring-polymer configuration; if None, all positions set to 0.')

parser.add_argument('--hist', action='store_true', help='Compute the thermal probability density of the system coordinate in the reactant well')
hist = parser.add_argument_group('histogram', description="Settings for calculating the probability distribution at the bottom of the reactant well.")
hist.add_argument('--restraint', default=None, help="json file specifying the harmonic restrain that keeps the system one side of the dividing surface.")
hist.add_argument('--hist-burn', type=str, default='100 fs',  help="Duration of initial equilibration before gathering data for the histogram.")
hist.add_argument('--hist-traj', type=str, default='1 ps', help="Duration of the production trajectory for histogramming system coordinates.")
hist.add_argument('--hist-props', type=str, default='', help="JSON file specifying property output from the production trajectory at the histogramming stage.")

parser.add_argument('--ti', action='store_true', help='Perform thermodynamic integration to compute the free-energy difference.')
ti = parser.add_argument_group('therm-integration', description="Settings for the thermodynamic integration.")
ti.add_argument('--nxi', type=int, default=10, help='Number of quadrature points.')
ti.add_argument('--ti-burn', type=str, default='100 fs',  help="Equilibration for thermodynamic integration.")
ti.add_argument('--ti-traj', type=str, default='1 ps', help="Duration of the production trajectory for single TI point.")
ti.add_argument('--ti-props', type=str, default='', help="JSON file for the output of a TI trajectory.")

parser.add_argument('--fs', action='store_true', help='Calculate the flux-side correlation function.')
fs = parser.add_argument_group('fluxside', description="Settings for the flux-side correlation function.")
fs.add_argument('--fs-burn', type=str, default='100 fs',  help="Equilibration at the top of the barrier")
fs.add_argument('--fs-traj', type=str, default='1 ps', help="Duration of the flux-side correlation function.")
fs.add_argument('--fs-props', type=str, default='', help="JSON file for the output of a flux-side calculation.")
fs.add_argument('--fs-stride', type=str, default=None, help="Stride for recording the flux-side correlation function.")
fs.add_argument('--spawn', default=100, type=int, help="Number of pairs of NVE trajectories to spawn")
fs.add_argument('--nboot', default=100, type=int, help="Number of resamples for bootstrap error estimation for the transmission coefficient.")

def main(args: argparse.Namespace) -> None:
    from rpmdgle.rates import fluxcorr, qtst
    import numpy as np
    x0 = args.a
    xd = args.d
    ss = np.random.SeedSequence(args.seed)
    s0, s1 = ss.spawn(2)
    d = vars(args)
    d['seed'] = s0
    d['x0'] = None
    # potentially a restrained PES - used for histogramming
    T, PES, SB, UNITS = qtst.make_SB(args)
    # fictitious dissipative dynamics for thermal sampling; MF potential included,
    # but no attempt at accurate friction/dissipation
    nvt = qtst.make_propa(args, PES, SB, UNITS, args.nvt_propa)
    d['restraint'] = None
    d['seed'] = s1
    # literal dissipative dynamics for flux-side TCF
    # remove the restraint here
    nve = qtst.make_propa(args, PES, SB, UNITS, args.nve_propa)
    if args.hist:
        print("Histogramming system coordinate in reactant well...")
        qtst.equilibrate(
            UNITS, T, nvt, 
            argparse.Namespace(burn=args.hist_burn, beta_eff=None))
        qtst.production(
            UNITS, T, nvt, argparse.Namespace(
            properties = args.hist_props,
            traj = args.hist_traj,
            x0 = None),
            prefix = 'hist')
    # Check if we are running explicit harmonic-bath dynamics or are using auxiliary variables
    explicit = nvt.PES.extPES is SB
    # switch out the restrained PES in nvt for the unrestrained PES in nve
    nvt.PES = nve.PES
    # Constrain the system centroid for thermodynamic integration
    nvt.fixed = [np.s_[...,0,0]]
    if args.ti:
        from rpmdgle.utils.calculus import GL_quad
        ti_grid, ti_weights = GL_quad(args.nxi, x0, xd)
        np.savetxt('GL_quad.csv', np.c_[ti_grid, ti_weights], fmt='%23.16e')
        print("Performing thermodynamic integration...")
        for i, ti_pos in enumerate(ti_grid):
            qtst.init_pos(nvt, ti_pos)
            qtst.equilibrate(
                UNITS, T, nvt, argparse.Namespace(burn=args.ti_burn, beta_eff=None),
                resample_bath=explicit)
            qtst.production(
                UNITS, T, nvt, argparse.Namespace(
                properties = args.ti_props,
                traj = args.ti_traj),
                prefix = f'ti{i:02d}')
    # Position the system at the dividing surface
    qtst.init_pos(nvt, xd)
    if args.fs:
        print("Calculating the flux-side TCF...")
        qtst.equilibrate(
                UNITS, T, nvt, argparse.Namespace(
                burn=args.fs_burn, beta_eff=None),
            resample_bath=explicit)
        fluxcorr.production_run(
                UNITS, T, nvt, nve, argparse.Namespace(
                stride = args.fs_stride,
                traj = args.fs_traj,
                relax = args.fs_burn,
                properties = args.fs_props,
                nrep = args.nrep,
                spawn = args.spawn,
                nboot = args.nboot,
                x0 = xd,
                progress = False),
            prefix='fs', resample_bath=explicit
        )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)