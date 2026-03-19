#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fluxcorr.py
@Time    :   2023/12/15 16:58:33
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import numpy as np
import json
import tqdm
import pickle
import argparse
from rpmdgle.rates import qtst
from rpmdgle.units import SI
from typing import Any


def setup_propagators(args):
    d = vars(args)
    seed = d['seed']
    ss = np.random.SeedSequence(seed)
    idum0, idum1 = ss.spawn(2)
    T, PES, SB, UNITS = qtst.make_SB(args)
    d['seed'] = idum0
    propa1 = qtst.make_propa(
        args, PES, SB, UNITS, args.propa, fix_centroid=True)
    d['seed'] = idum1
    propa2 = qtst.make_propa(
        args, PES, SB, UNITS, args.propa2, fix_centroid=False)
    return propa1, propa2, T, UNITS


def output_cfs(tcf, vel_h, nsample, nboot, nrep, rng, tvec):
    from scipy import stats
    tcf = tcf / nsample
    vel_h = vel_h / nsample
    with open('tcfs.pkl', 'wb') as f:
        pickle.dump(dict(tcf=tcf, vel_h=vel_h), f)
    avg_tcf = np.mean(tcf[1:], axis=-1)
    sem_tcf = stats.sem(tcf[1:], axis=-1)
    avg_vel = np.mean(vel_h)
    sem_vel = stats.sem(vel_h)
    kappa = avg_tcf/avg_vel
    # Revise the error estimate
    kappa_resample = []
    for _ in range(nboot):
        idx = rng.choice(nrep, nrep, replace=True)
        tcf_resample = tcf[...,idx]
        vel_resample = vel_h[...,idx]
        kappa_resample.append(np.mean(tcf_resample[1:], axis=-1)/
                              np.mean(vel_resample))
    kappa_sem = np.std(kappa_resample, axis=0)
    np.savetxt(
        'c_fs.csv',
        np.c_[
            tvec, avg_tcf, sem_tcf, kappa, kappa_sem], 
        fmt="%23.16e",
        header=f'{dict(v=avg_vel, err=sem_vel).__repr__()}')
    with open('vel_h.json', 'w') as f:
        json.dump(dict(v=avg_vel, err=sem_vel, nsample=nsample, nrep=nrep), f)

def production_run(
        UNITS: SI, 
        T: float, 
        propa1: Any, 
        propa2: Any,
        args: argparse.Namespace,
        resample_bath = False,
        prefix = None): 
    
    from rpmdgle.rates.qtst import make_property_tracker, sample_bath_modes
    # timestep in the units of the stride
    stride = propa2.dt if args.stride is None else args.stride
    nrelax = int(np.ceil(UNITS.str2base(args.relax)/propa1.dt)) 
    ntraj, nstride = [
        int(np.ceil(UNITS.str2base(time)/propa2.dt)) 
        for time in (args.traj, stride)]
    dt, _, proprec = make_property_tracker(args, propa2, prefix=prefix)
    x0 = np.zeros((1+ntraj//nstride,args.nrep))
    vel_h = np.zeros_like(x0[0])
    tcf = np.zeros((len(x0), args.nrep))
    nsample = 0
    tvec = dt*nstride*np.arange(len(tcf))[1:]
    if args.spawn <= 100:
        infostride=1
    else:
        infostride=10**(np.log10(args.spawn)//2)
    beta = UNITS.betaTemp(T)
    for i in range(args.spawn):
        if i%infostride == 0:
            print(f"Trajectory pair {i+1} of {args.spawn}")
            if i > 0:
                output_cfs(tcf, vel_h, nsample, args.nboot, args.nrep, propa2.rng, tvec)
        # relax
        propa1.set_pnm(propa1.psample(beta))
        if resample_bath:
            sample_bath_modes(propa1, beta)
        propa1.step(nrelax)
        x = propa1.x.copy()
        # NVE
        p = propa2.psample(beta)
        for _ in range(2):
            p *= -1
            propa2.set_pnm(p.copy())
            propa2.set_x(x.copy())
            v0 = propa2.pnm[...,0,0]/propa2.m3[...,0,0]
            x0[0] = propa2.xnm[...,0,0] - args.x0
            counter = range(ntraj)
            if args.progress:
                counter = tqdm.tqdm(counter)
            for i in counter:
                propa2.step()
                q, r = divmod(i+1, nstride)
                if r == 0:
                    x0[q] = propa2.xnm[...,0,0] - args.x0
                if proprec is not None:
                    proprec.update(i+1)
            tcf += v0 * (x0 > 0).astype(int)
            vel_h += v0 * (v0 > 0).astype(int)
            nsample += 1
    output_cfs(tcf, vel_h, nsample, args.nboot, args.nrep, propa2.rng, tvec)


def main():
    from rpmdgle.myargparse import MyArgumentParser
    from rpmdgle.rates.qtst import equilibrate

    parser = MyArgumentParser(description="Calculate the RPMD flux-side correlation function according to Collepardo-Guevara, Craig, and Manolopoulos https://doi.org/10.1063/1.2883593")

    # Create mutually exclusive group for temperature specification
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument('-T', type=float, help="Temperature in Kelvin.")
    temp_group.add_argument('--beta', type=float, help="Reciprocal temperature in internal units")
    #
    parser.add_argument('-P', '--potential', required=True, help="json file with the parameters needed to initialize the external potential")
    parser.add_argument('--bath', default=None, help="json file with the parameters needed to initialize a spectral density")
    parser.add_argument('--coupling', default=None, help="json file with the parameters needed to initialize a separable coupling")
    parser.add_argument('--nrep', type=int, default=1, help="Number of independent system replicas to run in parallel.")
    parser.add_argument('--nbead', default=1, type=int, help="Number of beads in ring-polymer discretisation.")
    parser.add_argument('--config', type=str, default=None, help='name of pkl with an input ring-polymer configuration; if None, all positions set to 0.')
    parser.add_argument('--x0', required=True, type=float, help='location of the dividing surface')
    parser.add_argument('--seed', default=31415, type=int, help="Integer seed for the random number generator" )
    parser.add_argument('--properties', type=str, default='', help="JSON file specifying which extra properties to record during the NVE trajectories.")
    parser.add_argument('--MC', action='store_true', help="Use Monte Carlo sampling to initialise the configuration of harmonic bath modes. Only relevant if using a explicit harmonic bath.")
    md_group = parser.add_argument_group('MD', 'molecular dynamics settings for sampling the thermal distribution and running NVE trajectories')
    md_group.add_argument('--propa', help="json file with the parameters needed to initialize the propagator for thermal sampling at the barrier top.")
    md_group.add_argument('--propa2', required=True, help="json file with the parameters to initialize the NVE propagator of the 'unpinned' system.")
    md_group.add_argument('--burn', type=str, default='100 fs',  help="Duration of initial equilibration.")
    md_group.add_argument('--relax', default='100 fs', type=str, help="Relaxation time between spawning NVE trajectories")
    md_group.add_argument('--traj', type=str, default='1 ps', help="Duration of a production trajectory.")
    md_group.add_argument('--stride', type=str, default=None, help="Stride for recording the flux-side correlation function.")
    md_group.add_argument('--spawn', default=100, type=int, help="Number of pairs of NVE trajectories to spawn")

    parser.add_argument('--progress', action='store_true', help="Display progress bar for NVE trajectories.")
    parser.add_argument('--nboot', default=100, type=int, help="Number of resamples for bootstrap error estimation of the transmission coefficient.")

    args = parser.parse_args()
    propa1, propa2, T, UNITS = setup_propagators(args)
    equilibrate(UNITS, T, propa1, args, resample_bath=args.MC)
    production_run(UNITS, T, propa1, propa2, args, resample_bath=args.MC)
    print('Done.')
    print()

if __name__ == "__main__":
    main()
