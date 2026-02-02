#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pi_rates_1D_process.py
@Time    :   2024/06/24 16:08:30
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import gvar
from pathlib import Path
from scipy import stats
from typing import Optional
from rpmdgle.rates.qtst import get_PES

parser = argparse.ArgumentParser(description="Companion script to pi_rates_1D.py for processing the output of Bennett-Chandler rate simulations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--show', action='store_true', help='Visualize post-processing results.')
parser.add_argument(
    '-a', required=True, type=float, help="Location of the `reference` point in the reactant well."
)
parser.add_argument(
    '-T', required=True, type=float, help="Simulation temperature in Kelvin."
)
parser.add_argument(
    '-P', '--potential', required=True, type=str, help="JSON input file defining the system potential.")

parser.add_argument(
    '--hist-coords', default='hist_coords.csv', type=str, help="csv file containing the position sampled during the histogramming stage of the simulation."
)
parser.add_argument(
    '--hist-bw', default=None, type=float, help="Bandwidth for KDE estimation of the position distribution function in the reactant well."
)
parser.add_argument(
    '--ti-grid', default='GL_quad_R.csv', type=str, help="Two-column csv file with knots and weights for Gauss-Legendre quadrature."
)
parser.add_argument(
    '--ti-nboot', default=100, type=int, help="Number of bootstrap samples for estimating the error in the free-energy difference."
)
parser.add_argument(
    '--ti-data', nargs='+', default=None, 
    help="Data files with force values, one per GL quadrature point. If not specified, will search for files matchin the string `ti*_forces.csv`")

parser.add_argument(
    '--cfs-data', type=str, default=None, 
    help="Name of CSV file containing the flux-side correlation function output.")

parser.add_argument(
    '--cfs-dt', default=None, type=str, 
    help="Time step, specified in string format as `value unit`, for the transmission coefficient grid.")

product = parser.add_argument_group('product', description="Input arguments for product-side quantities")

product.add_argument(
    '-aP', default=None, type=float, help="Location of the `reference` point in the product well."
)
product.add_argument(
    '--hist-coordsP', default=None, type=str, help="csv file containing the position sampled during the histogramming stage of the simulation."
)
product.add_argument(
    '--ti-gridP', default=None, type=str, help="Two-column csv file with knots and weights for Gauss-Legendre quadrature."
)
product.add_argument(
    '--ti-dataP', nargs='+', default=None, 
    help="Data files with force values, one per GL quadrature point.")


def rename(old: str, new: str) -> None:
    old_file = Path(old)
    new_file = old_file.with_name(new)
    old_file.rename(new_file)

def get_hist(
        datafile: str,
        bw: float,
        a: Optional[float] = None,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        show: Optional[bool] = False,
        ):
    """Estimate the probability density of finding the system at location `a'

    Args:
        datafile (str): name of input file storing the sampled system coordinates
        bw (float): bandwidth for KDE estimation
        a (Optional[float], optional): if not specified, computed at the maximum of the distribution. Defaults to None.
        lo (Optional[float], optional): if specified, data from coordinates below this value are discarded.
        hi (Optional[float], optional): if specified, data from coordinates above this value are discarded.
        show (Optional[bool], optional): plot the histogram. Defaults to False.
    """
    
    from rpmdgle.utils import kde
    from scipy.optimize import minimize_scalar
    
    data = np.loadtxt(datafile)[:,1:]
    if lo is not None:
        data = data[data > lo]
    if hi is not None:
        data = data[data < hi]

    kernel_type = 'epanechnikov'
    bracket = (data.min(), data.max())
    xvec = np.linspace(bracket[0], bracket[1], 250)
    kde_values = kde.kde(data[...,None], xvec, bw, kernel=kernel_type, axis=0)
    kde_mean = np.mean(kde_values, axis=0)
    kde_sem = np.std(kde_values, axis=0)/np.sqrt(len(kde_values))

    if a is None:
        # find the maximum of the distribution
        res = minimize_scalar(
            lambda x: -kde.kde(data, x, bw, kernel=kernel_type, axis=None),
            bounds=bracket,
            method='bounded'
        )
        print(f'Optimization status: {"success" if res.success else "fail"}')
        a = res.x

    P_values = kde.kde(data, a, bw, kernel=kernel_type, axis=0)
    P_mean = np.mean(P_values)
    P_sem = np.std(P_values) / np.sqrt(len(P_values))
    P_gvar = gvar.gvar(P_mean, P_sem)
    print(f"P({a = :.4f}) = ", P_gvar)    
    with open('hist.json', 'w') as f:
        json.dump(dict(a = a, P = P_mean, P_err = P_sem), f, indent=4)

    if show:
        plt.fill_between(xvec, kde_mean, fc="#AAAAFF")
        plt.errorbar(xvec, kde_mean, yerr=2*kde_sem, fmt='none',errorevery=3)
        plt.show()

def parse_forces(input_files: list[str]):
    """Extract the centroid forces and their sampling errors for thermodynamic integration.

    Args:
        input_files (list[str]): list of centroid forces sampled on the quadrature grid.
    """
    data = []
    f0 = []
    sdom = []
    for d in input_files:
        arr = np.loadtxt(d)
        if arr.ndim == 2:
            # forces are time resolved: each column corresponds to an independent simulation
            # (replica) and each row - to a sampling step; the first column is the time (discarded)
            forces = arr[:,1:]
            # time-average each trajectory
            data.append(np.mean(forces, axis=0))
            # estimate the standard error of the mean
            sdom.append(stats.sem(forces.flatten()))       
        else:
            # forces pre-averaged over time to save on space
            data.append(arr) 
            # estimate the standard error of the mean
            sdom.append(stats.sem(arr.flatten()))       
        # average over replicas
        f0.append(np.mean(data[-1]))         
    return np.asarray(data), np.asarray(f0), np.asarray(sdom)


def extend_bootstrap(data, nsample):
    """Given the output of multiple simulations, produce
    a set of `virtual` simulations results drawn, with replacement,
    from the original set

    Args:
        data (nd-array): data[...,j] corresponds to the j-th
        simulation result
        nsample (int): number of `virtual` runs to be simulated

    Returns:
        samples (nd-array): shape = (nsample,) + data.shape
    """
    samples = np.zeros((nsample,)+data.shape)
    nsim = data.shape[-1]
    for i in range(nsample):
        indices = np.random.randint(0, high=nsim, size=nsim)
        samples[i] = np.take(data, indices, axis=-1)
    return samples


def thermodynamic_integration(
        datafiles: list[str],
        quad: str,
        nboot: int,
        potential: str,
        T: float,
        show: Optional[bool] = False
    ):
    """Calculate the free-energy difference and its exponential by thermodynamic integration

    Args:
        datafiles (list[str]): list of centroid forces sampled along the quadrature grid
        quad (str): file with quadrature grid and weights
        nboot (int): number of samples for error estimation by bootstrapping
        potential (str): JSON input file for PES specs (used to fetch units)
        T (float): temperature in Kelvin
        show (Optional[bool], optional): plot the TI curve. Defaults to False.
    """
    
    force_time_avg, force_avg, force_err = parse_forces(datafiles)
    # Estimate error in the free-energy difference by bootstrapping
    means = np.mean(extend_bootstrap(force_time_avg, nboot), axis=-1)
    integrals = []
    x, w = np.loadtxt(quad, unpack=True)
    integral = -np.dot(w, force_avg)
    for mean in means:
        integrals.append(-np.dot(w, mean))
    integral_err = np.std(integrals, ddof=1)
    with open(potential, 'r') as f:
        UNITS = get_PES(json.load(f))[1]
    beta = UNITS.betaTemp(T)
    ratio = np.exp(-beta * integral)
    ratio_rel_err = beta * integral_err
    ratio_err = ratio * ratio_rel_err
    output = dict(
            beta=beta,
            UNITS=UNITS.__class__.__name__,
            dF=integral,
            dF_err=integral_err,
            ratio=ratio,
            ratio_err=ratio_err,
            forces=force_avg.tolist(),
            forces_err=force_err.tolist(), 
            x=x.tolist(),
            w=w.tolist())
    with open('ti.json', 'w') as f:
        json.dump(output, f, indent=4)
    if show:
        plt.errorbar(x, force_avg, 2*np.asarray(force_err))
        plt.show()

def get_kqtst(beta: float, m: float, hist: str, TI: str):
    """Compute the QTST rate from the output of histogramming and TI calculations

    Args:
        beta (float): 1/kB*T 
        m (float): system mass
        hist, TI (str): JSON files with the output of histogramming and TI analysis

    """
    
    kQTST = 1/np.sqrt(2*np.pi*beta*m)
    with open(hist, 'r') as f:
        data = json.load(f)
    prob = data['P']
    kQTST *= prob
    prob_err = data['P_err']/data['P']
    with open(TI, 'r') as f:
        data = json.load(f)
    ratio = data['ratio']
    kQTST *= ratio
    ratio_err = data['ratio_err'] / data['ratio']
    kQTST_err = np.sqrt(prob_err**2 + ratio_err**2)
    return (prob, prob_err, 
            ratio, ratio_err,
            kQTST, kQTST_err)

def process_kqtst(
        potential: str,
        T: float,
        hist: str,
        TI: str,
        histP: Optional[str] = None,
        TIP: Optional[str] = None,
    ):
    from rpmdgle.units import atomic

    with open(potential, 'r') as f:
        PES, UNITS = get_PES(json.load(f))
    beta = UNITS.betaTemp(T)

    AT = atomic()
    t0_at = AT.time
    t0_u = UNITS.time
    to_au = t0_u/t0_at
    to_sec = t0_u

    res = dict()
    # Compute the QTST rate for reactant -> product
    (res['P'], res['P_rel_err'],
     res['ratio'], res['ratio_rel_err'],
     res['kQTST'], res['kQTST_rel_err']) = get_kqtst(
         beta, PES.mass, hist, TI
    )
    res['kQTST_err'] = res['kQTST'] * res['kQTST_rel_err']
    res['UNITS'] = UNITS.__class__.__name__
    kQTST_au = gvar.gvar(res['kQTST']/to_au, res['kQTST_err']/to_au)
    kQTST_SI = gvar.gvar(res['kQTST']/to_sec, res['kQTST_err']/to_sec)

    print('kQTST =', kQTST_au,'[a.u.] =', kQTST_SI, '[s⁻¹]')

    if histP is not None:
        # Compute the QTST rate for product -> reactant
        (res['PP'], res['PP_rel_err'],
        res['ratioP'], res['ratioP_rel_err'],
        res['kQTSTP'], res['kQTSTP_rel_err']) = get_kqtst(
            beta, PES.mass, histP, TIP
        )
        res['kQTSTP_err'] = res['kQTSTP'] * res['kQTSTP_rel_err']
        kQTSTP_au = gvar.gvar(res['kQTSTP']/to_au, res['kQTSTP_err']/to_au)
        kQTSTP_SI = gvar.gvar(res['kQTSTP']/to_sec, res['kQTSTP_err']/to_sec)

        print('kQTST(prod) =', kQTSTP_au,'[a.u.] =', kQTSTP_SI, '[s⁻¹]')

    with open('kQTST.json', 'w') as f:
        json.dump(res, f, indent=4)

def fancy_flux(dt, kappa, Rqtst, Pqtst):
    from scipy.integrate import cumulative_trapezoid
    css = cumulative_trapezoid(kappa, dx=dt, initial=0.0)
    css *= (Rqtst + Pqtst)
    return Rqtst*kappa/(1-css)

def get_rate(
    potential: str,
    ktst: str,
    kappa: str,
    dt: str,
    show: Optional[bool] = False
    ):
    """Compute the full RPMD rate

    Args:
        potential (str): JSON file with PES specs
        ktst (str): JSON file with results of QTST rate analysis
        kappa (str): CSV file with transmission coefficient data
        dt (str): transmission coefficient time-step in the 'value unit' format
        show (Optional[bool], optional): plot the flux-side correlation function. Defaults to False.
    """

    with open(potential, 'r') as f:
        UNITS = get_PES(json.load(f))[1]

    t0_u = UNITS.time
    to_sec = t0_u

    with open(ktst, 'r') as f:
        ktst_data = json.load(f)

    tvec, kappa, kappa_err = np.loadtxt(kappa, usecols=(0,3,4), unpack=True)
    
    kt = ktst_data['kQTST']*kappa
    kt_err = kt * np.sqrt(
        (kappa_err/kappa)**2 + ktst_data['kQTST_rel_err']**2)
    if 'kQTSTP' in ktst_data:
        # TODO: figure out error estimation in this case
        dt = UNITS.str2base(dt)
        kt_rev = fancy_flux(dt, kappa, ktst_data['kQTST'], ktst_data['kQTSTP'])
        np.savetxt("rate_constant.csv", 
                    np.c_[tvec, kt/to_sec, kt_err/to_sec, kt_rev/to_sec])
    else:
        np.savetxt("rate_constant.csv", np.c_[tvec, kt/to_sec, kt_err/to_sec])
    

    if show:
        fig, ax = plt.subplots()
        ax.plot(tvec, kt/to_sec, 'C0-', label='standard')
        ax.fill_between(tvec, (kt-2*kt_err)/to_sec, (kt+2*kt_err)/to_sec, color='C0', alpha=.1)
        ax.plot(tvec, kt_rev/to_sec, 'C1-', label='revised')
        ax.legend()
        ax.set_xlabel('t [fs]')
        ax.set_ylabel('k(t) [s⁻¹]')
        plt.show()



def main(args: argparse.Namespace) -> None:

    none_test = [getattr(args, elem) is None for elem in ['aP', 'hist_coordsP', 'ti_gridP', 'ti_dataP']]

    if np.all(none_test):
        product = False
    elif np.any(none_test):
        raise RuntimeError("Insufficient information to perform product-side analysis!")
    else:
        product = True

    get_hist(args.hist_coords, args.hist_bw, a=args.a, show=args.show)
        
    if product:
        rename('hist.json', 'hist_R.json')
        get_hist(args.hist_coordsP, args.hist_bw, a=args.aP, show=args.show)
        rename('hist.json', 'hist_P.json')

    if args.ti_data is None:
        import glob
        ti_data: list[str] = glob.glob('ti*_forces.csv')
        ti_data.sort()
    else:
        ti_data = args.ti_data

    thermodynamic_integration(
        ti_data, 
        args.ti_grid, 
        args.ti_nboot, 
        args.potential, 
        args.T, 
        show=args.show)

    if product:
        rename('ti.json', 'ti_R.json')
        thermodynamic_integration(
            args.ti_dataP, 
            args.ti_gridP, 
            args.ti_nboot, 
            args.potential, 
            args.T, 
            show=args.show)
        rename('ti.json', 'ti_P.json')

    if product:
        process_kqtst(
            args.potential, 
            args.T, 
            "hist_R.json",
            "ti_R.json",
            histP="hist_P.json",
            TIP="ti_P.json"
        )
    else:
        process_kqtst(
            args.potential, 
            args.T, 
            "hist.json",
            "ti.json"
        )

    if args.cfs_data is not None:
        get_rate(
            args.potential,
            'kQTST.json',
            args.cfs_data, 
            args.cfs_dt,
            show=args.show
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
