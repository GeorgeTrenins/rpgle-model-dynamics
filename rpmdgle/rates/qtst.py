#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/12/14 14:45:22
@Author  :   George Trenins
@Desc    :   None
'''



from __future__ import print_function, division, absolute_import
from rpmdgle.myargparse import MyArgumentParser
from rpmdgle.sysbath import spectral, coupling
from rpmdgle import propagators
from rpmdgle.pes.pi import Ring, RestrainedRing
import importlib
import numpy as np
import json
import pickle

parser = MyArgumentParser(description="Calculate the RPMD QTST rate according to Collepardo-Guevara, Craig, and Manolopoulos https://doi.org/10.1063/1.2883593")

# Create mutually exclusive group for temperature specification
group = parser.add_mutually_exclusive_group()
group.add_argument('-T', type=float, help="Temperature in Kelvin.")
group.add_argument('--beta', type=float, help="Reciprocal temperature in internal units")
#
parser.add_argument('-P', '--potential', required=True, help="json file with the parameters needed to initialize the external potential")
parser.add_argument('--bath', default=None, help="json file with the parameters needed to initialize a spectral density")
parser.add_argument('--coupling', default=None, help="json file with the parameters needed to initialize a separable coupling")
parser.add_argument('--nrep', type=int, default=1, help="Number of independent system replicas to run in parallel.")
parser.add_argument('--nbead', default=1, type=int, help="Number of beads in ring-polymer discretisation.")
parser.add_argument('--config', type=str, default=None, help='name of pkl with an input ring-polymer configuration; if None, all positions set to 0.')
parser.add_argument('--x0', default=None, type=float, help='shift the centroid to this location')
parser.add_argument('--fix_centroid', action='store_true', help='Fix the centroid of the system ring-polymer.')
parser.add_argument('--seed', default=31415, type=int, help="Integer seed for the random number generator" )
parser.add_argument('--properties', type=str, default='', help="JSON file specifying property output.")
md_group = parser.add_argument_group('MD', 'settings for sampling the thermal distribution with MD')
md_group.add_argument('--burn', type=str, default='100 fs',  help="Duration of initial equilibration.")
md_group.add_argument('--traj', type=str, default='1 ps', help="Duration of a production trajectory.")
md_group.add_argument('--propa', help="json file with the parameters needed to initialize the propagator")
md_group.add_argument('--restraint', default=None, help="json file specifying the parameters of a harmonic restrain aimed at keeping the system one side of the dividing surface.")


def get_PES(kwargs):
    """fetch the classical external potential"""
    pesmod = importlib.import_module(kwargs.pop("module"))
    pesname = kwargs.pop("name")
    PES = getattr(pesmod, pesname)(**kwargs)
    return PES, PES.UNITS

def get_bath(PES, bath_data, F_data):
    """construct the Caldeira-Leggett model of the dissipative system
    given the classical external potential.
    """
    if bath_data is None:
        return PES
    if F_data is None:
        F = coupling.linear.Coupling(UNITS=PES.UNITS.__class__.__name__)
    else:
        gname = F_data.pop("name")
        F = getattr(coupling, gname).Coupling(**F_data)
    Jname = bath_data.pop("name")
    Nmodes = bath_data.pop("Nmodes")
    SB = getattr(spectral, Jname).Density(PES, Nmodes, coupling=F, **bath_data)
    return SB

def make_SB(args):
    """Initialise the classical external potential and the harmonic-bath (Caldeira--Leggett) representation of the external potential coupled to a dissipative environment.
    """
    
    F_data = None
    bath_data = None
    with open(args.potential) as f:
        pes_data = json.load(f)
    # External potential
    PES, UNITS = get_PES(pes_data)
    if args.T is None:
        d = vars(args)
        d['T'] = UNITS.betaTemp(args.beta)
    T = args.T
    if getattr(args, "bath", None) is not None:
        with open(args.bath) as f:
            # Specification of the position-independent part of the spectral density.
            bath_data = json.load(f)
        if args.coupling is not None:
            # Specification of the position-dependent part.
            with open(args.coupling) as g:
                F_data = json.load(g)
    # Caldeira--Leggett model and interaction potential
    SB = get_bath(PES, bath_data, F_data)
    return T, PES, SB, UNITS

def make_RP(nrep, nbead, xshape, T, PES, UNITS, restraint=None):
    """Create a ring-polymerised PES.

    Args:
        nrep (int): number of independent system realisations
        nbead (int): number of beads
        xshape (tuple): shape of the configuration array passed to the classial PES
        T (float): temperature in Kelvin
        PES (object): classical potential energy surface
        UNITS (object): unit system
        restraint (str): input file for specifying a restraint on the system coordinate
    """
    
    beta = UNITS.betaTemp(T)
    if restraint is None:
        rpPES = Ring(nbead, xshape, PES, beta)
    else:
        with open(restraint, 'r') as f:
            data = json.load(f)
        rpPES = RestrainedRing(
            data['shift'], data['k'], data['direction'],
            nbead, xshape, PES, beta
        )
    x = np.zeros((nrep, nbead)+xshape)
    return x, beta, rpPES

def init_pos(propa, x0):
    """Set the ring-polymer centroid to x0
    """
    xnm = np.copy(propa.xnm)
    xnm[...,0,:1] = x0
    propa.set_xnm(xnm)

def make_propa(args, PES, SB, UNITS, propa_json, fix_centroid=False):
    """Initialize the propagator

    Args:
        args (Namespace): command-line arguments (see parser)
        PES (object): external potential energy surface
        SB (object): PES coupled to a harmonic bath
        UNITS (rpmdgle.SI): unit system
        propa_json (str): JSON initialization file for the propagator
        fix_centroid (bool, optional): constrain the system centroids. 
        Defaults to False.
    """
    T = args.T
    nbath = getattr(SB, "Nmodes", 0)
    with open(propa_json, 'r') as f:
        kwarg_dict = json.load(f)
    try:
        propa_class = kwarg_dict.pop('class')
    except:
        print(f'{propa_json = }, {kwarg_dict = }')
        raise
    dt = kwarg_dict.pop('dt', '0.25 fs')
    if fix_centroid:
        kwarg_dict["fixed"] = [(slice(None), slice(1), slice(1))]
    if propa_class in {"SepGLEaux", "SepGLEPILE"}:
        # system-only potential
        x, beta, rpPES = make_RP(args.nrep, args.nbead, (1,), T, PES, UNITS, args.restraint)
        # implicit friction
        propa = getattr(propagators, propa_class)(rpPES, SB, dt, x.shape, rng=args.seed, beta=beta, **kwarg_dict)
    else:
        # full Caldeira-Leggett
        x, beta, rpPES = make_RP(args.nrep, args.nbead, (1+nbath,), T, SB, UNITS, args.restraint)
        # friction represented explicitly by harmonic bath modes
        propa = getattr(propagators, propa_class)(rpPES, dt, x.shape, rng=args.seed, beta=beta, **kwarg_dict)
    print(f"RNG seed: {args.seed}")
    if args.config is not None:
        # Set the ring-polymer configuration
        with open(args.config, 'rb') as f:
            config = pickle.load(f)
        x = config['x']
    propa.set_x(x)
    if args.x0 is not None:
        # Shift the centroids to x0
        init_pos(propa, args.x0)
    return propa

def make_property_tracker(args, propa, prefix = None):
    """Track and output some property along the RPMD trajectory

    Args:
        args (argparse.Namespace): command-line args, see parser.
        propa (Any): propagator
        prefix (Optional[str]): string prepended to all output file names. Defaults to None.
    """
    traj: str = args.traj # duration of the sampling trajectory
    value_unit = traj.split()
    if len(value_unit) == 1:
        # use internal units
        dt = propa.dt
        time_unit = None
    elif len(value_unit) == 2:
        # custom units
        u = propa.UNITS
        time_unit = value_unit[1]
        dt = propa.dt * u.time / u.str2SI(f"1 {time_unit}")
    else:
        raise RuntimeError(f"Expecting the variable 'traj' to be specified in the format 'value' or 'value unit'; instead got {' '.join(value_unit)}")
    if args.properties == '':
        proprec = None
    else:
        from rpmdgle.properties import PropertyTracker
        proprec = PropertyTracker(
            propa, args.properties, tunit=time_unit, prefix=prefix)
    return dt, time_unit, proprec

def equilibrate(UNITS, T, propa, args, resample_bath=False):
    """Equilibrate the system (no properties written at this stage)

    Args:
        UNITS (object): unit system
        T (float): temperature in Kelvin
        propa (object): propagator
        args (Namespace): command-line arguments (see parser)
        resample_bath (bool, optional): resample the bath positions and momenta from the thermal distribution, conditional on the system coordinates. Defaults to False.
    """
    nequil = int(np.ceil(UNITS.str2base(args.burn)/propa.dt)) 
    beta = UNITS.betaTemp(T)
    propa.set_pnm(propa.psample(beta))
    if resample_bath and ( getattr(args, "bath", None) is not None ):
        sample_bath_modes(propa, beta)
    n_info = 10
    infostride = max(1,nequil//(n_info-1))
    print("Equilibration", end="")
    for i in range(nequil):
        propa.step()
        if (i+1)%infostride == 0:
            print(".", end='')
    print(".")
    print('Done.')    

def production(UNITS, T, propa, args, prefix=None):
    """Run the sampling trajectory

    Args:
        UNITS (object): unit system
        T (float): temperature in Kelvin
        propa (object): propagator
        args (Namespace): command-line arguments (see parser)
        prefix (str, optional): prefix appended to all property-output files. Defaults to None.
    """
    print("Production trajectory")
    proprec = make_property_tracker(args, propa, prefix=prefix)[2]
    ntraj = int(np.ceil(UNITS.str2base(args.traj)/propa.dt))
    if ntraj <= 100:
        infostride=1
    else:
        infostride=10**int(np.floor(np.log10(ntraj//100)))
    for i in range(0,ntraj):
        propa.step()
        if proprec is not None:
            proprec.update(i+1)
        if i%infostride == 0:
            print(f"Step {i} of {ntraj}")
    print('Done.')
    print()

def sample_bath_modes(propa, beta):
    nrep, nbead, ncoord = propa.xshape
    nbath = ncoord - 1
    if nbath != propa.PES.extPES.Nmodes:
        raise RuntimeError(f"Unexpected array shape, final dimension should be {propa.PES.extPES.Nmodes+1}, instead found {ncoord}")
    bath_freqs2 = propa.PES.extPES.ww
    bath_masses = propa.PES.extPES.mass[...,1:]
    rp_freqs = propa.freqs
    combined_freqs2 = rp_freqs[:,None]**2 + bath_freqs2[None,:]
    combined_freqs = np.sqrt(combined_freqs2)
    x_system = propa.x[...,:1].copy()
    # get equilibrium positions
    bath_eq = propa.PES.extPES.bath_eq(x_system) # c[nu]*F / m[nu] omega[nu]^2
    nm_bath_eq = propa.nmtrans.cart2mats(bath_eq) * bath_freqs2[None,:] / combined_freqs2
    # sample the bath modes conditional on the system coordinates
    new_bath_nm = propa.rng.normal(size = (nrep, nbead, nbath)) / (
        np.sqrt(beta * bath_masses) * combined_freqs) + nm_bath_eq
    xnm = propa.xnm.copy()
    xnm[...,1:] = new_bath_nm
    propa.set_xnm(xnm)


def main(args):
    T, PES, SB, UNITS = make_SB(args)
    propa = make_propa(
        args, PES, SB, UNITS, args.propa, fix_centroid=args.fix_centroid)    
    equilibrate(UNITS, T, propa, args)
    production(UNITS, T, propa, args)
    with open("final.pkl", 'wb') as f:
        pickle.dump(dict(x = propa.x.copy()), f)

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)