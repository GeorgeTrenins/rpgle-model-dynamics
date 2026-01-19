#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   properties.py
@Time    :   2024/04/05 11:50:36
@Author  :   George Trenins
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.utils.arrays import idx_along_axis
from typing import Union, Any, Optional, Callable
import pickle
import os
import pandas as pd
import json
import numpy as np


class Property(object):
    def __init__(self, 
                 propa : Any, 
                 id : str,
                 post : Optional[Callable[[np.ndarray],np.ndarray]] = None):
        """Object for fetching some property along the RPMD trajectory

        Args:
            propa (Any): RPMD propagator
            id (str): string descriptor of the property; one of {
                      'xnm','pnm','fnm', 'x', 'p', 'f', 
                      'xnm0','pnm0','fnm0',
                      'vnm' ,'vnm0', 'V', 'potential', 'Vext', 'VRP', 
                      'KE', 'kinetic', 'econs', 'conserved'}
            post (Optional[Callable[[np.ndarray],np.ndarray]], optional): post-processing function applied to the raw property (e.g., slicing). Defaults to None.
        """
        self.propa = propa
        try:
            # Axis along which the beads/normal modes are indexed
            self.axis = propa.nmtrans.axis
        except AttributeError:
            # the object is not ring-polymerized
            self.axis = None
        id = id.strip()
        if id in {'xnm','pnm','fnm', 'x', 'p', 'f'}:
            _fun = lambda : getattr(self.propa, id).copy()
        elif id in {'xnm0','pnm0','fnm0'}:
            if self.axis is None:
                raise RuntimeError("You are requesting a centroid from an object that has not been ring-polymerized. Aborting...")
            _fun = lambda : idx_along_axis(
                getattr(self.propa, id[:-1]), self.axis, 0).copy()
        elif id == 'vnm':
            _fun = lambda : self.propa.pnm/self.propa.m3
        elif id == 'vnm0':
            if self.axis is None:
                raise RuntimeError("You are requesting a centroid from an object that has not been ring-polymerized. Aborting...")
            _fun = lambda : (
                idx_along_axis( self.propa.pnm, self.axis, 0) / 
                idx_along_axis( self.propa.m3, self.axis, 0))
        elif id in {'V', 'potential'}:
            _fun = lambda : self.propa.Vext.copy()
        elif id == 'Vext':
            _fun = lambda : self.propa.Vext.copy()
        elif id == 'VRP':
            _fun = lambda : self.propa.VRP.copy()
        elif id in {'KE', 'kinetic'}:
            _fun = lambda : self.propa.kinetic_energy()
        elif id in {'econs', 'conserved'}:
            _fun = lambda : self.propa.econs()
        else:
            raise NotImplementedError(f"Unknown property {id}")
        if post is None:
            self._fun = _fun
        else:
            self._fun = lambda : post(_fun())
        
    def __call__(self):
        return self._fun()
    
    
class PropertyWriter(Property):

    def __init__(self, 
                 propa : Any, 
                 id : str,
                 name : str,
                 stride : int,
                 dt : float,
                 post : Optional[Callable[[np.ndarray],np.ndarray]] = None,
                 ):
        """Object for fetching some property along the RPMD trajectory and appending it to file.

        Args:
            propa (Any): RPMD propagator
            id (str): string descriptor of the property; one of {
                'xnm','pnm','fnm', 'x', 'p', 'f', 
                'xnm0','pnm0','fnm0',
                'vnm' ,'vnm0', 'V', 'potential', 'Vext', 'VRP', 
                'KE', 'kinetic', 'econs', 'conserved'}
            name (str): name of the output file
            stride (int): stride, in multiples of the propagation time-step, for data output
            dt (float): propagation time-step converted to output units 
            post (Optional[Callable[[np.ndarray],np.ndarray]], optional): post-processing function
                pplied to the raw property (e.g., slicing). Defaults to None.
        """
        
        super().__init__(propa, id, post=post)
        self.name = name
        self.stride = stride
        self.dt = dt
        self.time = 0
        for ext in ['csv', 'pkl']:
            if os.path.exists(name := f'{self.name}.{ext}'):
                # raise RuntimeError(f"The output file {name} already exists!")
                open(name, 'w').close()

    def update(self, istep: int):
        """Update internal time counter and append property to output file as needed.

        Args:
            istep (int): index of propagation step
        """
        self.time += self.dt
        if istep % self.stride == 0:
            ans = self.__call__()
            data = dict(t=self.time, property=ans)
            with open(f'{self.name}.pkl', 'ab') as f:
                pickle.dump(data, f)
            # If the data is sufficiently flat, output as csv 
            ans = np.squeeze(ans)
            fmt = "%14.7e"
            if (n := ans.ndim) in {0, 1}:
                if n == 0:
                    df = pd.DataFrame(
                        {'Time' : [self.time],
                         'Property' : [ans.item()]})
                else:
                    df = pd.DataFrame(
                        index = [self.time],
                        data=np.reshape(ans, (1,-1)))
                    df.reset_index(inplace=True, names='Time')
                df.to_csv(f'{self.name}.csv',
                          mode='a',
                          header=False,
                          index=False,
                          sep="\t",
                          float_format=fmt)


class PropertyTracker(object):

    def __init__(self, 
                 propa : Any,
                 specs : Union[ list[dict], dict, str],
                 tunit : Optional[str] = None,
                 prefix : Optional[str] = None) -> None:
        
        """Object for managing output of all properties along an RPMD trajectory.

        Args:
            propa (Any): MD propagator
            specs (Union[ list[dict], dict, str]): 
                dictionary, or list of dictionaries, or name of JSON file containing 
                a dict / list[dict] with specifications of the properties to be recorded
                during the simulation
            tunit (Optional[str]): unit of time used in the output. Defaults to the internal unit of time of the propagator.
            prefix (Optional[str]): string prepended to the output file names given in `specs`
        """
        
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except json.JSONDecodeError:
                with open(specs, 'r') as f:
                    specs = json.load(f)
        if isinstance(specs, dict):
            specs = [specs]

        self.propa = propa
        self.prefix = prefix
        
        self.dt = propa.dt
        u = propa.UNITS
        if tunit:
            self.dt *= u.time / u.str2SI(f"1.0 {tunit}")
        self.make_entries(specs)

    def update(self, istep : int):
        for entry in self.entries.values():
            entry.update(istep)

    def make_entries(self, specs : list[dict]):
        self.entries = dict()
        names = []
        for d in specs:
            entry, name = self.parse_dict(self.propa, d, prefix=self.prefix)
            names.append(name)
            self.entries.update({name : entry})
        if len(set(names)) != len(names):
            raise RuntimeError("Please supply a unique file name for every property file.")
        
    def parse_dict(
            self, propa : Any, 
            d : list[dict],
            prefix: Optional[str] = None):
        stride = d.get('stride')
        if stride is None:
            stride = 1
        else:
            stride = int(np.ceil(propa.UNITS.str2base(stride)/propa.dt)) 
        postA = d.get('postA')
        if postA is not None:
            postA = eval(postA)
        if prefix:
            name = '_'.join([prefix, d['name']])
        else:
            name = d['name']
        return PropertyWriter(propa, d['A'], name, stride, self.dt, post=postA), name