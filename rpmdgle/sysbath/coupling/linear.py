#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   linear.py
@Time    :   2024/10/02 11:10:12
@Author  :   George Trenins
@Desc    :   Linear interaction potential for modelling position-independent friction
'''


from __future__ import print_function, division, absolute_import
from rpmdgle.pes._base import BasePES, check_onedim
from rpmdgle import units
import numpy as np
from typing import Optional, Union


class Coupling(BasePES):

    def __init__(self, UNITS: Optional[str] = "atomic"):
        self.UNITS = getattr(units, UNITS)()

    @check_onedim
    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        ans = np.copy(x)
        if x.ndim > 1:
            return np.reshape(ans, x.shape[:-1])
        else:
            return ans.item()
        
    @check_onedim
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
