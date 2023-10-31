#!/usr/bin/python3

import numpy as np
from scipy.special import sici
from typing import Any 
from ._base import DensityProfile
from .cmrelations import bullock01


builtinProfiles = {}

class NFW(DensityProfile):
    r"""
    An NFW halo profile.
    """

    def A(self, c: Any) -> Any: 
        return np.log( np.add( 1, c ) ) - np.divide( c, np.add( 1, c ) )
    
    def f(self, x: Any, c: Any) -> Any: 
        cx  = np.multiply(c, x)
        res =  ( cx * (1 + cx)**2 )**-1
        return res
    
    def u(self, q: Any, c: Any) -> Any:
        qc, cp1  = np.divide(q, c), np.add(1, c)
        res      = cp1 * qc
        si1, ci1 = sici( qc )
        si2, ci2 = sici( res )
        res      = (si2 - si1) * np.sin( qc ) + (ci2 - ci1) * np.cos( qc ) - np.sin(q) / res
        return res / np.power(c, 3.)
           
nfw = NFW( bullock01 )
builtinProfiles['nfw'] = nfw
