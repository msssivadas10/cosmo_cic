#!/usr/bin/python3

import numpy as np
from scipy.special import sici # sine and cosine integrals
from typing import Any 
from .._base import HaloDensityProfile

class NFW(HaloDensityProfile):
    r"""
    The NFW halo profile, by Navarro, Frenk and White (1997).
    """
    def A(self, c: Any) -> Any:
        res = np.log( np.add( 1, c ) ) - np.divide( c, np.add( 1, c ) )
        return res
    
    def call(self, 
             arg: Any, 
             c: Any, 
             fourier_transform: bool = False, ) -> Any:
        arg = np.asfarray(arg)
        c   = np.asfarray(c)
        # real space profile 
        if not fourier_transform:
            cx  = c * arg
            res =  ( cx * (1 + cx)**2 )**-1
            return res
        # fourier space profile
        kc, cp1  = arg / c, c + 1
        res      = cp1 * kc
        si1, ci1 = sici(kc)
        si2, ci2 = sici(res)
        res = (si2 - si1) * np.sin(kc) + (ci2 - ci1) * np.cos(kc) - np.sin(arg) / res
        res = res / c**3
        return res

# initialising models to be readily used
nfw = NFW()

_available_models__ = {'nfw': nfw, }

