#!/usr/bin/python3

import numpy as np
from typing import Any
from ._base import HaloBias
from ...utils.constants import DELTA_SC



builtinLinearBiases = {}


class Cole89(HaloBias):
    r"""
    Linear bias model given by Cole & Kaiser (1989) and Mo & White (1996).
    """
    
    def f(self, 
          model: object, 
          nu: Any, 
          z: Any, 
          overdensity: int = None,) -> float:
        
        nu   = np.asfarray( nu )
        z    = np.asfarray( z )
        bias = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        return bias

cole89   = Cole89()
builtinLinearBiases['cole89'] = cole89


class Tinker10(HaloBias):
    r"""
    Linear bias model given by Tinker et al. (2010).
    """
    
    def f(self, 
          model: object, 
          nu: Any, 
          z: Any, 
          overdensity: int, ) -> float:

        nu = np.asfarray( nu )
        z  = np.asfarray( z )
        y  = np.log10( overdensity )
        A  = 1.0 + 0.24 * y * np.exp( -( 4. / y )**4 )
        a  = 0.44 * y - 0.88
        B  = 0.183
        b  = 1.5
        C  = 0.019 + 0.107 * y + 0.19 * np.exp( -( 4. / y )**4 )
        c  = 2.4
        
        bias = 1.0 - A * nu**a / ( nu**a + DELTA_SC**a ) + B * nu**b + C * nu**c
        return bias

tinker10 = Tinker10()
builtinLinearBiases['tinker10'] = tinker10