#!/usr/bin/python3

import numpy as np
from typing import Any

from ..utils.constants import *


class HaloBias:
    r"""
    Base class representing a halo bias function object.
    """

    def f(self, 
          model: object, 
          nu: Any, 
          z: Any, 
          overdensity: Any, ) -> Any:
        r"""
        Calculate the halo bias function values.
        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any, 
                 grid: bool = False, ) -> Any:
        r"""
        Calculate the halo bias function values as function of mass.
        """

        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )

        r  = model.lagrangianR( m, overdensity )
        nu = model.peakHeight(r, z, 
                              grid = False, 
                              exact = not model.settings.useInterpolation ) # delta_c / sigma(r)

        res = self.f( model, nu, z, overdensity )
        return res


#########################################################################################
# Predefined models
#########################################################################################

builtinLinearBiases = {}


class Cole89(HaloBias):
    r"""
    Linear bias model given by Cole & Kaiser (1989) and Mo & White (1996).
    """
    
    def f(self, 
          model: object, 
          nu: Any, 
          z: Any, 
          overdensity: int, ) -> float:
        
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