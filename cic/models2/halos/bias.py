#!/usr/bin/python3

import numpy as np
from typing import Any 
from .._base import HaloBias, Cosmology, DELTA_SC

class Cole89(HaloBias):
    r"""
    Linear bias model given by Cole & Kaiser (1989) and Mo & White (1996).
    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        nu  = np.asfarray( nu )
        z   = np.asfarray( z )
        res = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        return res

class Tinker10(HaloBias):
    r"""
    Linear bias model given by Tinker et al. (2010).
    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        nu = np.asfarray( nu )
        z  = np.asfarray( z )
        y  = np.log10( overdensity )
        A  = 1.0 + 0.24 * y * np.exp( -( 4. / y )**4 )
        a  = 0.44 * y - 0.88
        B  = 0.183
        b  = 1.5
        C  = 0.019 + 0.107 * y + 0.19 * np.exp( -( 4. / y )**4 )
        c  = 2.4
        res = 1.0 - A * nu**a / ( nu**a + DELTA_SC**a ) + B * nu**b + C * nu**c
        return res


# initialising models to be readily used
cole89 = Cole89()
tinker10 = Tinker10()

_available_models__ = {'cole89'  : cole89,
                       'tinker10': tinker10, }

