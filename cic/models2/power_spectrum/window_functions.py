#!/usr/bin/python3

import numpy as np
from typing import Any 
from .._base import WindowFunction

class SphericalTophat(WindowFunction):
    r"""
    Spherical top-hat filter.
    """
    def __init__(self, disable_masks: bool = False) -> None:
        super().__init__()
        # mask at zero and inifinity ends to make the function smooth
        self.disableMasks = disable_masks 

    def call(self, x: Any, deriv: int = 0) -> Any:
        # sherical top-hat function
        def f1(x): return 3.*( np.sin(x) - x * np.cos(x) ) / x**3
        # first derivative 
        def f2(x): return 3.*( ( x**2 - 3. ) * np.sin(x)  + 3.*x * np.cos(x) ) / x**4
        # second derivative
        def f3(x): return 3.*( ( x**2 - 12. ) * x * np.cos(x)  - ( 5*x**2 - 12. ) * np.sin(x) ) / x**5
        
        x = np.asfarray( x )
        if self.disableMasks:
            if deriv == 0: return f1(x)
            if deriv == 1: return f2(x)
            if deriv == 2: return f3(x)
        else:
            # mask for x closer to 0
            closeToZero = ( np.abs(x) < 1e-03 )
            # mask for other smallr x; at both ends, the function values is almost 0
            otherValues = np.logical_not( closeToZero | ( np.abs(x) > 1e+04 ) )
            res = np.zeros_like( x )
            if deriv == 0:
                res[closeToZero] = 1.0
                res[otherValues] = f1( x[otherValues] )
                return res
            if deriv == 1:
                res[otherValues] = f2( x[otherValues] )
                return res
            if deriv == 2:
                res[closeToZero] = -0.2
                res[otherValues] = f3( x[otherValues] )
                return res
        raise ValueError(f"invalid value for argument 'deriv': {deriv}")
    
class Gaussian(WindowFunction):
    r"""
    Gaussian function.
    """
    def call(self, x: Any, deriv: int = 0) -> Any:
        x = np.asfarray( x )
        if deriv == 0:
            return np.exp( -0.5*x**2 )
        if deriv == 1:
            return -x * np.exp( -0.5*x**2 )
        if deriv == 2:
            return ( x**2 - 1 ) * np.exp( -0.5*x**2 )
        raise ValueError(f"invalid value for argument 'deriv': {deriv}")


# initialising models to be readily used
tophat   = SphericalTophat()
gaussian = Gaussian()

def _init_module() -> None:
    WindowFunction.available.add('tophat', tophat)
    WindowFunction.available.add('gaussian', gaussian)
    return