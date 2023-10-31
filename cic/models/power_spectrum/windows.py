#!/usr/bin/python3

import numpy as np
from scipy.integrate import simpson
from abc import ABC, abstractmethod
from typing import Any, Callable
from ._base import WindowFunction


# Pre-defined models 
builtinWindows = {}

class SphericalTophat(WindowFunction):
    r"""
    Spherical tophat function.
    """

    def f(self, x: Any, nu: int = 0) -> Any:

        x   = np.asfarray( x )
        res = np.zeros_like( x )

        # NOTE: applied some masks to make sure the function looks good!...
        
        # mask for x closer to 0
        closeToZero = ( np.abs(x) < 1e-03 )

        # mask for other smallr x; at both ends, the function values is practically 0
        otherValues = np.logical_not( closeToZero | ( np.abs(x) > 1e+04 ) )

        if nu == 0:
            res[ closeToZero ] = 1.0
            res[ otherValues ] = 3.0 * ( 
                                            np.sin( x[ otherValues ] ) - x[ otherValues ] * np.cos( x[ otherValues ] ) 
                                    ) / x[ otherValues ]**3
            return res
        elif nu == 1:
            # res[ closeToZero ] = 0.0
            res[ otherValues ] = 3.0 * ( 
                                            ( x[ otherValues ]**2 - 3.0 ) * np.sin( x[ otherValues ] ) 
                                                + 3.0 * x[ otherValues ] * np.cos( x[otherValues] ) 
                                    ) / x[ otherValues ]**4
            return res
        elif nu == 2:
            res[ closeToZero ] = -0.2
            res[ otherValues ] = 3.0 * ( 
                                            ( x[ otherValues ]**2 - 12.0 ) * x[ otherValues ] * np.cos( x[ otherValues ] ) 
                                                - ( 5 * x[ otherValues ]**2 - 12.0 ) * np.sin( x[ otherValues ] ) 
                                    ) / x[ otherValues ]**5
            return res
        
        raise ValueError("nu can only be 0, 1 or 2")

tophat = SphericalTophat()
builtinWindows['tophat'] = tophat


class Gaussian(WindowFunction):
    r"""
    Gaussian function.
    """
    def f(self, x: Any, nu: int = 0) -> Any:

        x = np.asfarray( x )

        if nu == 0:
            return np.exp( -0.5*x**2 )
        elif nu == 1:
            return -x * np.exp( -0.5*x**2 )
        elif nu == 2:
            return ( x**2 - 1 ) * np.exp( -0.5*x**2 )

        raise ValueError("nu can only be 0, 1 or 2")

gaussian = Gaussian()
builtinWindows['gaussian'] = gaussian


# TODO: sharp-k filter
