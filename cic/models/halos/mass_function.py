#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from collections import namedtuple
from typing import Any

from ..utils.constants import *


_MassFunctionResult = namedtuple('_MassFunctionResult', 
                                 ['m', 'r', 'sigma', 'dlnsdlnm', 'fsigma', 'dndlnm',])

class MassFunction:
    r"""
    Base class representing a halo mass-function object.
    """

    def f(self, 
          model: object, 
          s: Any, 
          z: Any, 
          overdensity: Any, ) -> Any:
        r"""
        Calculate the mass function.
        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any, 
                 retval: str = 'dndm', 
                 grid: bool = False, ) -> Any:
        r"""
        Calculate the halo mass-function as function of halo mass.
        """

        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )

        # mass-function f(sigma)
        r  = model.lagrangianR( m, overdensity )
        s  = np.sqrt( model.matterVariance(r, z, 
                                           nu = 0, 
                                           grid = False, 
                                           exact = not model.settings.useInterpolation ))
        fs = self.f( model, s, z, overdensity )
        if retval in ( 'f', 'fsigma' ):
            return fs
        
        # mass function dn/dn(m)
        dlnsdlnm = model.matterVariance(r, z, 
                                        nu = 1, 
                                        grid = False, 
                                        exact = not model.settings.useInterpolation ) / 6.0
        dndlnm   = fs * np.abs( dlnsdlnm ) * model.matterDensity( z ) / m
        if retval == 'dndlnm':
            return dndlnm

        # mass function dn/dm
        if retval == 'dndm':
            return dndlnm / m
        
        # return all values as an array
        if retval == 'full':
            return _MassFunctionResult(m        = m, 
                                       r        = r, 
                                       sigma    = s, 
                                       dlnsdlnm = dlnsdlnm, 
                                       fsigma   = fs,
                                       dndlnm   = dndlnm,  )
        
        raise ValueError( "invalid value for argument retval: '%s'" % retval ) 


#########################################################################################
# Predefined models
#########################################################################################

builtinMassfunctions = {}


class Press74(MassFunction):
    r"""
    Halo mass function model by Press & Schechter (1974). It is based on spherical collapse.
    """

    def f(self, 
          model: object, 
          s: Any, 
          z: Any, 
          overdensity: Any = None, ) -> Any:
        
        nu = model.collapseDensity( z ) / np.asfarray( s )
        f  = np.sqrt( 2 / np.pi ) * nu * np.exp( -0.5 * nu**2 )
        return f
    
Press74 = Press74()
builtinMassfunctions['press74'] = Press74
    

class Sheth01(MassFunction):
    r"""
    Halo mass function model by Sheth et al (2001). It is based on ellipsoidal collapse.
    """

    def __init__(self) -> None:
        self.A, self.a, self.p = 0.3222, 0.707, 0.3 # parameters
        return super().__init__()

    def f(self, 
          model: object, 
          s: Any, 
          z: Any, 
          overdensity: Any = None, ) -> Any:
        
        A = self.A
        a = self.a
        p = self.p

        nu = model.collapseDensity( z ) / np.asarray( s )
        f  = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
        return f

sheth01 = Sheth01()
builtinMassfunctions['sheth01'] = sheth01


class Tinker08(MassFunction):
    r"""
    Halo mass function model by Tinker et al (2008). This model is redshift dependent.
    """

    def __init__(self) -> None:
        
        self.A  = CubicSpline(
                                [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                                [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],
                            )
        self.a  = CubicSpline(
                                [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                                [1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66 ],
                            )
        self.b  = CubicSpline(
                                [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                                [2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41 ],
                            )
        self.c  = CubicSpline(
                                [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                                [1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44 ],
                            )
        
        return super().__init__()
        
    def f(self, 
          model: object, 
          s: Any, 
          z: Any, 
          overdensity: Any, ) -> Any:
        
        s   = np.asfarray( s )
        zp1 = np.asfarray( z ) + 1.

        assert np.all( zp1 > 0. ), "redshift values must be greater than -1"

        alpha = 10.0**( -( 0.75 / np.log10( overdensity / 75 ) )**1.2 ) # eqn 8 
        A     = self.A( overdensity ) / zp1**0.14  # eqn 5
        a     = self.a( overdensity ) / zp1**0.06  # eqn 6     
        b     = self.b( overdensity ) / zp1**alpha # eqn 7 
        c     = self.c( overdensity )

        f = A * ( 1 + ( b / s )**a ) * np.exp( -c / s**2 ) # eqn 3
        return f
    
tinker08 = Tinker08()   
builtinMassfunctions['tinker08'] = tinker08
 
