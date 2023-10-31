#!/usr/bin/python3

import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any


_MassFunctionResult = namedtuple('_MassFunctionResult', 
                                 ['m', 'r', 'sigma', 'dlnsdlnm', 'fsigma', 'dndlnm',])

class MassFunction(ABC):
    r"""
    Base class representing a halo mass-function object.
    """

    @abstractmethod
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

