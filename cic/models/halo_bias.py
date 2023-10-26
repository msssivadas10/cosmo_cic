#!/usr/bin/python3

import numpy as np
from cic.models.constants import *
from abc import ABC, abstractmethod
from typing import Any


class HaloBias( ABC ):
    r"""
    Base class representing a halo bias function object.
    """

    @abstractmethod
    def func(self, model: object, nu: Any, z: Any, overdensity: Any) -> Any:
        r"""
        Calculate the halo bias function values.
        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any, 
                 variance_args: dict = {}, ) -> Any:
        r"""
        Calculate the halo bias function values.
        """

        m, z = np.asfarray( m ), np.asfarray( z )

        # flatten any multi-dimensional arrays, if given
        m = np.ravel( m ) if np.ndim( m ) > 1 else m
        z = np.ravel( z ) if np.ndim( z ) > 1 else z

        r  = model.lagrangianR( m ) # * np.cbrt( overdensity )
        nu = DELTA_SC / np.sqrt( model.matterVariance( r, z, nu = 0, **variance_args ) )

        res = self.func( model, nu, z, overdensity )
        return res


class Cole89( HaloBias ):
    r"""
    Linear bias model given by Cole & Kaiser (1989) and Mo & White (1996).
    """
    
    def func(self, model: object, nu: float, z: float = 0, overdensity: int = None) -> float:
        
        nu   = np.asfarray( nu )
        z    = np.asfarray( z )
        bias = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        return bias


class Tinker10( HaloBias ):
    r"""
    Linear bias model given by Tinker et al. (2010).
    """
    
    def func(self, model: object, nu: float, z: float = 0, overdensity: int = 200) -> float:

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


# a dict of predefined models
available = {
                'cole89'  : Cole89(), 
                'tinker10': Tinker10(), 
            }
