#!/usr/bin/python3

import numpy as np
from scipy.special import sici
from typing import Any 


class Profile:
    r"""
    Base class representing a halo density profile.
    """

    __slots__ = 'fvalue', 'kvalue'

    def __init__(self, fvalue: float = 0.01, kvalue: float = 4.0) -> None:
        self.fvalue, self.kvalue = fvalue, kvalue

    def set(self, fvalue: float = None, kvalue: float = None):
        r"""
        Reset the free parameters of the profile.
        """

        if fvalue is not None: self.fvalue = fvalue 
        if kvalue is not None: self.kvalue = kvalue
        return self 
    
    def A(self, c: Any) -> Any:
        r"""
        Function related to virial mass of the halo and concentration. 
        """
        ...

    def getConcentration(self, model: object, rvir: Any, z: Any, **kwargs) -> Any:
        r"""
        Return the concentration parameter corresponding to a virial radius.
        """

        rvir  = np.asfarray( rvir )
        z     = np.asfarray( z )
        if np.ndim( rvir ) and np.ndim( z ):
            assert np.size( rvir, -1 ) == np.size( z )

        rstar = np.cbrt( self.fvalue ) * rvir # charecteristic radius

        # collapse redshift
        zcoll = model.collapseRedshift( rstar.flatten(), **kwargs ).reshape( rstar.shape )

        # concentration parameter
        cvir = self.kvalue * (1. + zcoll) / (1. + z)
        return cvir

    def getCharecteristicValues(self, 
                                model: object, 
                                m: Any, 
                                z: Any, 
                                overdensity: float = None, 
                                **kwargs                 ) -> Any:
        r"""
        Return the values of the charecteristic radius and density.
        """
        
        m = np.ravel( m ) if np.ndim( m ) < 1 else np.asfarray( m )
        z = np.ravel( z ) if np.ndim( z ) < 1 else np.asfarray( z )

        # virial radius
        rvir = model.lagrangianR( m, overdensity )[:, None] / (z + 1)

        # halo radius at collapse, r*
        rstar = np.cbrt( self.fvalue ) * rvir

        # collapse redshift
        zcoll = model.collapseRedshift( rstar.flatten(), **kwargs ).reshape( rstar.shape )

        # concentration parameter
        cvir = self.kvalue * (1. + zcoll) / (1. + z)
        
        # universe density, in units of critical density
        pu = model.Om0 * (z + 1)**3

        # inner radius parameter
        rs = rvir / cvir

        # inner density parameter
        ps = pu * overdensity * cvir**3 / self.A( cvir ) / 3

        return rs, ps

    def f(self, r: Any, rs: Any, ps: Any) -> Any:
        r"""
        Return the value of the real space density profile function. 
        """
        ...

    def u(self, k: Any, rs: Any, ps: Any) -> Any:
        r"""
        Return the value of the fourier space density profile function. 
        """
        ...

    def __call__(self, 
                 model: object, 
                 arg: Any, 
                 m: Any, 
                 z: Any, 
                 overdensity: float = None, 
                 ft: bool = False,
                 **kwargs                 ) -> Any:
        r"""
        Return the value of the density profile function.
        """

        rs, ps = self.getCharecteristicValues(model, m, z, overdensity, **kwargs)
        return self.u(arg, rs, ps) if ft else self.f(arg, rs, ps) 


######################################################################################
# Predefined models
######################################################################################

builtinProfiles = {}


class NFW(Profile):
    r"""
    NFW halo density profile.
    """

    def A(self, c: Any) -> Any:
        
        c = np.asfarray( c )
        return np.log( 1 + c ) - c / ( 1 + c )
    
    def f(self, r: Any, rs: Any, ps: Any) -> Any:
         
        assert np.shape( rs ) == np.shape( ps ), "rs and psmust have same shape"

        d  = np.ndim( rs )
        rs = np.transpose(rs)[..., None] if d else rs
        ps = np.transpose(ps)[..., None] if d else ps

        res = r / rs
        res = res * ( res + 1 )**2
        res = ps / res
        return np.transpose( res ) if d else res

nfw = NFW()
builtinProfiles['nfw'] = nfw
