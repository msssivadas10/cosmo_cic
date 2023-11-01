#!/usr/bin/python3

import numpy as np
from scipy.special import erf 
from scipy.integrate import simpson
from typing import Any

from .utils.constants import *
from .cosmology import Cosmology

INF_M = 1e+20  # infinity for mass calculations


class HaloError(Exception):
    r"""
    Base class of exceptions used in halo calculations.
    """


class HaloModel:

    __slots__ = ('lnMmin', 'lnMsat', 'lnMcut', 'sigma', 'alpha', 'overdensity', 'cosmo', 'profile')

    def __init__(self,
                 cosmo: Cosmology, 
                 m_min: float, 
                 m_sat: float, 
                 sigma: float, 
                 alpha: float, 
                 m_cut: float = None, 
                 overdensity: int = 200,
                 halo_profile: str = 'nfw',
                 ) -> None:
        
        # parameters for central galaxy count
        self.lnMmin, self.sigma = np.log(m_min), sigma

        # parameters for satellite galaxy count
        self.lnMsat, self.alpha = np.log(m_sat), alpha
        self.lnMcut             = -0.5 * self.lnMmin if m_cut is None else np.log(m_cut)

        # other models to use:
        
        # cosmology model
        assert isinstance(cosmo, Cosmology), "cosmo must be a 'Cosmology' object"
        assert cosmo._model.mass_function is not None, "cosmology should have a mass function model"
        assert cosmo._model.halo_bias is not None, "cosmology should have a bias model"
        assert cosmo._model.power_spectrum is not None, "cosmology should have a power spectrum model"
        self.cosmo = cosmo 

        # halo density profile
        self.profile = halo_profile 

        # halo overdensity value
        self.overdensity = overdensity

    def centralCount(self, m: Any) -> Any:
        r"""
        Calculate the average number of central galaxies in halo of mass m.
        """

        x = ( np.log(m) - self.lnMmin ) / ( SQRT_2 * self.sigma )
        return 0.5 * ( 1. + erf(x) )

    def satelliteFraction(self, m: Any) -> float:
        r"""
        Calculate the average fraction of satellite galaxies in halo of mass m.
        """

        m_cut, m_sat = np.exp(self.lnMcut), np.exp(self.lnMsat)
        return ( ( np.asfarray(m) - m_cut ) / m_sat )**self.alpha
    
    def totalCount(self, m: Any) -> float:
        r"""
        Calculate the average number of galaxies in a halo of mass m.
        """

        return self.centralCount(m) * ( 1. + self.satelliteFraction(m) )
    
    def massFunction(self, 
                     m: Any, 
                     z: Any, 
                     retval: str = 'dndlnm', 
                     grid: bool = False,    ) -> Any:
        r"""
        Calculate the halo mass function.
        """

        return self.cosmo.massFunction(m, z, self.overdensity, retval, grid)
    
    def biasFunction(self, 
                     m: Any, 
                     z: Any, 
                     grid: bool = False, ) -> Any:
        r"""
        Calculate the halo bias function.
        """

        return self.cosmo.biasFunction(m, z, self.overdensity, grid)
    
    def galaxyDensity(self, z: Any) -> Any:
        r"""
        Calculate the galaxy number density at a redshift.
        """

        

        m, dlnm = np.linspace(self.lnMmin - 5 * SQRT_2 * self.sigma, 
                              np.log( self.cosmo.settings.mInfinity ), 
                              self.cosmo.settings.mIntegralPoints, 
                              retstep = True, )
        m       = np.exp( m )
        
        if np.ndim( z ):
            z, m = np.ravel( z ), m[:, None]

        # function to integrate
        res = self.totalCount( m ) * self.massFunction(m, z, 'dndlnm', False)

        # log space integration
        res =  simpson( res, dx = dlnm, axis = 0 )
        return res
    
    def averageGalaxyDensity(self) -> Any:
        r"""
        Calculate the average galaxy number density.
        """
        raise NotImplementedError()
    
    def averageHaloMass(self, z: Any) -> Any:
        r"""
        Calculate the average halo mass of galaxies.
        """

        m, dlnm = np.linspace(self.lnMmin - 5 * SQRT_2 * self.sigma, 
                              np.log( self.cosmo.settings.mInfinity ), 
                              self.cosmo.settings.mIntegralPoints, 
                              retstep = True, )
        m       = np.exp( m )
        
        if np.ndim( z ):
            z, m = np.ravel( z ), m[:, None]

        # function to integrate
        res1 = self.totalCount( m ) * self.massFunction(m, z, 'dndlnm', False) 
        res2 = m * res1

        # log space integration
        res1 =  simpson( res1, dx = dlnm, axis = 0 )
        res2 =  simpson( res2, dx = dlnm, axis = 0 )
        return res2 / res1

    def effectiveBias(self, z: Any) -> Any:
        r"""
        Calculate the effective bias of galaxies.
        """

        m, dlnm = np.linspace(self.lnMmin - 5 * SQRT_2 * self.sigma, 
                              np.log( self.cosmo.settings.mInfinity ), 
                              self.cosmo.settings.mIntegralPoints, 
                              retstep = True, )
        m       = np.exp( m )
        
        if np.ndim( z ):
            z, m = np.ravel( z ), m[:, None]

        # function to integrate
        res1 = self.totalCount( m ) * self.massFunction(m, z, 'dndlnm', False) 
        res2 = res1 * self.biasFunction( m, z, False ) 

        # log space integration
        res1 =  simpson( res1, dx = dlnm, axis = 0 )
        res2 =  simpson( res2, dx = dlnm, axis = 0 )
        return res2 / res1
    
    
    # TODO: galaxy power spectrum


