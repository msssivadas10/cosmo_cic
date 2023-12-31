#!/usr/bin/python3

import numpy as np
from scipy.special import hyp0f1
from typing import Any 
from ._base import Cosmology
from .utils.objects import Settings

class HaloError(Exception):
    r"""
    Base class of exceptions raised by halo model calculations.
    """

class HaloModel:
    r"""
    Base class representing a halo model.
    """

    __slots__ = 'cosmology', 'overdensity', 'settings'

    def __init__(self, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> None:
        # cosmology model to use
        self.link(cosmology)
        # halo overdensity
        self.overdensity = overdensity
        # general settings table
        self.settings = Settings()

    def link(self, model: Cosmology) -> None:
        r"""
        Link a cosmology model.
        """
        if model is not None and not isinstance(model, Cosmology):
            raise TypeError("model must be a 'Cosmology' object")
        self.cosmology = model
        return

    def centralCount(self, m: Any) -> Any:
        r"""
        Return the average number of central galaxies in halo of mass m.

        Parameters
        ----------
        m: array_like

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def satelliteFraction(self, m: Any) -> float:
        r"""
        Return the average fraction of satellite galaxies in halo of mass m.
        
        Parameters
        ----------
        m: array_like

        Returns
        -------
        res: array_like
        
        """
        raise NotImplementedError()
    
    def totalCount(self, m: Any) -> float:
        r"""
        Return the total (average) number of galaxies in a halo of mass m.
        
        Parameters
        ----------
        m: array_like

        Returns
        -------
        res: array_like
        
        """
        return self.centralCount(m) * ( 1. + self.satelliteFraction(m) )
    
    def massFunction(self, 
                     m: Any, 
                     z: Any, 
                     retval: str = 'dndlnm',) -> Any:
        r"""
        Return the halo mass function as function of mass m (Msun/h) and redshift z.
        
        Parameters
        ----------
        m: array_like
        z: array_like
        retval: str, default = `dndlnm`

        Returns
        -------
        res: array_like
        
        """
        if self.cosmology is None:
            raise HaloError("no cosmology model is linked with this model")
        res = self.cosmology.haloMassFunction(m, z, self.overdensity, retval)
        return res
    
    def biasFunction(self, 
                     m: Any, 
                     z: Any, ) -> Any:
        r"""
        Return the halo bias as function of halo mass m (Msun/h) and redshift z.
        
        Parameters
        ----------
        m: array_like
        z: array_like
        
        Returns
        -------
        res: array_like
        
        """
        if self.cosmology is None:
            raise HaloError("no cosmology model is linked with this model")
        res = self.cosmology.haloBias(m, z, self.overdensity)
        return res
    
    def fourierProfile(self,
                       k: Any, 
                       m: Any, 
                       z: Any, ) -> Any:
        r"""
        Return the fourier space density profile as function of wavenumber k (h/Mpc) for a halo of 
        mass m (Msun/h) at redshift z.
        
        Parameters
        ----------
        k: array_like
        m: array_like
        z: array_like
        
        Returns
        -------
        res: array_like
        
        """
        if self.cosmology is None:
            raise HaloError("no cosmology model is linked with this model")
        res = self.cosmology.haloProfile(k, m, z, self.overdensity, fourier_transform = True)
        return res
    
    def matterPowerSpectrum(self, 
                            k: Any, 
                            z: Any, ) -> Any:
        r"""
        Return the matter power spectrum as function of wavenumber k (h/Mpc) at redshift z.
        
        Parameters
        ----------
        k: array_like
        z: array_like

        Returns
        -------
        res: array_like

        """
        if self.cosmology is None:
            raise HaloError("no cosmology model is linked with this model")
        res = self.cosmology.matterPowerSpectrum(k, z, deriv = 0, normalize = True, nonlinear = False)
        return res
    
    def galaxyDensity(self, z: Any) -> Any:
        r"""
        Return the galaxy number density at redshift z.
        
        Parameters
        ----------
        z: array_like

        Returns
        -------
        res: array_like
        
        """
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.shape(z))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # function to integrate
        res = self.totalCount(m) * self.massFunction(m, z, 'dndlnm')
        # logspace integration
        res = np.sum( res * wts, axis = 0 )
        return res

    def averageHaloMass(self, z: Any) -> Any:
        r"""
        Return the average halo mass occupied by galaxies at redshift z.
        
        Parameters
        ----------
        z: array_like

        Returns
        -------
        res: array_like

        """
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.shape(z))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm') # for normalization
        res2 = m * res1
        # logspace integration
        res2 = np.sum( res2 * wts, axis = 0 )
        res1 = np.sum( res1 * wts, axis = 0 )
        return res2 / res1
    
    def effectiveBias(self, z: Any) -> Any:
        r"""
        Return the effective bias of galaxies at redshift z.
        
        Parameters
        ----------
        z: array_like

        Returns
        -------
        res: array_like

        """
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.shape(z))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm') # for normalization
        res2 = self.biasFunction(m, z) * res1
        # logspace integration
        res2 = np.sum( res2 * wts, axis = 0 )
        res1 = np.sum( res1 * wts, axis = 0 )
        return res2 / res1
    
    def galaxyPowerSpectrum(self, 
                            k: Any, 
                            z: Any, 
                            retval: str | None = None, ) -> Any:
        r"""
        Return the galaxy power spectrum as function of wavenumber k (h/Mpc) at redshift z.
        
        Parameters
        ----------
        k: array_like
        z: array_like
        retval: str, default = None
            Specify the pairing. `cs` for central-satellite pairs, `ss` for satellite-satellite and 
            `2h` for pairs in two different halos (2-halo). `cs+ss` (same as `1h`) gives the 1-halo 
            result and `1h+2h` gives the total.

        Returns
        -------
        res: array_like

        """ 
        # get spectrums wanted to calculate  
        retval = ( (retval or '').lower().strip().replace('1h', 'cs+ss') or 'cs+ss+2h' ).split('+')
        retval = map(lambda __str: __str.strip(), retval)
        retval = filter(lambda __str: len(__str) > 0, retval)
        retval = sorted(set(retval), key = lambda __str: {'cs':0, 'ss':1, '2h':2}.get(__str, 3))
        if retval[-1] not in {'cs', 'ss', '2h'} and retval[-1]:
            raise ValueError(f"invalid power spectrum type: { retval[-1] }")
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.broadcast_shapes(np.shape(k), np.shape(z)))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # fourier space density profile
        prof = self.fourierProfile(k, m, z)
        res1 = self.massFunction(m, z, 'dndlnm') * prof
        res = 0
        # power spectrum calculation: central-satellite
        if 'cs' in retval:
            res2 = res1 * self.centralCount(m)**2 * self.satelliteFraction(m)
            res += 2*np.sum( res2 * wts, axis = 0 )
        # power spectrum calculation: satellite-satellite
        if 'ss' in retval:
            res2 = res1 * prof * (self.centralCount(m) * self.satelliteFraction(m))**2
            res += np.sum( res2 * wts, axis = 0 )
        # power spectrum calculation: central-satellite
        if '2h' in retval:
            res2 = res1 * self.totalCount(m) * self.biasFunction(m, z)
            res2 = np.sum( res2 * wts, axis = 0 )
            res += res2**2 * self.matterPowerSpectrum(k, z) 
        # normalization
        res = res / self.galaxyDensity(z)**2
        return res
    
    def galaxyCorrelation(self, 
                          r: Any, 
                          z: Any, 
                          retval: str | None = None, 
                          average: bool = True, 
                          ht: bool = False, ) -> Any:
        r"""
        Return the galaxy correlation as function of scale r (Mpc/h) at redshift z.
        
        Parameters
        ----------
        k: array_like
        z: array_like
        retval: str, default = None
            Specify the pairing. `cs` for central-satellite pairs, `ss` for satellite-satellite and 
            `2h` for pairs in two different halos (2-halo). `cs+ss` (same as `1h`) gives the 1-halo 
            result and `1h+2h` gives the total.
        average: bool, default = True
            If true, return the average correlation over a sphere of radius r.

        ht: bool, default = False
            Use hankel transform to find correlation (Experimental).

        Returns
        -------
        res: array_like
            Matter 2-point correlation functions 

        Notes
        -----
        Using a hankel transform is an experimental stage. At this time, result not matching with the 
        usual result (implementation of hankel transform rule is to be checked).  

        """
        # experimental: using spherical hankel transform
        if ht:
            pts = self.settings.corr_quad.nodes
            wts = self.settings.corr_quad.weights[1 if average else 0]
            # reshaping arrays to work with array inputs
            shape = np.shape(pts) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z)))
            pts = np.reshape(pts, shape)
            wts = np.reshape(wts, shape)
            k   = pts / r # pts is k*r
            res = self.galaxyPowerSpectrum(k, z, retval) * k**2
            # integration
            res = np.sum( res*wts, axis = 0 ) / r
            return res
        
        # generate integration points in log space
        pts = self.settings.k_quad.nodes
        wts = self.settings.k_quad.weights
        # reshaping k array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z)))
        pts = np.reshape(pts, shape)
        wts = np.reshape(wts, shape)
        # correlation 
        k  = np.exp(pts)
        r  = np.asfarray(r)
        kr = k*r
        res = self.galaxyPowerSpectrum(k, z, retval) * k**3
        if average: # average correlation function
            res = res * hyp0f1(2.5, -0.25*kr**2) # 0F1(2.5, -0.25*x**2) = 3*( sin(x) - x * cos(x) ) / x**3
        else: # correlation function
            res = res * np.sinc(kr / np.pi)
        res = np.sum( res*wts, axis = 0 ) / (2*np.pi**2)
        return res
    