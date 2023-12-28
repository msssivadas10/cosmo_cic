#!/usr/bin/python3

import numpy as np
from scipy.integrate import simpson
from scipy.special import hyp0f1
from typing import Any 
from ._base import Cosmology

class HaloError(Exception):
    r"""
    Base class of exceptions raised by halo model calculations.
    """

class HaloModel:
    r"""
    Base class representing a halo model.
    """
    __slots__ = 'cosmology', 'overdensity', 'ma', 'mb', 'mpts'

    def __init__(self, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> None:
        # cosmology model to use
        self.link(cosmology)
        # halo overdensity
        self.overdensity = overdensity
        # integration settings
        self.ma, self.mb, self.mpts = 1e+06, 1e+16, 1001

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
        # generating integration points
        lnma, lnmb, pts = np.log(self.ma), np.log(self.mb), self.mpts
        m, dlnm = np.linspace(lnma, lnmb, pts, retstep = True) # m is ln(m)
        m = np.reshape(np.exp(m), np.shape(m) + tuple(1 for _ in np.shape(z)))
        # function to integrate
        res = self.totalCount(m) * self.massFunction(m, z, 'dndlnm')
        # logspace integration
        res = simpson(res, dx = dlnm, axis = 0)
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
        # generating integration points
        lnma, lnmb, pts = np.log(self.ma), np.log(self.mb), self.mpts
        m, dlnm = np.linspace(lnma, lnmb, pts, retstep = True) # m is ln(m)
        m = np.reshape(np.exp(m), np.shape(m) + tuple(1 for _ in np.shape(z)))
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm') # for normalization
        res2 = m * res1
        # logspace integration
        res2 = simpson(res2, dx = dlnm, axis = 0)
        res1 = simpson(res1, dx = dlnm, axis = 0)
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
        # generating integration points
        lnma, lnmb, pts = np.log(self.ma), np.log(self.mb), self.mpts
        m, dlnm = np.linspace(lnma, lnmb, pts, retstep = True) # m is ln(m)
        m = np.reshape(np.exp(m), np.shape(m) + tuple(1 for _ in np.shape(z)))
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm') # for normalization
        res2 = self.biasFunction(m, z) * res1
        # logspace integration
        res2 = simpson(res2, dx = dlnm, axis = 0)
        res1 = simpson(res1, dx = dlnm, axis = 0)
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
        # generating integration points
        lnma, lnmb, pts = np.log(self.ma), np.log(self.mb), self.mpts
        m, dlnm = np.linspace(lnma, lnmb, pts, retstep = True) # m is ln(m)
        # m = np.reshape(np.exp(m), np.shape(m) + tuple(1 for _ in np.shape(z)))
        m = np.reshape(np.exp(m), np.shape(m) + tuple(1 for _ in np.broadcast_shapes(np.shape(k), np.shape(z))))
        # fourier space density profile
        prof = self.fourierProfile(k, m, z)
        res1 = self.massFunction(m, z, 'dndlnm') * prof
        res = 0
        # power spectrum calculation: central-satellite
        if 'cs' in retval:
            res2 = res1 * self.centralCount(m)**2 * self.satelliteFraction(m)
            res += 2*simpson(res2, dx = dlnm, axis = 0)
        # power spectrum calculation: satellite-satellite
        if 'ss' in retval:
            res2 = res1 * prof * (self.centralCount(m) * self.satelliteFraction(m))**2
            res += simpson(res2, dx = dlnm, axis = 0)
        # power spectrum calculation: central-satellite
        if '2h' in retval:
            res2 = res1 * self.totalCount(m) * self.biasFunction(m, z)
            res2 = simpson(res2, dx = dlnm, axis = 0)
            res += res2**2 * self.matterPowerSpectrum(k, z) 
        # normalization
        res = res / self.galaxyDensity(z)**2
        return res
    
    def galaxyCorrelation(self, 
                          r: Any, 
                          z: Any, 
                          retval: str | None = None, 
                          average: bool = True, ) -> Any:
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

        Returns
        -------
        res: array_like

        """
        if average: # average correlation function
            # generate integration points in log space
            lnka, lnkb = np.log(self.cosmology.settings.k_zero), np.log(self.cosmology.settings.k_inf)
            k, dlnk    = np.linspace(lnka, lnkb, self.cosmology.settings.k_pts, retstep = True) # k is log(k)
            # reshaping k array to work with array inputs
            k  = np.reshape(k, np.shape(k) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z))))
            k  = np.exp(k)
            r  = np.asfarray(r)
            kr = k*r
            # correlation
            res = self.galaxyPowerSpectrum(k, z, retval = retval) * k**3
            # hypergeometric function, 0F1(2.5, -0.25*x**2) = 3*( sin(x) - x * cos(x) ) / x**3
            res = hyp0f1(2.5, -0.25*kr**2) * res 
            res = simpson(res, dx = dlnk, axis = 0)
        else: # correlation function
            reltol   = self.cosmology.settings.reltol
            pts      = self.cosmology.settings.k_pts
            kra, krb = self.cosmology.settings.k_zero, 5*np.pi
            res = 0
            for i in range(500):
                # generate integration points in log space
                kr, dlnkr = np.linspace(np.log(kra), np.log(krb), pts, retstep = True) # kr is log(kr)
                # reshaping k array to work with array inputs
                kr = np.reshape(kr, np.shape(kr) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z))))
                kr = np.exp(kr)
                r  = np.asfarray(r)
                k  = kr / r
                # correlation 
                delta = self.galaxyPowerSpectrum(k, z, retval = retval) * k**3 * np.sinc(kr)
                delta = simpson(delta, dx = dlnkr, axis = 0)
                # if the step is much less than the sum, break (TODO: check condition)
                if np.all( np.abs(np.abs(delta) - reltol * np.abs(res)) < 1e-08 ): break
                res  += delta 
                kra, krb = krb, krb + np.pi
                # after the first step, number of points is 1/10 of the original
                if not i: pts = pts // 10
        return res / (2*np.pi**2)
    