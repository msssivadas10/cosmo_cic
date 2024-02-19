#!/usr/bin/python3

import numpy as np
from scipy.special import hyp0f1, erf 
from typing import Any, Callable
from .._base import Cosmology
from ..utils.objects import Settings

class HaloError(Exception):
    r"""
    Base class of exceptions raised by halo model calculations.
    """

class HaloModel:
    r"""
    Base class representing a halo model.

    Parameters
    ----------
    cosmology: Cosmology
        Cosmology model to use for other calculations.
    overdensity: float, None
        Halo over-density value to use.
    z_distribution: callable
        Redshift distribution function. Must be a function of redshift z and the cosmology 
        model, returning a float (or an array).
    
    """

    __slots__ = 'cosmology', 'overdensity', 'settings', 'z_distribution', 'z_range', 'averageDensity', 

    def __init__(self, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> None:
        # general settings table
        self.settings = Settings()
        self.overdensity = overdensity
        # defaults
        self.cosmology = None
        self.z_distribution = None
        self.z_range        = (0., np.inf)
        self.averageDensity = None
        # link cosmology model 
        self.setCosmology(cosmology)

    def setCosmology(self, cosmology: Cosmology = None, ) -> None:
        r"""
        Link a cosmology model.

        Parameters
        ----------
        cosmology: Cosmology

        """
        if cosmology is not None:
            if not isinstance(cosmology, Cosmology):
                raise TypeError("model must be a 'Cosmology' object")
            self.cosmology = cosmology
        return
    
    def setRedshiftDistribution(self, 
                                z_distribution: Callable[[Any, Cosmology], Any] = None, 
                                z_min: float = 0.0,
                                z_max: float = np.inf, ) -> None:
        r"""
        Link a redshift distribution.

        Parameters
        ----------
        z_distribution: callable
            Redshift distribution function. Must be a function of redshift z and the cosmology 
            model, returning a float (or an array).
        z_min, z_max: float, optional
            Redshift range to average. Default is :math:`[0, \infty]`.

        """
        if z_distribution is not None:
            if not callable(z_distribution):
                raise TypeError("z_distribution must be a callable object")
            self.z_distribution = z_distribution
            self.z_range = ( z_min, z_max )
            # calculate average density
            self.averageDensity = self.galaxyDensity()
        return

    def centralCount(self, m: Any) -> Any:
        r"""
        Return the average number of central galaxies in halo of mass m (Msun/h).

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
        Return the average fraction of satellite galaxies in halo of mass m (Msun/h).
        
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
        Return the total (average) number of galaxies in a halo of mass m (Msun/h).
        
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
    
    def redshiftAverageRule(self) -> tuple:
        r"""
        Calculate the points and weights for taking an average over the redshift range.

        Returns
        -------
        pts: array_like
        wts: array_like 

        """
        if self.z_distribution is None:
            raise ValueError("no redshift distribution is linked to model")
        if self.cosmology is None:
            raise ValueError("no cosmology model is available for normalising")
        # generating integration points
        pts, wts   = self.settings.z_quad.nodes, self.settings.z_quad.weights
        z_min, z_max = self.z_range   
        if np.isinf(z_max): # converting redshift to time
            a_min, a_max = (z_min + 1.)**-1, (z_max + 1.)**-1
            scale    = 0.5*( a_max - a_min )
            pts, wts = (pts + 1.) * scale + a_min, wts * scale
            non_zero = ( pts > 0.)
            # select non-zero time and convert to redshift 
            # NOTE: this assume the distribution value is zero at z = inf
            pts, wts = 1./pts[non_zero] - 1., wts[non_zero]
        else: # use redshift
            scale    = 0.5*( z_max - z_min )
            pts, wts = (pts + 1.) * scale + z_min, wts * scale
        # weights
        wts = wts * self.z_distribution( pts, self.cosmology ) * self.cosmology.comovingVolumeElement( pts ) * (1. + pts)**2
        # normalize weights
        wts = wts / np.sum( wts, axis = 0 )
        return pts, wts
    
    def galaxyDensity(self, z: Any = None) -> Any:
        r"""
        Return the galaxy number density at redshift z.
        
        Parameters
        ----------
        z: array_like, optional
            If not given, return the average value over the redshift range.

        Returns
        -------
        res: array_like
        
        """
        if z is None: # calculate the redshift average
            z, wts = self.redshiftAverageRule()
            res = np.sum( self.galaxyDensity( z ) * wts, axis = 0 )
            return res
        
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
    
    def averageHaloMass(self, z: Any = None) -> Any:
        r"""
        Return the average halo mass occupied by galaxies at redshift z.
        
        Parameters
        ----------
        z: array_like, optional
            If not given, return the average value over the redshift range.

        Returns
        -------
        res: array_like

        """
        if z is None: # calculate the redshift average
            z, wts = self.redshiftAverageRule()
            res = np.sum( self.averageHaloMass( z ) * wts, axis = 0 )
            return res
        
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.shape(z))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm')
        res1 = m * res1
        # logspace integration
        res1 = np.sum( res1 * wts, axis = 0 ) / self.averageDensity
        return res1
    
    def effectiveBias(self, z: Any = None) -> Any:
        r"""
        Return the effective bias of galaxies at redshift z.
        
        Parameters
        ----------
        z: array_like, optional
            If not given, return the average value over the redshift range.

        Returns
        -------
        res: array_like

        """
        if z is None: # calculate the redshift average
            z, wts = self.redshiftAverageRule()
            res = np.sum( self.effectiveBias( z ) * wts, axis = 0 )
            return res
        
        # generating integration points in log space
        pts, wts = self.settings.m_quad.nodes, self.settings.m_quad.weights
        # reshaping m array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.shape(z))
        m   = np.reshape(np.exp(pts), shape) 
        wts = np.reshape(wts, shape)
        # function to integrate
        res1 = self.totalCount(m) * self.massFunction(m, z, 'dndlnm')
        res1 = self.biasFunction(m, z) * res1 
        # logspace integration
        res1 = np.sum( res1 * wts, axis = 0 ) / self.averageDensity
        return res1
    
    def galaxyPowerSpectrum(self, 
                            k: Any, 
                            z: Any = None, 
                            retval: str | None = None, ) -> Any:
        r"""
        Return the galaxy power spectrum as function of wavenumber k (h/Mpc) at redshift z.
        
        Parameters
        ----------
        k: array_like
        z: array_like, optional
            If not given, return the average value over the redshift range.
        retval: str, default = None
            Specify the pairing. `cs` for central-satellite pairs, `ss` for satellite-satellite and 
            `2h` for pairs in two different halos (2-halo). `cs+ss` (same as `1h`) gives the 1-halo 
            result and `1h+2h` gives the total.

        Returns
        -------
        res: array_like

        """
        if z is None: # calculate the redshift average
            z, wts = self.redshiftAverageRule() 
            shape    = np.shape(z) + tuple(1 for _ in np.shape(k))
            res, wts = np.reshape(z, shape), np.reshape(wts, shape)
            res      = np.sum( self.galaxyPowerSpectrum(k, res, retval) * wts, axis = 0 ) 
            return res
        
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
        res = res / self.averageDensity**2
        return res
    
    def galaxyCorrelation(self, 
                          r: Any, 
                          z: Any = None, 
                          retval: str | None = None, 
                          average: bool = True, 
                          ht: bool = False, ) -> Any:
        r"""
        Return the galaxy correlation as function of scale r (Mpc/h) at redshift z.
        
        Parameters
        ----------
        k: array_like
        z: array_like, optional
            If not given, return the average value over the redshift range.
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
        if z is None: # calculate the redshift average
            z, wts = self.redshiftAverageRule() 
            shape    = np.shape(z) + tuple(1 for _ in np.shape(r))
            res, wts = np.reshape(z, shape), np.reshape(wts, shape)
            res      = np.sum( self.galaxyCorrelation(r, res, retval, average, ht) * wts, axis = 0 ) 
            return res
        
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
    
#########################################################################################################
#                                           Built-in models                                             #
#########################################################################################################
    
class Zehavi05(HaloModel):
    r"""
    A 3-parameter halo model given in Zehavi (2005).

    Parameters
    ----------
    logm_min: float
        Minimum mass of the halos that can host central galaxies above a luminosity threshold.
    logm1: float
        Specify the amplitude for the power law count for satellite galaxies.
    alpha: float
        Slope of the satellite galaxy power law relation.
    cosmology: Cosmology
        Cosmology model to use for other calculations.
    overdensity: float, None
        Halo over-density value to use.
    
    **Note**: log with base 10 is used here.

    """

    __slots__ = 'logm_min', 'logm1', 'alpha', 

    def __init__(self, 
                 logm_min: float, 
                 logm1:float, 
                 alpha: float, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> None:
        super().__init__(cosmology, overdensity)
        # model parameters:
        self.logm_min = logm_min   
        self.logm1    = logm1 
        self.alpha    = alpha

    def __repr__(self) -> str:
        logm_min, logm1, alpha = self.logm_min, self.logm1, self.alpha
        return f"HaloModel({logm_min=}, {logm1=}, {alpha=})"

    @classmethod
    def zehavi05(cls,
                 mag: float, 
                 z: float = None, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> 'Zheng07':
        r"""
        Create a halo model using a pre-defined set of parameters, given in table 3 of Zehavi (2005).

        Parameters
        ----------
        mag: float
            Absolute magnitude threshold used for sample selection.
        z: float, unused
        cosmology: Cosmology
            Cosmology model to use for other calculations.
        overdensity: float, None
            Halo over-density value to use. 

        """
        #-----------------------------------------------
        #            Mag  :   log(Mmin), log(M1), alpha
        #-----------------------------------------------
        paramList = {-22.0: [ 13.91    , 14.92  , 1.43 ],
                     -21.5: [ 13.27    , 14.60  , 1.94 ],
                     -21.0: [ 12.72    , 14.09  , 1.39 ],
                     -20.5: [ 12.30    , 13.67  , 1.21 ], 
                     -20.0: [ 12.01    , 13.42  , 1.16 ], 
                     -19.5: [ 11.76    , 13.15  , 1.13 ], 
                     -19.0: [ 11.59    , 12.94  , 1.08 ],
                     -18.5: [ 11.44    , 12.77  , 1.01 ],
                     -18.0: [ 11.27    , 12.57  , 0.92 ],}
        # interpolate to the nearest value 
        params = paramList[ min( paramList, key = lambda __m: abs(__m - mag) ) ]
        return cls( *params, cosmology, overdensity )

    def centralCount(self, m: Any) -> Any:
        res = np.log10(m) - self.logm_min
        res = np.heaviside( res, 1. )
        return res 
    
    def satelliteFraction(self, m: Any) -> float:
        m1  = 10**self.logm1
        res = np.divide( m, m1 )**self.alpha
        return res

class Zheng07(HaloModel):
    r"""
    A 5-parameter halo model given in Zheng (2007).

    Parameters
    ----------
    logm_min: float
        Minimum mass of the halos that can host central galaxies above a luminosity threshold.
    sigma_logm: float
        Specify the width of the cutoff profile.
    logm0, logm1: float
        Specify the shift and amplitude for the power law count for satellite galaxies.
    alpha: float
        Slope of the satellite galaxy power law relation.
    cosmology: Cosmology
        Cosmology model to use for other calculations.
    overdensity: float, None
        Halo over-density value to use.

    """

    __slots__ = 'logm_min', 'sigma_logm', 'logm0', 'logm1', 'alpha', 

    def __init__(self, 
                 logm_min: float, 
                 sigma_logm: float, 
                 logm0: float, 
                 logm1:float, 
                 alpha: float, 
                 cosmology: Cosmology | None = None, 
                 overdensity: float | None = None, ) -> None:
        super().__init__(cosmology, overdensity)
        # model parameters:
        # - central galaxy:
        self.logm_min   = logm_min   
        self.sigma_logm = sigma_logm 
        # - satellite galaxy:
        self.logm0 = logm0
        self.logm1 = logm1 
        self.alpha = alpha

    def __repr__(self) -> str:
        logm_min, sigma_logm, logm0, logm1, alpha = self.logm_min, self.sigma_logm, self.logm0, self.logm1, self.alpha
        return f"HaloModel({logm_min=}, {sigma_logm=}, {logm0=}, {logm1=}, {alpha=})"

    @classmethod
    def deep2(cls, 
              mag: float, 
              z: float = None, 
              cosmology: Cosmology | None = None, 
              overdensity: float | None = None, ) -> 'Zheng07':
        r"""
        Create a halo model using a pre-defined set of parameters, given in Table 1 of Zheng (2007), 
        for DEEP2. Nearest neighbour interpolation is used for values not in the table.

        Parameters
        ----------
        mag: float
            Absolute magnitude threshold used for sample selection.
        z: float, optional
            Redshift (average) used for sample selection. This is not used.
        cosmology: Cosmology
            Cosmology model to use for other calculations.
        overdensity: float, None
            Halo over-density value to use. 

        """
        #---------------------------------------------------------------
        #            Mag  :   log(Mmin), sigma, log(M0), log(M1), alpha
        #---------------------------------------------------------------
        paramList = {-19.0: [ 11.64    , 0.32 , 12.02  , 12.57  , 0.89 ],  
                     -19.5: [ 11.83    , 0.30 , 11.53  , 13.02  , 0.97 ],  
                     -20.0: [ 12.07    , 0.37 ,  9.32  , 13.27  , 1.08 ],  
                     -20.5: [ 12.63    , 0.82 ,  8.58  , 13.56  , 1.27 ],}
        # interpolate to the nearest value 
        params = paramList[ min( paramList, key = lambda __m: abs(__m - mag) ) ]
        return cls( *params, cosmology, overdensity, )
    
    @classmethod
    def sdss(cls, 
              mag: float, 
              z: float = None, 
              cosmology: Cosmology | None = None, 
              overdensity: float | None = None, ) -> 'Zheng07':
        r"""
        Create a halo model using a pre-defined set of parameters, given in Table 1 of Zheng (2007), 
        for SDSS. Nearest neighbour interpolation is used for values not in the table.

        Parameters
        ----------
        mag: float
            Absolute magnitude threshold used for sample selection.
        z: float, optional
            Redshift (average) used for sample selection. This is not used.
        cosmology: Cosmology
            Cosmology model to use for other calculations.
        overdensity: float, None
            Halo over-density value to use. 

        """
        #---------------------------------------------------------------
        #            Mag  :   log(Mmin), sigma, log(M0), log(M1), alpha
        # --------------------------------------------------------------
        paramList = {-18.0: [ 11.53    , 0.25 , 11.20  , 12.40  , 0.83 ],   
                     -18.5: [ 11.46    , 0.24 , 10.59  , 12.68  , 0.97 ],  
                     -19.0: [ 11.60    , 0.26 , 11.49  , 12.83  , 1.02 ],  
                     -19.5: [ 11.75    , 0.28 , 11.69  , 13.01  , 1.06 ],  
                     -20.0: [ 12.02    , 0.26 , 11.38  , 13.31  , 1.06 ],  
                     -20.5: [ 12.30    , 0.21 , 11.84  , 13.58  , 1.12 ],  
                     -21.0: [ 12.79    , 0.39 , 11.92  , 13.94  , 1.15 ],  
                     -21.5: [ 13.38    , 0.51 , 13.94  , 13.91  , 1.04 ],  
                     -22.0: [ 14.22    , 0.77 , 14.00  , 14.69  , 0.87 ],}
        # interpolate to the nearest value 
        params = paramList[ min( paramList, key = lambda __m: abs(__m - mag) ) ]
        return cls( *params, cosmology, overdensity )
    
    @classmethod
    def harikane22(cls, 
                   mag: float, 
                   z: float, 
                   cosmology: Cosmology, 
                   overdensity: float | None = None, ) -> 'Zheng07':
        r"""
        Create a halo model using a pre-defined set of parameters, given in Table 8 of Harikane (2022). 
        Nearest neighbour interpolation is used for values not in the table.

        Parameters
        ----------
        mag: float
            Absolute magnitude threshold used for sample selection.
        z: float
            Redshift (average) used for sample selection.
        cosmology: Cosmology
            Cosmology model to use for other calculations.
        overdensity: float, None
            Halo over-density value to use. 

        """
        #----------------------------------------------
        #            z  :  mag  :   log(Mmin), log(M1)
        #----------------------------------------------
        paramList = {1.7: {-20.5: [ 12.46    , 14.18 ],
                           -20.0: [ 12.09    , 13.47 ],
                           -19.5: [ 11.79    , 12.86 ],
                           -19.0: [ 11.55    , 12.48 ],
                           -18.5: [ 11.33    , 12.28 ],
                           -18.0: [ 11.16    , 12.08 ],}, 
                     2.2: {-21.0: [ 12.72    , 15.91 ],
                           -20.5: [ 12.30    , 13.92 ],
                           -20.0: [ 11.95    , 13.23 ],
                           -19.5: [ 11.68    , 12.62 ],
                           -19.0: [ 11.45    , 12.23 ],
                           -18.5: [ 11.26    , 11.94 ],}, 
                     2.9: {-21.5: [ 12.55    , 15.39 ],
                           -21.0: [ 12.19    , 13.80 ],
                           -20.5: [ 11.92    , 13.12 ],
                           -20.0: [ 11.71    , 12.55 ],
                           -19.5: [ 11.55    , 12.20 ],
                           -19.0: [ 11.36    , 11.84 ],},
                     3.8: {-22.5: [ 13.08    , 15.25 ],
                           -22.0: [ 12.71    , 14.80 ],
                           -21.5: [ 12.32    , 13.96 ],
                           -21.0: [ 11.98    , 13.23 ],
                           -20.5: [ 11.66    , 12.24 ],
                           -20.0: [ 11.48    , 11.94 ],},
                     4.9: {-22.9: [ 12.95    , 16.65 ],
                           -22.4: [ 12.60    , 15.70 ],
                           -21.9: [ 12.29    , 14.63 ],
                           -21.4: [ 12.00    , 13.45 ],
                           -20.9: [ 11.76    , 12.57 ],
                           -20.4: [ 11.57    , 11.86 ],},
                     5.9: {-22.2: [ 12.33    , 14.67 ],
                           -21.7: [ 12.09    , 13.73 ],
                           -21.2: [ 11.78    , 12.93 ],},}
        # select parameters for nearest z
        paramList = paramList[ min( paramList, key = lambda __z: abs(__z - z) ) ]
        # interpolate to the nearest magnitude value 
        params = paramList[ min( paramList, key = lambda __m: abs(__m - mag) ) ]
        # convert from Msun to Msun/h units
        params = np.add( params, np.log10(cosmology.h) )
        # add other parameters:
        #--------------------------------------------------------------------------
        #          log(Mmin), sigma = 0.2*sqrt(2), log(M0)       , log(M1)  , alpha
        #--------------------------------------------------------------------------
        params = [ params[0], 0.28284271247461906, -0.5*params[0], params[1], 1.0 ]
        return cls( *params, cosmology, overdensity, )

    def centralCount(self, m: Any) -> Any:
        res = ( np.log10(m) - self.logm_min ) / self.sigma_logm
        res = 0.5 * ( erf(res) + 1. )
        return res 
    
    def satelliteFraction(self, m: Any) -> float:
        m0, m1 = 10**self.logm0, 10**self.logm1
        res    = np.divide( np.subtract( m, m0 ), m1 )**self.alpha
        return res
