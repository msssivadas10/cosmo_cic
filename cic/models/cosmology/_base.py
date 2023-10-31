#!/usr/bin/python3

import numpy as np
import warnings
from scipy.integrate import simpson
from scipy.optimize import brentq
from typing import Any
from ._helpers import _ModelsTable, _InterpolationTables, _CosmologySettings
from ..power_spectrum import PowerSpectrum, WindowFunction, builtinWindows
from ..halos import MassFunction, HaloBias
from ..utils import randomString
from ..utils.constants import *


class CosmologyError(Exception):
    r"""
    Base class of exceptions raised in cosmology related calculations.
    """
    ...


class Cosmology:
    r"""
    A Lambda CDM cosmology class.
    """

    __slots__ = ('h', 
                 'Om0', 
                 'Ob0', 
                 'Ode0', 
                 'Onu0', 
                 'Ok0', 
                 'Nnu', 
                 'ns', 
                 'sigma8', 
                 'Tcmb0', 
                 'name', 
                 'settings',
                 '_psnorm', 
                 '_model',
                 '_interp' )
    
    def __init__(self, 
                 h: float, 
                 Om0: float, 
                 Ob0: float, 
                 Ode0: float = None, 
                 Onu0:float = 0.0, 
                 Nnu: float = 3.0, 
                 ns: float = 1.0, 
                 sigma8: float = None, 
                 Tcmb0: float = 2.725, 
                 name: str = None,    ) -> None:
        
        #
        # checking and setting model parameters:
        #
        if h <= 0:
            raise CosmologyError("hubble parameter must be positive")
        if Om0 < 0:
            raise CosmologyError("matter density must be non-negative")
        if Ob0 < 0:
            raise CosmologyError("baryon density must be non-negative")
        if Onu0 < 0:
            raise CosmologyError("neutrino density must be non-negative")
        if Nnu <= 0:
            raise CosmologyError("neutrino number must be positive")
        if Ob0 + Onu0 > Om0:
            raise CosmologyError("sum of baryon and neutrino density must be less than matter density")
        if Ode0 is None:
            Ode0, Ok0 = 1 - Om0, 0.
        elif Ode0 < 0:
            raise CosmologyError("dark energy density must be non-negative")
        else:
            Ok0  = 1 - Om0 - Ode0
        if sigma8 is None:
            sigma8 = 1.0
        elif sigma8 < 0.:
            raise CosmologyError("value of sigma8 parameter must be positive")
        if Tcmb0 <= 0:
            return CosmologyError("CMB temperature must be positive")
        
        self.h      = h      # hubble parameter in 100 km/sec/Mpc
        self.Om0    = Om0    # total matter (baryon + cdm + neutrino) density
        self.Ob0    = Ob0    # baryon density
        self.Onu0   = Onu0   # massive nuetrino density
        self.Nnu    = Nnu    # number of massive neutrinos
        self.Ode0   = Ode0   # dark energy density
        self.Ok0    = Ok0    # curvature energy density 
        self.ns     = ns     # initial power spectrum index
        self.sigma8 = sigma8 # matter variance smoothed at 8 Mpc/h scale
        self.Tcmb0  = Tcmb0 # cosmic microwave background (CMB) temperature in K 


        # set a name for this cosmology model
        if name is None:
            self.name = '_'.join([ 'Cosmology', randomString(16) ])
        else:
            if not isinstance(name, str):
                raise TypeError( "name must be 'str' or None" )
            self.name = name 


        self._psnorm  = 1. # power spectrum noramlization factor
        self._model   = _ModelsTable()
        self._interp  = _InterpolationTables()
        self.settings = _CosmologySettings()
        return
    
    @property
    def powerSpectrumNorm(self) -> float: return self._psnorm

    @property 
    def flat(self) -> bool: return abs(self.Ok0) < EPS

    @property
    def K(self) -> float: return np.sqrt( abs( self.Ok0 ) ) / ( SPEED_OF_LIGHT_KMPS * 0.01 )

    def __repr__(self) -> str:
        r"""
        String representation of the object.
        """
        
        attrs    = ['name', 'h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Nnu', 'ns', 'sigma8', 'Tcmb0']
        data_str = ', '.join( map(lambda __x: f'{__x}={ getattr(self, __x) }', attrs ) )
        return f"Cosmology(%s)" % data_str
    
    def set(self, 
            power_spectrum: str | PowerSpectrum = None, 
            mass_function: str | MassFunction = None, 
            halo_bias: str | HaloBias = None) -> None:
        r"""
        Set model for quantities like power spectrum, mass function etc.
        """
        
        if power_spectrum is not None: 
            self._model.power_spectrum = power_spectrum
        if mass_function is not None:
            self._model.mass_function = mass_function
        if halo_bias is not None:
            self._model.halo_bias = halo_bias
        return
     
    def createInterpolationTables(self) -> None:
        r"""
        Create interpolation tables for fast calculations.
        """

        lnzap1 = np.log( self.settings.zInterpMin + 1 )
        lnzbp1 = np.log( self.settings.zInterpMax + 1 )
        lnka   = np.log( self.settings.kInterpMin )
        lnkb   = np.log( self.settings.kInterpMax )
        lnra   = np.log( self.settings.rInterpMin )
        lnrb   = np.log( self.settings.rInterpMax )

        # growth factor
        self._interp.create_lnDplus(self.dplus, 
                                    lnzap1, lnzbp1, self.settings.zInterpPoints,
                                    nu = 0, log = True, exact = True, )
        
        # matter power spectrum
        self._interp.create_lnPowerSpectrum(self.matterPowerSpectrum, 
                                            lnka,   lnkb,   self.settings.kInterpPoints, # k-grid
                                            lnzap1, lnzbp1, self.settings.zInterpPoints, # z-grid
                                            nu = 0, grid = True, log = True, exact = True, normalize = False, )
        
        # matter power spectrum
        self._interp.create_lnVariance(self.matterVariance, 
                                       lnra,   lnrb,   self.settings.rInterpPoints, # r-grid
                                       lnzap1, lnzbp1, self.settings.zInterpPoints, # z-grids
                                       nu = 0, grid = True, log = True, exact = True, normalize = False, )
        
        return
    
    def fde(self, 
            lnzp1: Any, 
            der: bool = False) -> Any:
        r"""
        Return the redshift evolution of dark energy density. 
        """

        if der:
            return np.zeros_like(lnzp1, dtype = 'float')
        return np.ones_like(lnzp1, dtype = 'float')
    
    def lnE2(self, 
             lnzp1: Any, 
             der: bool = False ) -> float:
        r"""
        Calculate the redshift evolution of hubble parameter.
        """

        zp1 = np.exp( lnzp1 )

        res1 = self.Om0 * zp1**3 
        res2 = 0
        if der:
            res2 = 3. * res1 

        if not self.flat:
            tmp  = self.Ok0 * zp1**2
            res1 = res1 + tmp
            if der:
                res2 = res2 + 2 * tmp

        res1 = res1 + self.Ode0 * self.fde(lnzp1, 0)
        if der:
            res2 = res2 + self.Ode0 * self.fde(lnzp1, 1)

        return 0.5 * res2 / res1 if der else np.log( res1 )
    
    def E(self, z: Any) -> float:
        r"""
        Calculate the redshift evolution of hubble parameter.
        """

        lnzp1 = np.log( np.add(z, 1) )
        return np.exp( 0.5 * self.lnE2( lnzp1 ) )
    
    def Om(self, z: Any) -> float:
        r"""
        Calculate the evolution of matter density.
        """

        zp1 = np.add(z, 1)
        return self.Om0 * zp1**3 /self.E(z)**2
    
    def Ode(self, z: Any) -> float:
        r"""
        Calculate the evolution of dark-energy density.
        """

        lnzp1 = np.log( np.add(z, 1) )
        return self.Ode0 * self.fde(lnzp1) / self.E(z)**2
    
    def matterDensity(self, z: Any) -> float:
        r"""
        Calculate the matter density.
        """

        zp1 = np.asfarray(z) + 1
        return  RHO_CRIT0_ASTRO * self.Om0 * zp1**3
    
    def criticalDensity(self, z: Any) -> float:
        r"""
        Calculate the critical density of the universe.
        """

        return RHO_CRIT0_ASTRO * self.E(z)**2
    
    #
    # Distance calculations
    #
    
    def comovingDistance(self, 
                         z: Any,
                         nu: int = 0,
                         log: bool = False, 
                         exact: bool = False, ) -> float:
        r"""
        Calculate the comoving distance corresponding to redshift z.
        """

        lnzp1 = np.log( np.asfarray(z) + 1 )

        x, dx = np.linspace(0.0, lnzp1, self.settings.zIntegralPoints, retstep = True, axis = -1) 
        
        # function to integrate
        res = np.exp( x - 0.5 * self.lnE2( x ) )

        # logspace integration
        res = simpson( res, dx = dx, axis = -1 )
        return res * SPEED_OF_LIGHT_KMPS * 0.01 # Mpc/h

    def luminocityDistance(self, 
                           z: Any, 
                           nu: int = 0,
                           log: bool = False, 
                           exact: bool = False, ) -> float:
        r"""
        Calculate the luminocity distance corresponding to redshift z.
        """

        r = self.comovingDistance( z )

        if self.Ok0:
            K = self.K
            if self.Ok0 < 0.0:
                r = np.sin( K*r ) / K # k > 0 : closed/spherical
            else:
                r = np.sinh( K*r ) / K    # k < 0 : open / hyperbolic
        
        return r * ( 1 + np.asfarray( z ) ) # Mpc/h

    def angularDiameterDistance(self, 
                                z: Any, 
                                nu: int = 0,
                                log: bool = False, 
                                exact: bool = False, ) -> float:
        r"""
        Calculate the angular diameter distance corresponding to redshift z.
        """

        return self.luminocityDistance( z ) / ( 1 + np.asfarray( z ) )**2 # Mpc/h
    
    def angularSize(self, 
                    value: Any, 
                    z: Any, 
                    inverse: bool = False ) -> float:
        r"""
        Convert size from physical (Mpc/h) to angular (arcsec) units.
        """

        value = np.asfarray(value)
        dist  = self.angularDiameterDistance( z )

        if inverse: # angular to physical 
            return value * dist * np.pi / 180. / 3600.
        
        # physical to angular
        return value / dist * 180. / np.pi * 3600.
            

    #
    # Linear growth calculations
    #
    
    def dplus(self, 
              z: Any, 
              nu: int = 0,
              log: bool = False, 
              exact: bool = False, ) -> Any:
        r"""
        Calculate the linear growth factor or its logarithmic derivative.
        """


        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")

        lnzp1 = np.asfarray( z ) if log else np.log( np.add(z, 1) )

        if not exact: 
            if self._interp.lnDplus is not None:
                res = self._interp.lnDplus( lnzp1, nu )
                return np.exp( res ) if not log and nu == 0 else res
            
            # get the exact result 
            warnings.warn("growth factor interpolation table is not created, using exact calculations")
        
        #
        # exact calculations
        #

        INF   = self.settings.zInfinity
        x, dx = np.linspace(lnzp1, np.log(INF + 1), self.settings.zIntegralPoints, retstep = True, axis = -1) 
        
        # function to integrate
        res = np.exp( 2 * x - 1.5 * self.lnE2( x ) )

        # logspace integration
        res = simpson( res, dx = dx, axis = -1 )

        if nu:
            res  = self.lnE2( lnzp1, der = True ) - np.exp( 2 * lnzp1 - 1.5 * self.lnE2( lnzp1 ) ) / res
        else:        
            res = 2.5 * self.Om0 * np.exp( 0.5 * self.lnE2( lnzp1 ) ) * res
        return np.log( res ) if log and nu == 0 else res
    
    #
    # matter power spectrum, variance, correlation etc
    #
    
    def matterPowerSpectrum(self, 
                            k: Any, 
                            z: Any = 0., 
                            nu: int = 0, 
                            grid: bool = False, 
                            log: bool = False, 
                            exact: bool = False, 
                            normalize: bool = True, ) -> Any:
        r"""
        Calculate the linear matter matter power spectrum.
        """

        if self._model.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked to this cosmology")
        
        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")
        
        lnk   = np.asfarray( k ) if log else np.log( k )
        lnzp1 = np.asfarray( z ) if log else np.log( np.add(z, 1) )

        if not exact: 
            if self._interp.lnPowerSpectrum is not None:
                res = self._interp.lnPowerSpectrum( lnk, lnzp1, nu, 0, grid )
                if nu == 0 and normalize: 
                    res = res + np.log( self.powerSpectrumNorm )
                return np.exp( res ) if not log and nu == 0 else res
            
            # get the exact result 
            warnings.warn("power spectrum interpolation table is not created, using exact calculations")
        
        #
        # exact calculations
        #
        
        res = self._model.power_spectrum(self, 
                                         lnk, 
                                         lnzp1, 
                                         der   = bool(nu), 
                                         grid  = grid, )
        if nu == 0 and normalize:
            res = res + np.log( self.powerSpectrumNorm )
        return np.exp( res ) if not log and nu == 0 else res
    
    def matterVariance(self, 
                       r: Any, 
                       z: Any = 0., 
                       nu: int = 0, 
                       grid: bool = False, 
                       log: bool = False, 
                       exact: bool = False, 
                       normalize: bool = True) -> Any:
        r"""
        Calculate the the linear matter variance or its derivatives.
        """

        if nu not in range(3):
            raise ValueError( "nu can only be 0, 1 or 2" )        

        if self._model.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked to this cosmology")
        
        lnr   = np.asfarray( r ) if log else np.log( r )
        lnzp1 = np.asfarray( z ) if log else np.log( np.add(z, 1) )
        
        if not exact: 
            if self._interp.lnVariance is not None:
                res = self._interp.lnVariance( lnr, lnzp1, nu, 0, grid )
                if nu == 0 and normalize: 
                    res = res + np.log( self.powerSpectrumNorm )
                return np.exp( res ) if not log and nu == 0 else res
            
            # get the exact result 
            warnings.warn("variance interpolation table is not created, using exact calculations")
        
        #
        # exact calculations
        #

        win = self.settings.smoothWindow
        if isinstance( win, str ):
            if win not in builtinWindows:
                raise ValueError("window function '%s' is not available" % win)
            win = builtinWindows[win]
        
        res = self._model.power_spectrum.matterVariance(self, 
                                                        lnr, 
                                                        lnzp1, 
                                                        nu     = nu, 
                                                        window = win, 
                                                        ka     = self.settings.kZero,
                                                        kb     = self.settings.kInfinity,
                                                        pts    = self.settings.kIntegralPoints, 
                                                        grid   = grid, )
        
        if nu == 0 and normalize:
            res = res + np.log( self.powerSpectrumNorm )
        return np.exp( res ) if not log and nu == 0 else res
    
    def normalizePowerSpectrum(self, reset: bool = False) -> None:
        r"""
        Normalize the matter power spectrum useing `sigma8` values.
        """
        
        self._psnorm = 1.        
        if reset:
            return 
        
        calculatedValue = self.matterVariance( 8.0, exact = not self.settings.useInterpolation )
        observedValue   = self.sigma8**2
        self._psnorm    = observedValue / calculatedValue
        return 
    
    #
    # halo mass-function and bias
    #

    def lagrangianM(self, 
                    r: Any, 
                    overdensity: float = None, ) -> Any:
        r"""
        Mass corresponding to lograngian radius r.
        """

        rho = self.Om0 * RHO_CRIT0_ASTRO 
        if overdensity is not None:
            rho = rho * overdensity

        m = 4 * np.pi / 3 * ( np.asfarray( r )**3 ) * rho
        return m

    def lagrangianR(self, 
                    m: Any, 
                    overdensity: float = None, ) -> Any:
        r"""
        Lagrangian radius corresponding to mass m.
        """

        rho = self.Om0 * RHO_CRIT0_ASTRO
        if overdensity is not None:
            rho = rho * overdensity

        r = np.cbrt( 0.75 * np.asfarray( m ) / np.pi / rho )
        return r
    
    def collapseDensity(self, z: Any) -> Any:
        r"""
        Retrun the critical density for spherical collapse.
        """
        
        res = np.ones_like( z ) * DELTA_SC

        # corrections for specific cosmologies
        if abs( self.Om0 - 1. ) < EPS and abs( self.Ode0 ) < EPS:
            res = self.Om( z )**0.0185
        elif abs( self.Om0 + self.Ode0 - 1. ) < EPS:
            res = self.Om( z )**0.0055

        return res * DELTA_SC
    
    def peakHeight(self, 
                   r: Any, 
                   z: Any = 0., 
                   grid: bool = False, 
                   exact: bool = False, 
                   normalize: bool = True, ) -> Any:
        r"""
        Calculate the peak height of an overdensity.
        """

        delta_c = self.collapseDensity( z )
        sigma2  = self.matterVariance( r, z, grid = grid, exact = exact, normalize = normalize )
        return delta_c / np.sqrt( sigma2 )

    def massFunction(self, 
                     m: Any, 
                     z: float = 0., 
                     overdensity: Any = 200, 
                     retval: str = 'dndlnm', 
                     grid: bool = False,   ) -> Any:
        r"""
        Calculate the halo mass-function.
        """

        if self._model.mass_function is None:
            raise CosmologyError("no mass-function model is linked to this cosmology")
        
        res = self._model.mass_function(self,
                                        m = m, 
                                        z = z, 
                                        overdensity = overdensity, 
                                        retval = retval, 
                                        grid = grid, )
        return res
    
    def biasFunction(self, 
                     m: Any, 
                     z: float = 0., 
                     overdensity: Any = 200, 
                     grid: bool = False,   ) -> Any:
        r"""
        Calculate the halo bias function.
        """

        if self._model.halo_bias is None:
            raise CosmologyError("no bias model is linked to this cosmology")
        
        res = self._model.halo_bias(self, 
                                    m = m, 
                                    z = z, 
                                    overdensity = overdensity, 
                                    grid = grid, )
        return res
    
    def collapseRadius(self, 
                       z: Any, 
                       exact: bool = False, 
                       **kwargs,        ) -> Any:
        r"""
        Calculate the collapse radius of a halo at redshift z.
        """

        lnra, lnrb = np.log(self.settings.rInterpMin), np.log(self.settings.rInterpMax)

        # cost function: function to find root
        def cost(lnr: float, z: float) -> float:
            y = self.peakHeight(np.exp( lnr ), 
                                z, 
                                exact = exact, ) - 1.
            return y
        
        # vectorised function returning collapse radius
        @np.vectorize
        def result(z: float) -> float: 
            lnr = brentq(cost, lnra, lnrb, args = (z, ), disp = False, **kwargs)
            return np.exp( lnr )

        return result(z)
    
    def collapseRedshift(self, 
                         r: Any, 
                         exact: bool = False, 
                         **kwargs,      ) -> Any:
        
        r"""
        Calculate the collapse redshift of a halo of radius r.
        """

        # cost function: function to find root
        def cost(t: float, r: float) -> float:
            y = self.peakHeight(r, 
                                z = t**-1 - 1, 
                                exact = exact, ) - 1.
            return y
        
        # vectorised function returning collapse z
        @np.vectorize
        def result(r: float) -> float: 
            ta, tb = 1e-08, 1.
            ya, yb = cost(ta, r), cost(tb, r)

            # if radius is too large (cost function is +ve at both ends), then the halos are 
            # collapsed at a redshift later than 0 (return -1)
            if np.sign(ya) == 1 and np.sign(yb) == 1:
                return -1.
            # if radius is too small (cost function is +ve at both ends), then the halos are 
            # collapsed at a redshift z = inf
            if np.sign(ya) != 1 and np.sign(yb) != 1:
                return np.inf
            # solve for collapse time
            t = brentq(cost, ta, tb, args = (r, ), disp = False, **kwargs)
            return t**-1 - 1

        return result(r)

