#!/usr/bin/python3

import numpy as np
import warnings
from scipy.integrate import simpson
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Any, Callable

from . import halos
from .power_spectrum import linear_models as cps
from .utils import typestr
from .utils import randomString, Interpolator1D, Interpolator2D
from .utils.constants import *

    

class CosmologyError(Exception):
    r"""
    Base class of exceptions raised in cosmology related calculations.
    """
    ...

@dataclass
class _CosmologySettings:
    r"""
    General setting for cosmology and related calculations. 
    """

    useInterpolation: bool = True # for faster calculations

    # redshift / z
    zInterpPoints: int   = 21       # number of interpolation points for z
    zInterpMin: float    = 0.0      # minimum z for interpolation
    zInterpMax: float    = 10.0     # maximum z for interpolation
    zIntegralPoints: int = 1001     # number of points for z integration
    zZero: float         = 0.0      # zero value for z
    zInfinity: float     = 1e+08    # infinte value for z

    # wavenumber / k
    kInterpPoints: int   = 101      # number of interpolation points for k
    kInterpMin: float    = 1e-08    # minimum k for interpolation
    kInterpMax: float    = 1e+08    # maximum k for interpolation
    kIntegralPoints: int = 1001     # number of points for k integration
    kZero: float         = 1e-08    # zero value for k
    kInfinity: float     = 1e+08    # infinte value k
    smoothWindow: str    = 'tophat' # window used for smoothing the density 

    # mass / m
    mInterpPoints: int   = 101      # number of interpolation points for m
    mInterpMin: float    = 1e+06    # minimum m for interpolation
    mInterpMax: float    = 1e+16    # maximum m for interpolation
    mIntegralPoints: int = 1001     # number of points for m integrations
    mZero: float         = 1e-08    # zero value for m
    mInfinity: float     = 1e+20    # infinite value for m

    # radius / r
    rInterpPoints: int   = 101      # number of interpolation points for r
    rInterpMin: float    = 1e-03    # minimum r for interpolation
    rInterpMax: float    = 1e+03    # maximum r for interpolation
    rIntegralPoints: int = 1001     # number of points for r integrations
    rZero: float         = 1e-04    # zero value for r
    rInfinity: float     = 1e+04    # infinite value for r


@dataclass
class _InterpolationTables:
    lnDistance: Interpolator2D      = None # ln of comoving distance: args = ln(z+1)
    lnDplus: Interpolator1D         = None # ln of linear growth: args = ln(z+1)
    lnPowerSpectrum: Interpolator2D = None # ln of power spectrum: args = log(k), ln(z+1)
    lnVariance: Interpolator2D      = None # ln of variance: args = ln(r), ln(z+1)


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
                 'powerSpectrumNorm', 
                 'name', 
                 'settings',
                 '_model_power_spectrum',
                 '_model_mass_function',
                 '_model_halo_bias',
                 '_interp'    )
    
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
        
        # hubble parameter in 100 km/sec/Mpc
        if h <= 0:
            raise CosmologyError("hubble parameter must be positive")
        self.h = h 

        # total matter (baryon + cdm + neutrino) density
        if Om0 < 0:
            raise CosmologyError("matter density must be non-negative")
        self.Om0 = Om0 

        # baryon density
        if Ob0 < 0:
            raise CosmologyError("baryon density must be non-negative")
        self.Ob0 = Ob0 

        # massive nuetrino density
        if Onu0 < 0:
            raise CosmologyError("neutrino density must be non-negative")
        if Nnu <= 0:
            raise CosmologyError("neutrino number must be positive")
        self.Onu0, self.Nnu = Onu0, Nnu

        if Ob0 + Onu0 > Om0:
            raise CosmologyError("sum of baryon and neutrino density must be less than matter density")

        # dark-energy (cosmological constant) and curvature density
        if Ode0 is None:
            self.Ode0 = 1 - self.Om0
            self.Ok0  = 0.
        else:
            assert Ode0 >= 0
            self.Ode0 = Ode0
            self.Ok0  = 1 - self.Om0 - self.Ode0


        self.ns = ns # initial power spectrum index

        if sigma8 is not None:
            if sigma8 < 0.:
                raise CosmologyError("sigma8 must be positive")
        self.sigma8 = sigma8 # matter variance smoothed at 8 Mpc/h scale
        
        # cmb temperature 
        if Tcmb0 <= 0:
            return CosmologyError("CMB temperature must be positive")
        self.Tcmb0  = Tcmb0 

        self.powerSpectrumNorm = 1. # power spectrum noramlization factor

        # name for this cosmology model
        if name is None:
            name = '_'.join([ 'Cosmology', randomString(16) ])
        assert isinstance(name, str), "name must be 'str' or None"
        self.name = name 

        # models:
        self._model_power_spectrum: cps.PowerSpectrum = None
        self._model_mass_function: halos.MassFunction = None
        self._model_halo_bias: halos.HaloBias         = None
        
        self.settings = _CosmologySettings()
        self._interp  = _InterpolationTables()
        return
    
    def set(self, 
            power_spectrum: str | cps.PowerSpectrum = None, 
            mass_function: str | halos.MassFunction = None, 
            halo_bias: str | halos.HaloBias = None) -> None:
        r"""
        Set model for quantities like power spectrum, mass function etc.
        """

        if power_spectrum is not None:
            if isinstance( power_spectrum, str ):
                if power_spectrum not in cps.builtinPowerSpectrums:
                    raise CosmologyError(f"power spectrum model '{power_spectrum}' is not available")
                power_spectrum = cps.builtinPowerSpectrums.get( power_spectrum ) 
            if not isinstance( power_spectrum, cps.PowerSpectrum ):
                raise CosmologyError(f"cannot use a '{typestr(power_spectrum)}' object as power spectrum model")
            self._model_power_spectrum = power_spectrum

        if mass_function is not None:
            if isinstance( mass_function, str ):
                if mass_function not in halos.builtinMassfunctions:
                    raise CosmologyError(f"mass function model '{mass_function}' is not available")
                mass_function = halos.builtinMassfunctions.get( mass_function ) 
            if not isinstance( mass_function, halos.MassFunction ):
                raise CosmologyError(f"cannot use a '{typestr(mass_function)}' object as mass function model")
            self._model_mass_function = mass_function

        if halo_bias is not None:
            if isinstance( halo_bias, str ):
                if halo_bias not in halos.builtinLinearBiases:
                    raise CosmologyError(f"halo bias model '{halo_bias}' is not available")
                halo_bias = halos.builtinLinearBiases.get( halo_bias ) 
            if not isinstance( halo_bias, halos.HaloBias ):
                raise CosmologyError(f"cannot use a '{typestr(halo_bias)}' object as halo bias model")
            self._model_halo_bias = halo_bias
            
        return
     
    def createInterpolationTables(self) -> None:

        lnzap1 = np.log( self.settings.zInterpMin + 1 )
        lnzbp1 = np.log( self.settings.zInterpMax + 1 )
        lnka   = np.log( self.settings.kInterpMin )
        lnkb   = np.log( self.settings.kInterpMax )
        lnra   = np.log( self.settings.rInterpMin )
        lnrb   = np.log( self.settings.rInterpMax )

        # growth factor
        self._interp.lnDplus = Interpolator1D(self.dplus, 
                                              xa = lnzap1, 
                                              xb = lnzbp1, 
                                              xpts = self.settings.zInterpPoints,
                                              fkwargs = dict(nu = 0, log = True, exact = True), )
        
        # matter power spectrum
        self._interp.lnPowerSpectrum = Interpolator2D(self.matterPowerSpectrum, 
                                                      xa = lnka, 
                                                      xb = lnkb,
                                                      xpts = self.settings.kInterpPoints,
                                                      ya = lnzap1, 
                                                      yb = lnzbp1, 
                                                      ypts = self.settings.zInterpPoints, 
                                                      fkwargs = dict(nu = 0, 
                                                                     grid = True, 
                                                                     log = True, 
                                                                     exact = True, 
                                                                     normalize = False, ))
        
        # matter power spectrum
        self._interp.lnVariance = Interpolator2D(self.matterVariance, 
                                                 xa = lnra, 
                                                 xb = lnrb,
                                                 xpts = self.settings.rInterpPoints,
                                                 ya = lnzap1, 
                                                 yb = lnzbp1, 
                                                 ypts = self.settings.zInterpPoints, 
                                                 fkwargs = dict(nu = 0, 
                                                                grid = True, 
                                                                log = True, 
                                                                exact = True, 
                                                                normalize = False, ))
        
        return

    def __repr__(self) -> str:
        r"""
        String representation of the object.
        """
        
        attrs    = ['name', 'h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Nnu', 'ns', 'sigma8', 'Tcmb0']
        data_str = ', '.join( map(lambda __x: f'{__x}={ getattr(self, __x) }', attrs ) )
        return f"Cosmology(%s)" % data_str
    
    def lnE2(self, 
             lnzp1: Any, 
             der: bool = False ) -> float:
        r"""
        Calculate the redshift evolution of hubble parameter.
        """

        zp1 = np.exp( lnzp1 )

        res1 = self.Om0 * zp1**3 
        res2 = 3. * res1 if der else 0

        if abs( self.Ok0 ) > EPS:
            tmp  = self.Ok0 * zp1**2
            res1 = res1 + tmp
            res2 = res2 + 2 * tmp if der else 0

        res1 = res1 + self.Ode0

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

        zp1 = np.asfarray(z) + 1.
        return self.Om0 * zp1**3 / self.E(z)**2
    
    def Ode(self, z: Any) -> float:
        r"""
        Calculate the evolution of dark-energy density.
        """

        return self.Ode0 / self.E(z)**2
    
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
    
    def comovingDistance(self, z: Any) -> float:
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

    def luminocityDistance(self, z: Any) -> float:
        r"""
        Calculate the luminocity distance corresponding to redshift z.
        """

        r = self.comovingDistance( z )

        if self.Ok0:
            K = np.sqrt( abs( self.Ok0 ) ) / ( SPEED_OF_LIGHT_KMPS * 0.01 )
            if self.Ok0 < 0.0:
                r = np.sin( K*r ) / K # k > 0 : closed/spherical
            else:
                r = np.sinh( K*r ) / K    # k < 0 : open / hyperbolic
        
        return r * ( 1 + np.asfarray( z ) ) # Mpc/h

    def angularDiameterDistance(self, z: Any) -> float:
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

        if inverse: # angular to physical 
            return value * self.angularDiameterDistance( z ) * np.pi / 180. / 3600.
        
        # physical to angular
        return value / self.angularDiameterDistance( z ) * 180. / np.pi * 3600.
            

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

        if self._model_power_spectrum is None:
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
        
        res = self._model_power_spectrum(self, 
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

        if self._model_power_spectrum is None:
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
        
        res = self._model_power_spectrum.matterVariance(self, 
                                                        lnr, 
                                                        lnzp1, 
                                                        nu     = nu, 
                                                        window = self.settings.smoothWindow, 
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
        
        self.powerSpectrumNorm = 1.        
        if reset:
            return 
        
        calculatedValue        = self.matterVariance( 8.0, exact = not self.settings.useInterpolation )
        observedValue          = self.sigma8**2
        self.powerSpectrumNorm = observedValue / calculatedValue
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

        if self._model_mass_function is None:
            raise CosmologyError("no mass-function model is linked to this cosmology")
        
        res = self._model_mass_function(self,
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

        if self._model_halo_bias is None:
            raise CosmologyError("no bias model is linked to this cosmology")
        
        res = self._model_halo_bias(self, 
                                    m = m, 
                                    z = z, 
                                    overdensity = overdensity, 
                                    grid = grid, )
        return res
    
    def collapseRadius(self, 
                       z: Any, 
                       exact: bool = False, 
                       ra: float = 1e-03, 
                       rb: float = 1e+03,
                       window: str = 'tophat', 
                       integral_pts: int = 10001, 
                       **kwargs,                ) -> Any:
        r"""
        Calculate the collapse radius of a halo at redshift z.
        """

        lnra, lnrb = np.log(ra), np.log(rb)

        # cost function: function to find root
        def cost(lnr: float, z: float) -> float:
            y = self.peakHeight(np.exp( lnr ), 
                                z, 
                                exact = exact, 
                                window = window, 
                                integral_pts = integral_pts ).flatten() - 1.
            return y

        z   = np.ravel( z )
        res = [ brentq(cost, lnra, lnrb, args = (zi, ), disp = False, **kwargs) for zi in z ]
        return np.exp( res )
    
    def collapseRedshift(self, 
                         r: Any, 
                         exact: bool = False, 
                         ta: float = 1e-08, 
                         tb: float = 1.0,
                         window: str = 'tophat', 
                         integral_pts: int = 10001, 
                         **kwargs,                ) -> Any:
        
        r"""
        Calculate the collapse redshift of a halo of radius r.
        """

        # ta, tb = (za  + 1)**-1, (zb + 1)**-1

        # cost function: function to find root
        def cost(t: float, r: float) -> float:
            y = self.peakHeight(r, 
                                z = t**-1 - 1, 
                                exact = exact, 
                                window = window, 
                                integral_pts = integral_pts ).flatten() - 1.
            return y
        

        r   = np.ravel( r )
        res = np.zeros_like( r, dtype = 'float' )
        
        ya, yb = cost(ta, r), cost(tb, r)

        # if radius is too large (cost function is -ve at both ends), then the halos are 
        # collapsed at a redshift later than the given za and the corresponding return
        # value is set to -1
        m      = np.logical_and( np.sign(ya) == 1, np.sign(yb) == 1 )
        res[m] = -1 

        # if radius is too small (cost function is +ve at both ends), then the halos are 
        # collapsed at a redshift z = inf        
        m      = np.logical_and( np.sign(ya) != 1, np.sign(yb) != 1 )
        res[m] = np.inf

        m      = ( np.sign(ya) != np.sign(yb) )
        res[m] = [ brentq(cost, ta, tb, args = (ri, ), disp = False, **kwargs) for ri in r[m] ]
        res[m] = res[m]**-1 - 1

        return res



#
# some ready-to-use cosmology models:
#

builtinCosmology = {}

# cosmology with parameters from Plank et al (2018)
plank18 =  Cosmology(h = 0.6790, Om0 = 0.3065, Ob0 = 0.0483, Ode0 = 0.6935, sigma8 = 0.8154, ns = 0.9681, Tcmb0 = 2.7255, name = 'plank18')
plank18.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['plank18'] = plank18

# cosmology with parameters from Plank et al (2015)
plank15 =  Cosmology(h = 0.6736, Om0 = 0.3153, Ob0 = 0.0493, Ode0 = 0.6947, sigma8 = 0.8111, ns = 0.9649, Tcmb0 = 2.7255, name = 'plank15')
plank15.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['plank15'] = plank15

# cosmology with parameters from WMAP survay
wmap08 =  Cosmology(h = 0.719, Om0 = 0.2581, Ob0 = 0.0441, Ode0 = 0.742, sigma8 = 0.796, ns = 0.963, Tcmb0 = 2.7255, name = 'wmap08')
wmap08.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['wmap08'] = wmap08

# cosmology for millanium simulation
millanium = Cosmology(h = 0.73, Om0 = 0.25, Ob0 = 0.045, sigma8 = 0.9, ns = 1.0, Tcmb0 = 2.7255, name = 'millanium')
millanium.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['millanium'] = millanium


