#!/usr/bin/python3

import re
import numpy as np
import cic.models.power_spectrum as cps
import cic.models.mass_function as cmf
import cic.models.halo_bias as chb
from cic.models.constants import *
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from typing import Any, Callable


def createInterpolatorFromData(values: Any, x: Any, y: Any = None, spline_args: dict = {}):
    
    assert np.ndim(x) == 1, "x must be a 1d array"

    spline_args = {'s': 0, **spline_args} # set default s = 0 

    if y is None: # univariate interpolation

        assert np.ndim(values) == 1, "values must be a 1d array"
        assert np.size(values) == np.size(x), "x and values should have same size"
        return UnivariateSpline( x, values, **spline_args ), np.stack( [x, values], axis = -1 )
    
    # bivariate interpolation
    assert np.ndim(y) == 1, "y must be a 1d array"
    assert np.ndim(values) == 2, "values must be a 2d array"
    assert np.size(values, 0) == np.size(x) and np.size(values, 1) == np.size(y), "values has incorrect shape"
    return RectBivariateSpline( x, y, values, **spline_args ), (x, y, values)


def createInterpolator(func: Callable,
                       xa: float,
                       xb: float, 
                       xpts: int, 
                       ya: float = None,
                       yb: float = None,
                       ypts: int = None,
                       spline_args: dict = {},
                       **kwargs              ):
    
    assert callable(func), "func must be a callable"

    spline_args = {'s': 0, **spline_args} # set default s = 0 

    x = np.linspace( xa, xb, xpts )
    y = None

    if ya is None: 
        values = func( x, **kwargs )
    else:
        y      = np.linspace( ya, yb, ypts )
        values = func( x, y, **kwargs )

    return createInterpolatorFromData( values, x, y, spline_args )
    

###########################################################################################
# Cosmology basic model
###########################################################################################

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
                 'powerSpectrumNorm', 
                 'name', 
                 '_model_power_spectrum',
                 '_model_mass_function',
                 '_model_linear_bias',
                 '_interpolate_distance', 
                 '_interpolate_growth_factor', 
                 '_interpolate_power_spectrum', 
                 '_interpolate_variance', 
                 '_data_distance', 
                 '_data_growth_factor', 
                 '_data_power_spectrum', 
                 '_data_variance',           )
    
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


        self.ns = ns     # power spectrum index

        if sigma8 is not None:
            if sigma8 < 0.:
                raise CosmologyError("sigma8 must be positive")
        self.sigma8 = sigma8 # matter variance smoothed at 8 Mpc/h scale
        
        # cmb temperature 
        if Tcmb0 <= 0:
            return CosmologyError("CMB temperature must be positive")
        self.Tcmb0  = Tcmb0 

        self.powerSpectrumNorm = 1. # power spectrum noramlization factor

        self.name = name # name for this cosmology model

        self._model_power_spectrum = None
        self._model_mass_function  = None
        self._model_linear_bias    = None

        self._interpolate_distance,       self._data_distance       = None, None # comoving distance
        self._interpolate_growth_factor,  self._data_growth_factor  = None, None # linear growth factor
        self._interpolate_power_spectrum, self._data_power_spectrum = None, None # linear matter power spectrum
        self._interpolate_variance,       self._data_variance       = None, None # linear variance

    def set(self, **kwrgs):
        r"""
        Set model for quantities like power spectrum, mass function etc.
        """

        mapping = {
                        'power spectrum': {
                                            'keys'  : { 'power_spectrum', 'ps', 'power' },
                                            'type'  : cps.PowerSpectrum, 
                                            'models': cps.available,
                                            'target': '_model_power_spectrum',
                                          },
                        'mass function': {
                                            'keys'  : { 'mass_function', 'hmf' },
                                            'type'  : cmf.MassFunction, 
                                            'models': cmf.available,
                                            'target': '_model_mass_function',
                                          },
                        'halo bias'    : {
                                            'keys'  : { 'halo_bias', 'linear_bias', 'bias' },
                                            'type'  : chb.HaloBias, 
                                            'models': chb.available,
                                            'target': '_model_linear_bias',
                                          },
                }
        
        for quantity, model in kwrgs.items():

            unknown_quantity = True
            for valid_quantity, details in mapping.items():

                if quantity not in details['keys']:
                    continue

                if isinstance( model, str ):
                    
                    if model not in details['models']:
                        raise CosmologyError(f"{valid_quantity} model '{model}' is not available")

                    model = details['models'].get( model ) 
                
                if not isinstance( model, details['type'] ):
                    __t = re.search( '(?<=\<class \')[\.\w]+(?=\'>)', repr( type( model ) ) ).group(0)
                    raise CosmologyError(f"cannot use a '{__t}' object as {valid_quantity} model")
                
                setattr( self, details['target'], model )
                unknown_quantity = False
            
            if unknown_quantity:
                raise CosmologyError(f"unknown quatity '{quantity}'")
        
        return self

    def createInterpolators(self, 
                            za: float = 0., 
                            zb: float = 7.,
                            zpts: int = 21, 
                            ka: float = 1e-08, 
                            kb: float = 1e+08, 
                            kpts: int = 101, 
                            ra: float = 1e-03, 
                            rb: float = 1e+03, 
                            rpts: int = 101, 
                            z_integral_pts: int = 10001, 
                            k_integral_pts: int = 10001, 
                            window: str = 'tophat'            ):
        r"""
        Create interpolation tables for fast calculations.
        """

        lnzap1, lnzbp1 = np.log(za + 1), np.log(zb + 1)
        lnka, lnkb     = np.log(ka), np.log(kb)
        lnra, lnrb     = np.log(ra), np.log(rb)

        (self._interpolate_distance, 
         self._data_distance       ) = createInterpolator(self._comovingDistance,
                                                          xa = lnzap1, 
                                                          xb = lnzbp1, 
                                                          xpts = zpts, 
                                                          integral_pts = z_integral_pts, )
        
        (self._interpolate_growth_factor, 
         self._data_growth_factor       ) = createInterpolator(self._lndplus,
                                                               xa = lnzap1, 
                                                               xb = lnzbp1, 
                                                               xpts = zpts,
                                                               nu = 0, 
                                                               integral_pts = z_integral_pts, )
        
        (self._interpolate_power_spectrum, 
         self._data_power_spectrum       ) = createInterpolator(self._matterPowerSpectrum, 
                                                                xa = lnka,
                                                                xb = lnkb, 
                                                                xpts = kpts, 
                                                                ya = lnzap1, 
                                                                yb = lnzbp1, 
                                                                ypts = zpts,
                                                                nu = 0, 
                                                                exact = True, )
        
        (self._interpolate_variance, 
         self._data_variance       ) = createInterpolator(self._matterVariance, 
                                                          xa = lnra,
                                                          xb = lnrb, 
                                                          xpts = rpts, 
                                                          ya = lnzap1, 
                                                          yb = lnzbp1, 
                                                          ypts = zpts,
                                                          nu = 0, 
                                                          exact = True, 
                                                          window = window,
                                                          integral_pts = k_integral_pts, )
        
        return self        

    def __repr__(self) -> str:
        r"""
        String representation of the object.
        """
        
        attrs    = ['name', 'h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Nnu', 'ns', 'sigma8', 'Tcmb0']
        data_str = ', '.join( map(lambda __x: f'{__x}={ getattr(self, __x) }', attrs ) )
        return f"Cosmology(%s)" % data_str
    
    def lnE2(self, lnzp1: Any, der: bool = False) -> float:
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
    
    def _comovingDistance(self, lnzp1: Any, integral_pts: int = 10001) -> Any:
        r"""
        Calculate the comoving distance in units of c/H0 Mpc/h.
        """
        
        x, dx = np.linspace(0.0, lnzp1, integral_pts, retstep = True, axis = -1) # := log (z + 1)
        
        # function to integrate
        res = np.exp( x - 0.5 * self.lnE2( x ) )

        # logspace integration
        res = simpson( res, dx = dx, axis = -1 )
        return res
    
    def comovingDistance(self, z: Any, exact: bool = False) -> float:
        r"""
        Calculate the comoving distance corresponding to redshift z.
        """

        lnzp1 = np.log( np.asfarray(z) + 1 )

        if exact or self._interpolate_distance is None:
            res = self._comovingDistance( lnzp1 )
        else:
            res = self._interpolate_distance( lnzp1, nu = 0 )
        return res * SPEED_OF_LIGHT_KMPS * 0.01 # Mpc/h

    def luminocityDistance(self, z: Any, exact: bool = False) -> float:
        r"""
        Calculate the luminocity distance corresponding to redshift z.
        """

        r = self.comovingDistance( z, exact )
        if self.Ok0:
            K = np.sqrt( abs( self.Ok0 ) ) / ( SPEED_OF_LIGHT_KMPS * 0.01 )
            if self.Ok0 < 0.0:
                r = np.sin( K*r ) / K # k > 0 : closed/spherical
            else:
                r = np.sinh( K*r ) / K    # k < 0 : open / hyperbolic
        
        return r * ( 1 + np.asfarray( z ) ) # Mpc/h

    def angularDiameterDistance(self, z: Any, exact: bool = False) -> float:
        r"""
        Calculate the angular diameter distance corresponding to redshift z.
        """

        return self.luminocityDistance(z, exact) / ( 1 + np.asfarray( z ) )**2 # Mpc/h
    
    def angularSize(self, value: Any, z: Any, inverse: bool = False, exact: bool = False) -> float:
        r"""
        Convert size from physical (Mpc/h) to angular (arcsec) units.
        """

        value = np.asfarray(value)

        if inverse: # angular to physical 
            return value * self.angularDiameterDistance( z, exact ) * np.pi / 180. / 3600.
        
        # physical to angular
        return value / self.angularDiameterDistance( z, exact ) * 180. / np.pi * 3600.
            

    #
    # Linear growth calculations
    #
    
    def _lndplus(self, lnzp1: Any, nu: int = 0, integral_pts: int = 10001) -> Any:
        r"""
        Calculate the log of the linear growth factor.
        """

        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")

        x, dx = np.linspace(lnzp1, np.log(INF + 1), integral_pts, retstep = True, axis = -1) 
        
        # function to integrate
        res = np.exp( 2 * x - 1.5 * self.lnE2( x ) )

        # logspace integration
        res = simpson( res, dx = dx, axis = -1 )

        if nu:
            res  = self.lnE2( lnzp1, der = True ) - np.exp( 2 * lnzp1 - 1.5 * self.lnE2( lnzp1 ) ) / res
            return res
        
        res = np.log( 2.5 * self.Om0 ) + 0.5 * self.lnE2( lnzp1 ) +  np.log( res )
        return res
    
    def dplus(self, z: Any, nu: int = 0, exact: bool = False) -> Any:
        r"""
        Calculate the linear growth factor or its logarithmic derivative.
        """

        lnzp1 = np.log( np.asfarray(z) + 1 )

        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")

        if exact or self._interpolate_growth_factor is None:
            res = self._lndplus( lnzp1, nu = nu )
        else:
            res = self._interpolate_growth_factor( lnzp1, nu = nu )
        
        return -res if nu else np.exp( res )
    
    #
    # matter power spectrum, variance, correlation etc
    #

    def _matterPowerSpectrum(self, lnk: Any, lnzp1: Any = 0., nu: int = 0, exact: bool = True) -> Any:
        r"""
        Calculate the log of linear matter matter power spectrum (exact values).
        """

        if self._model_power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked to this cosmology")
        
        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")

        res = self._model_power_spectrum( self, lnk, lnzp1, bool(nu), exact )
        return res
    
    def matterPowerSpectrum(self, k: Any, z: Any = 0., nu: int = 0, exact: bool = False) -> Any:
        r"""
        Calculate the linear matter matter power spectrum.
        """

        lnzp1 = np.log( np.asfarray(z) + 1 )
        lnk   = np.log( k )

        if nu not in (0, 1):
            raise ValueError("nu can only be 0 or 1")

        if exact or self._interpolate_power_spectrum is None:
            res = self._matterPowerSpectrum( lnk, lnzp1, nu, exact )
        else:
            res = self._interpolate_power_spectrum( lnk, lnzp1, dx = nu, dy = 0 )
        
        return res if nu else np.exp( res ) * self.powerSpectrumNorm 
    
    def _matterVariance(self, 
                        lnr: Any, 
                        lnzp1: Any = 0., 
                        nu: int = 0, 
                        exact: bool = True, 
                        window: str = 'tophat', 
                        integral_pts: int = 10001, ) -> Any:
        r"""
        Calculate the log of the linear matter variance or its derivatives.
        """

        if self._model_power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked to this cosmology")
        
        res = self._model_power_spectrum.matterVariance( self, lnr, lnzp1, nu, exact, window, integral_pts )
        return res
    
    def matterVariance(self, 
                       r: Any, 
                       z: Any = 0., 
                       nu: int = 0, 
                       exact: bool = False, 
                       window: str = 'tophat', 
                       integral_pts: int = 10001, ) -> Any:
        r"""
        Calculate the the linear matter variance or its derivatives.
        """

        if nu not in range(3):
            raise ValueError( "nu can only be 0, 1 or 2" )

        lnzp1 = np.log( np.asfarray(z) + 1 )
        lnr   = np.log( r )

        if exact or self._interpolate_variance is None:
            res = self._matterVariance( lnr, lnzp1, nu, exact, window, integral_pts )
        else:
            res = self._interpolate_variance( lnr, lnzp1, dx = nu, dy = 0 )
        
        return res if nu else np.exp( res ) * self.powerSpectrumNorm
    
    def normalizePowerSpectrum(self, reset: bool = False, 
                               exact: bool = False, 
                               window: str = 'tophat', 
                               integral_pts: int = 10001, ) -> None:
        r"""
        Normalize the matter power spectrum useing `sigma8` values.
        """
        
        if reset:
            self.powerSpectrumNorm = 1.
            return 
        
        calculatedValue        = self.matterVariance(8., 
                                                     exact = exact, 
                                                     window = window, 
                                                     integral_pts = integral_pts, )
        observedValue          = self.sigma8**2
        self.powerSpectrumNorm = observedValue / calculatedValue
        return 
    
    #
    # halo mass-function and bias
    #

    def lagrangianM(self, r: Any) -> Any:
        r"""
        Mass corresponding to lograngian radius r.
        """

        rho = self.Om0 * RHO_CRIT0_ASTRO
        r   = np.asfarray( r )
        m   = 4 * np.pi / 3 * ( r**3 ) * rho
        return m

    def lagrangianR(self, m: Any) -> Any:
        r"""
        Lagrangian radius corresponding to mass m.
        """
        rho = self.Om0 * RHO_CRIT0_ASTRO
        m   = np.asfarray( m )
        r   = np.cbrt( 0.75 * m / np.pi / rho )
        return r

    def massFunction(self, 
                     m: Any, 
                     z: float = 0., 
                     overdensity: Any = 200, 
                     retval: str = 'dndlnm', 
                     variance_args: dict = {}, ) -> Any:
        r"""
        Calculate the halo mass-function.
        """

        if retval not in ['f', 'dndm', 'dndlnm', 'full']:
            raise ValueError("invalid value for retval: '%s'" % retval)

        z, m = np.asfarray( z ), np.asfarray( m )

        variance_args = { **dict(exact = False, window = 'tophat', integral_pts = 10001), **variance_args }

        if self._model_mass_function is None:
            raise CosmologyError("no mass-function model is linked to this cosmology")
        
        res = self._model_mass_function( self, m, z, overdensity, retval, variance_args )
        return res
    
    def linearHaloBias(self, 
                       m: Any, 
                       z: float = 0., 
                       overdensity: Any = 200, 
                       variance_args: dict = {}, ) -> Any:
        r"""
        Calculate the linear halo bias function.
        """

        z, m = np.asfarray( z ), np.asfarray( m )

        variance_args = { **dict(exact = False, window = 'tophat', integral_pts = 10001), **variance_args }

        if self._model_linear_bias is None:
            raise CosmologyError("no bias model is linked to this cosmology")
        
        res = self._model_linear_bias( self, m, z, overdensity, variance_args, )
        return res



def predefined(name: str, 
               power_spectrum: str | cps.PowerSpectrum = 'eisenstein98_zb', 
               mass_function: str | cmf.MassFunction = 'tinker08',        
               linear_bias: str | chb.HaloBias = 'tinker10',
               interpolate: bool = True,           
               **kwargs                ) -> Cosmology:
    r"""
    Return a predefined cosmology model.
    """

    # cosmology with parameters from Plank et al (2018)
    if name == 'plank18':
        res =  Cosmology(
                            h      = 0.6790, 
                            Om0    = 0.3065, 
                            Ob0    = 0.0483, 
                            Ode0   = 0.6935, 
                            sigma8 = 0.8154, 
                            ns     = 0.9681, 
                            Tcmb0  = 2.7255, 
                            name   = name,  
                        )
    
    # cosmology with parameters from Plank et al (2015)
    elif name == 'plank15':
        res =  Cosmology( 
                            h      = 0.6736, 
                            Om0    = 0.3153, 
                            Ob0    = 0.0493, 
                            Ode0   = 0.6947, 
                            sigma8 = 0.8111, 
                            ns     = 0.9649, 
                            Tcmb0  = 2.7255, 
                            name   = name,
                        )
    
    # cosmology with parameters from WMAP survay
    elif name == 'wmap08':
        res =  Cosmology( 
                            h      = 0.719, 
                            Om0    = 0.2581, 
                            Ob0    = 0.0441, 
                            Ode0   = 0.742, 
                            sigma8 = 0.796, 
                            ns     = 0.963, 
                            Tcmb0  = 2.7255, 
                            name   = name,
                        )
    
    # cosmology for millanium simulation
    elif name == 'millanium':
        res = Cosmology( 
                            h      = 0.73, 
                            Om0    = 0.25, 
                            Ob0    = 0.045, 
                            sigma8 = 0.9, 
                            ns     = 1.0, 
                            Tcmb0  = 2.7255, 
                            name   = name,
                        )
    
    else:
        raise ValueError( "model '%s' not found :(" % name )
    

    if power_spectrum is not None:
        res.set( power_spectrum = power_spectrum )
    if mass_function is not None:
        res.set( mass_function = mass_function )
    if linear_bias is not None:
        res.set( linear_bias = linear_bias )

    if interpolate:
        res.createInterpolators( **kwargs )

    return res

    



