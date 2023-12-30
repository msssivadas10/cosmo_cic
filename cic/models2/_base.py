#!/usr/bin/python3

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp0f1
from dataclasses import dataclass
from typing import Any
from .utils.objects import ModelDatabase, IntegrationRule, DefiniteUnweightedCC
from .utils.constants import SPEED_OF_LIGHT_KMPS, RHO_CRIT0_ASTRO, DELTA_SC, MPC, YEAR                

#########################################################################################
#                                   Cosmology model                                     #
#########################################################################################

class CosmologyError(Exception):
    r"""
    Base class of exceptions raised in cosmology calculations.
    """

@dataclass
class _CosmologySettings:
    # inifinity value for redshift integration
    z_inf: float = 1e+08
    # zero value for k integration (variance and correlation calculations)
    k_zero: float = 1e-04
    # infinity value for k integration
    k_inf: float = 1e+04 
    # number of points for k integration
    k_pts: float = 1001
    # relative tolerance for convergence check
    reltol: float = 1e-06
    # redshift integration plan (points in log(z+1) space) NOTE: not used currently
    z_plan: IntegrationRule = DefiniteUnweightedCC(a = -1., b = 1., pts = 128)
    # k integration plan (points in log(k) space)
    k_plan: IntegrationRule = (DefiniteUnweightedCC(a = -14., b = -7., pts = 64 ) + # 1e-06 to 1e-03, approx. 
                               DefiniteUnweightedCC(a =  -7., b =  7., pts = 512) + # 1e-03 to 1e+03, approx. 
                               DefiniteUnweightedCC(a =   7., b = 14., pts = 64 ) ) # 1e+03 to 1e+06, approx. 

class Cosmology:
    r"""
    Base class representing a general cosmology model.
    """
    
    __slots__ = ('h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Ok0', 'Nnu', 'ns', 'sigma8', 
                 'Tcmb0', 'w0', 'wa', 'name', 'settings', '_POWERSPECTRUM_NORM', 
                 'power_spectrum', 'window_function', 'mass_function', 'halo_bias', 
                 'halo_cmreln', 'halo_profile', )

    def __init__(self, 
                 h: float, 
                 Om0: float, 
                 Ob0: float, 
                 Ode0: float | None = None, 
                 Onu0: float = 0.0,
                 Nnu: float = 3.0, 
                 ns: float = 1.0, 
                 sigma8: float = 1.0, 
                 w0: float = -1.,
                 wa: float = 0.,
                 Tcmb0: float = 2.725, 
                 name: str | None = None) -> None:
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
        self.Onu0 = Onu0   
        # number of massive neutrinos
        if Nnu <= 0: 
            raise CosmologyError("neutrino number must be positive")
        self.Nnu = Nnu   
        # check if all matter components have correct densities 
        if Ob0 + Onu0 > Om0: 
            raise CosmologyError("baryon + neutrino density cannot exceed total matter density")
        # dark energy density and curvature energy density 
        if Ode0 is None: 
            self.Ode0 = 1 - Om0
            self.Ok0  = 0.
        elif Ode0 >= 0: 
            self.Ok0  = 1 - Om0 - Ode0
            self.Ode0 = Ode0
            # round very small curvatures to zero (flat space)
            if abs(self.Ok0) < 1e-08:
                self.Ode0 += self.Ok0
                self.Ok0   = 0.   
        else: 
            raise CosmologyError("dark energy density must be non-negative")
        # w0-wa dark-energy model parameters
        self.w0, self.wa = w0, wa
        # initial power spectrum index
        self.ns = ns     
        # power spectrum normalisation #1: matter variance smoothed at 8 Mpc/h scale
        if sigma8 is not None and sigma8 <= 0: 
            raise CosmologyError("value of sigma8 parameter must be positive")
        self.sigma8 = sigma8 
        # cosmic microwave background (CMB) temperature in K
        if Tcmb0 <= 0: 
            raise CosmologyError("CMB temperature must be positive")
        self.Tcmb0  = Tcmb0 

        # name of the model
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        # power spectrum normalisation #2: set value of variance at r = 8 to 1
        self._POWERSPECTRUM_NORM = 1.0
        # general settings table
        self.settings = _CosmologySettings()
        # linked models (not initialised)
        self.power_spectrum : PowerSpectrum  = None
        self.window_function: WindowFunction = None
        self.mass_function  : MassFunction   = None
        self.halo_bias      : HaloBias       = None
        self.halo_cmreln    : HaloConcentrationMassRelation = None
        self.halo_profile   : HaloDensityProfile = None

    def __repr__(self) -> str:
        attrs = ('h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'ns', 'sigma8')
        if self.name is not None: 
            attrs = ('name', *attrs)
        return f"{self.__class__.__name__}({', '.join([f'{attr}={self.__getattribute__(attr)}' for attr in attrs])})"
    
    def link(self, **kwargs: Any, ) -> None:
        r"""
        Link models to calculate quantities. Models can be an instance of the class representing 
        the model, or a name of an available model. 

        Parameters
        ----------
        power_spectrum: str, PowerSpectrum
            Matter power spectrum model.
        window: str, WindowFunction
            Smoothing window or a name of an availa model.
        mass_function: str, MassFunction
            Halo mass-function model.
        halo_bias: str, HaloBias
            Halo bias model.
        cmreln: str, HaloConcentrationMassRelation
            Halo concentration-mass relation model.
        halo_profile: str, HaloDensityProfile
            Halo density profile model.

        """
        attrmap = {
                    'power_spectrum': [ 'power_spectrum' , PowerSpectrum  ],
                    'window'        : [ 'window_function', WindowFunction ],
                    'mass_function' : [ 'mass_function'  , MassFunction   ],
                    'halo_bias'     : [ 'halo_bias'      , HaloBias       ],
                    'cmreln'        : [ 'halo_cmreln'    , HaloConcentrationMassRelation ],
                    'halo_profile'  : [ 'halo_profile'   , HaloDensityProfile ],
                  }
        for __key, __value in kwargs.items():
            if __key not in attrmap: 
                raise TypeError(f"unknown argument '{ __key }'")
            attr, cls = attrmap[ __key ]
            if isinstance(__value, str):
                if not cls.available.exists(__value):
                    raise CosmologyError(f"{ attr.replace('_', ' ') } model not available: '{ __value }' ")    
                __value = cls.available.get(__value)
            elif not isinstance(__value, cls):
                raise CosmologyError(f"{ attr.replace('_', ' ') } model must be a '{ cls.__name__ }' object")
            self.__setattr__(attr, __value)

        # normalising the power spectrum (normalization #2) 
        if self.power_spectrum is not None and self.window_function is not None:
            self._POWERSPECTRUM_NORM = 1.0
            self._POWERSPECTRUM_NORM = 1.0 / self.matterVariance(8, normalize = False)  
        return

    def darkEnergyModel(self, 
                        z: Any,
                        deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the dark-energy density.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like

        """
        # constant w model: wa = 0, w = w0
        if abs(self.wa) < 1e-08: 
            # cosmological constant: w0 = -1 
            if abs(self.w0 + 1) < 1e-08: 
                if deriv:
                    return np.zeros_like(z, dtype = 'float')
                return np.ones_like(z, dtype = 'float')
            # other constant w values:
            p = (3*self.w0 + 3)
            if deriv:
                return p * np.add(z, 1.)**(p-1)
            return np.add(z, 1.)**p
        # general w0-wa model:
        zp1 = np.add(z, 1.)
        p   = 3*( self.w0 + self.wa * (zp1 - 1) / zp1 ) + 3
        res = zp1**p
        if deriv:
            return res / zp1 * ( p + 3*self.wa * np.log(zp1) / zp1 )
        return res
    
    def isFlat(self) -> bool:
        r"""
        Tell if the cosmology is flat or curved.
        """
        return abs(self.Ok0) < 1e-08
    
    def curvature(self) -> float:
        r"""
        Return the curvature of space.
        """
        if self.isFlat(): return 0.0
        curv = np.sqrt( abs(self.Ok0) ) / ( 0.01*SPEED_OF_LIGHT_KMPS )
        # spherical space has positive curvature
        if self.Ok0 < 0.: return curv 
        # hyperbolic space has negative curvature
        return -curv 
    
    def lnE2(self, 
             lnzp1: Any, 
             deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the hubble parameter, as function of the redshift 
        variable :math:`\ln(z+1)`.

        Parameters
        ----------
        lnzp1: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
        
        """
        zp1  = np.exp(lnzp1)
        res1 = self.Om0 * zp1**3
        if deriv: 
            res2 = 3*res1 

        if self.isFlat():
            __tmp = self.Ok0 * zp1**2
            res1 += __tmp
            if deriv:
                res2 += 2*__tmp

        res1 += self.Ode0 * self.darkEnergyModel(zp1-1, deriv = 0)
        if deriv:
            res2 += self.Ode0 * self.darkEnergyModel(zp1-1, deriv = 1)

        if deriv:
            return res2 / res1
        return np.log(res1)
    
    def E(self, 
          z: Any, 
          deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the hubble parameter.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like

        See Also
        --------
        Cosmology.lnE2
        
        """
        lnzp1 = np.log( np.add(z, 1.) )
        res   = np.exp( 0.5 * self.lnE2(lnzp1, deriv = 0) )
        if deriv:
            res *= 0.5 * self.lnE2(lnzp1, deriv = 1) * np.exp(-lnzp1)
        return res
    
    def Om(self, 
           z: Any, 
           deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the matter density.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Matter density in unit of critical density.

        """
        lnzp1 = np.log( np.add(z, 1.) )
        res   = self.Om0 * np.exp( 3*lnzp1 - self.lnE2(lnzp1, deriv = 0) )
        if deriv:
            res *= (3. - self.lnE2(lnzp1, deriv = 1)) * np.exp(-lnzp1)
        return res
    
    def Ode(self, 
            z: Any, 
            deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the dark-energy density.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Dark-energy density in unit of critical density.

        """
        lnzp1 = np.log( np.add(z, 1.) )
        fde   = self.darkEnergyModel(z, deriv = 0)
        res   = self.Ode0 * fde * np.exp( -self.lnE2(lnzp1, deriv = 0) )
        if deriv:
            res *= self.darkEnergyModel(z, deriv = 1) / fde - self.lnE2(lnzp1, deriv = 1) * np.exp(-lnzp1)
        return res
    
    def criticalDensity(self, 
                        z: Any, 
                        deriv: int = 0, ) -> Any:
        r"""
        Return the redshift evolution of the critical density.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Critical density in astrophysical units (h^2 Msun/Mpc^3).

        """
        res = self.E(z, deriv)
        if deriv:
            res *= 2*self.E(z, deriv = 0)
        else:
            res *= res
        return RHO_CRIT0_ASTRO * res
    
    ###########################################################################################################
    #                                       Distance calculations                                             #
    ###########################################################################################################

    def comovingDistance(self, 
                         z: Any,
                         deriv: int = 0, ) -> Any:
        r"""
        Return the comoving distance corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Comoving distance in astrophysical units (Mpc/h).

        """
        FACT = 0.01*SPEED_OF_LIGHT_KMPS
        # function to integrate
        def func(lnzp1: float) -> float:
            res = np.exp(lnzp1 - 0.5*self.lnE2(lnzp1, deriv = 0))
            return res

        zp1 = np.add(z, 1.)
        if deriv:
            return FACT * func( np.log(zp1) ) / zp1
        zp1 = 0.5*np.log(zp1)
        # generating integration points
        pts, wts = self.settings.z_plan.nodes, self.settings.z_plan.weights
        shape    = np.shape(pts) + tuple(1 for _ in np.shape(zp1))
        pts, wts = np.reshape(1 + pts, shape) * zp1, np.reshape(wts, shape) * zp1
        # log-space integration
        res = np.sum( func( pts ) * wts, axis = 0 )
        return FACT * res
    
    def comovingCoordinate(self, 
                           z: Any,
                           deriv: int = 0, ) -> Any:
        r"""
        Return the comoving coordinate distance corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Comoving coordinate distance in astrophysical units (Mpc/h).

        See Also
        --------
        Cosmology.comovingDistance
        
        """
        if self.isFlat(): 
            if deriv: 
                return self.comovingDistance(z, deriv = 1)
            return self.comovingDistance(z, deriv = 0)
        
        # for curved space
        k   = np.sqrt( abs(self.Ok0) ) / ( 0.01*SPEED_OF_LIGHT_KMPS ) # curvature
        res = self.comovingDistance(z, deriv = 0)
        if self.Ok0 < 0: # spherical or closed geometry
            if deriv:
                return np.cos(k*res) * self.comovingDistance(z, deriv = 1)
            return np.sin(k*res) / k
        # hyperbolic or open geometry
        if deriv:
            return np.cosh(k*res) * self.comovingDistance(z, deriv = 1)
        return np.sinh(k*res) / k
    
    def luminocityDistance(self, 
                           z: Any, 
                           deriv: int = 0, ) -> Any:
        r"""
        Return the luminocity distance corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Luminocity distance in astrophysical units (Mpc/h).

        See Also
        --------
        Cosmology.comovingDistance
        
        """
        res = self.comovingCoordinate(z, deriv = 0)
        if deriv:
            return res + self.comovingCoordinate(z, deriv = 1) * np.add(z, 1.)
        return res * np.add(z, 1.)

    def angularDiameterDistance(self, 
                                z: Any,
                                deriv: int = 0, ) -> Any:
        r"""
        Return the angular diameter distance corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Angular diameter distance in astrophysical units (Mpc/h).

        See Also
        --------
        Cosmology.comovingDistance
        
        """ 
        res = self.comovingCoordinate(z, deriv = 0) / np.add(z, 1.)
        if deriv:
            return (self.comovingCoordinate(z, deriv = 1) - res) / np.add(z, 1.)
        return res
    
    def angularSize(self, 
                    z: Any, 
                    value: float = 1.0, 
                    inverse: bool = False, ) -> Any:
        r"""
        Convert angular sizes at a redshift z to corresponding physical size.

        Parameters
        ----------
        z: array_like
        value: float
            Angular size in arcseconds (physical size in Mpc/h, if `invert` is true).
        invert: bool, default = False
            If true, convert from physical size to angular size.

        Returns
        -------
        res: array_like
            Physical size in Mpc/h (angular size in arcsec, if `invert` is true)

        """
        UNIT_ARCSEC_IN_RADIAN = np.pi / 180. / 3600.
        dist = self.angularDiameterDistance(z, deriv = 0)
        if inverse: # angular size (arcsec) to physical size (Mpc/h)
            return np.multiply(value, dist) * UNIT_ARCSEC_IN_RADIAN
        # physical size (Mpc/h) to angular size (arcsec)
        return np.divide(value, dist) / UNIT_ARCSEC_IN_RADIAN
    
    ###########################################################################################################
    #                                       Time and age calculations                                         #
    ###########################################################################################################

    def time(self, 
            z: Any, 
            deriv: int = 0, ) -> Any:
        r"""
        Return the time corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Time in units of present hubble time.

        """
        # function to integrate
        def func(lnzp1: float) -> float:
            res = np.exp( -0.5*self.lnE2(lnzp1, deriv = 0) )
            return res
        
        zp1 = np.add(z, 1.)
        if deriv:
            return -func( np.log(zp1) ) / zp1
        inf = np.log(self.settings.z_inf+1)
        zp1 = 0.5*( inf - np.log(zp1) )
        # # generating integration points
        pts, wts = self.settings.z_plan.nodes, self.settings.z_plan.weights
        shape    = np.shape(pts) + tuple(1 for _ in np.shape(zp1))
        pts, wts = np.reshape(pts - 1, shape) * zp1 + inf, np.reshape(wts, shape) * zp1
        # # log-space integration
        res = np.sum( func( pts ) * wts, axis = 0 )
        return res
    
    def hubbleTime(self, 
                   z: Any, 
                   deriv: int = 0, ) -> Any:
        r"""
        Return the hubble time corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Time in astrophysical units (Gyr).

        """
        HT  = 1e-14 * MPC / YEAR / self.h # present hubble time in Gyr
        zp1 = np.add(z, 1.)
        res = np.exp( self.lnE2(np.log(zp1), deriv = 0) )
        if deriv:
            return -HT * self.E(z, deriv = 1) / res
        return HT / np.sqrt(res)
    
    def age(self, 
            z: Any = 0., 
            deriv: int = 0, ) -> float:
        r"""
        Return the age of the universe corresponding to redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first derivative of the function.

        Returns
        -------
        res :array_like
            Time in astrophysical units (Gyr).

        """
        HT  = 1e-14 * MPC / YEAR / self.h # present hubble time in Gyr
        return HT * self.time(z, deriv)        
    
    ###########################################################################################################
    #                                       Linear growth calculations                                        #
    ###########################################################################################################

    def dplus(self, 
              z: Any, 
              deriv: int = 0, ) -> Any:
        r"""
        Return the linear growth factor at redshift z.

        Parameters
        ----------
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first log derivative of the function.

        Returns
        -------
        res :array_like
            Linear growth factor, without normalization.
 
        """
        # function to integrate
        def func(lnzp1: float) -> float:
            res = np.exp( 2*lnzp1 - 1.5*self.lnE2(lnzp1, deriv = 0) )
            return res
        
        inf   = np.log(self.settings.z_inf+1)
        lnzp1 = np.log(np.add(z, 1.))
        # # generating integration points
        _scale   = 0.5*( inf - lnzp1 )
        pts, wts = self.settings.z_plan.nodes, self.settings.z_plan.weights
        shape    = np.shape(pts) + tuple(1 for _ in np.shape(lnzp1))
        pts, wts = np.reshape(pts - 1, shape) * _scale + inf, np.reshape(wts, shape) * _scale
        # # log-space integration
        res = np.sum( func( pts ) * wts, axis = 0 )
        if deriv: # growth rate: f(z)
            res = func(lnzp1) / res - 0.5*self.lnE2(lnzp1, deriv = 1)
        else: # growth factor: D_+(z)
            res = np.exp( 0.5*self.lnE2(lnzp1, deriv = 0) ) * res # * 2.5 * self.Om0
        return res

    ###########################################################################################################
    #                                  Matter power spectrum calculations                                     #
    ###########################################################################################################

    def matterTransferFunction(self, 
                               k: Any, 
                               z: Any = 0., 
                               deriv: int = 0, ) -> Any:
        r"""
        Return the linear matter transfer function for wavenumber k and redshift z.

        Parameters
        ----------
        k: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first log derivative of the function.

        Returns
        -------
        res: array_like
            Matter transfer function.

        """
        if self.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked with this model")
        res = self.power_spectrum.call(self, k, z, deriv)
        return res

    def matterPowerSpectrum(self, 
                            k: Any, 
                            z: Any = 0., 
                            deriv: int = 0, 
                            normalize: bool = True, 
                            nonlinear: bool = False, ) -> Any:
        r"""
        Return the matter power spectrum for wavenumber k and redshift z.

        Parameters
        ----------
        k: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first log derivative of the function (effective index).
        normalize: bool, default = True
            If true, normalize the result using value of :math:`\sigma_8`.
        nonlinear: bool, default = False    
            Not used.

        Returns
        -------
        res: array_like
            Matter power spectrum in astrophysical units (h^3/Mpc^3).

        """
        if self.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked with this model")
        
        k   = np.asfarray(k)
        res = self.matterTransferFunction(k, z, deriv)
        if not deriv:
            res = res**2 * k**self.ns
            # power spectrum is normalised in three ways: 
            if normalize is not None:
                if normalize: # normalise the value at r = 8 to sigma8
                    NORM = (self._POWERSPECTRUM_NORM * self.sigma8**2)  
                else: # normalise the value at r = 8 to 1
                    NORM = self._POWERSPECTRUM_NORM
                res *= NORM    
            # or, no normalisation (NOTE: used in variance calculations)
            return res
        
        # effective index (1-st log derivative) calculation
        res = self.ns + 2.*res
        return res
    
    def matterCorrelation(self, 
                          r: Any, 
                          z: Any = 0., 
                          average: bool = True, 
                          normalize: bool = True,
                          nonlinear: bool = False, ) -> Any:
        r"""
        Return the matter correlation function for scale r (in Mpc/h) and redshift z.

        Parameters
        ----------
        r: array_like
        z: array_like
        average: bool, default = True
            If true, return the average correlation over a sphere of radius r.
        normalize: bool, default = True
            If true, normalize the result using value of :math:`\sigma_8`.
        nonlinear: bool, default = False    
            Not used.

        Returns
        -------
        res: array_like
            Matter 2-point correlation functions 

        """
        if self.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked with this model")
        
        # generate integration points in log space
        pts = self.settings.k_plan.nodes
        wts = self.settings.k_plan.weights
        # reshaping k array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z)))
        pts = np.reshape(pts, shape)
        wts = np.reshape(wts, shape)
        # correlation 
        k  = np.exp(pts)
        r  = np.asfarray(r)
        kr = k*r
        res = self.matterPowerSpectrum(k, z, deriv = 0, normalize = normalize, nonlinear = nonlinear) * k**3
        if average: # average correlation function
            res = res * hyp0f1(2.5, -0.25*kr**2) # 0F1(2.5, -0.25*x**2) = 3*( sin(x) - x * cos(x) ) / x**3
        else: # correlation function
            res = res * np.sinc(kr / np.pi)
        res = np.sum( res*wts, axis = 0 )
        return res
    
    def matterVariance(self, 
                       r: Any, 
                       z: Any = 0., 
                       deriv: int = 0,
                       normalize: bool = True, 
                       nonlinear: bool = False, ) -> Any:
        r"""
        Return the smoothed matter variance for scale r (in Mpc/h) and redshift z.

        Parameters
        ----------
        r: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the log derivative of that order (values = 0, 1 or 2).
        normalize: bool, default = True
            If true, normalize the result using value of :math:`\sigma_8`.
        nonlinear: bool, default = False    
            Not used.

        Returns
        -------
        res: array_like
            Matter variance.
        """
        j = 0 # TODO: as input argument
        if self.power_spectrum is None:
            raise CosmologyError("no power spectrum model is linked with this model")
        if self.window_function is None:
            raise CosmologyError("no window function model is linked with this model")
        if not isinstance(j, int) or j < 0:
            raise CosmologyError("j must be zero or positive integer")
        
        # generate integration points in log space
        pts = self.settings.k_plan.nodes
        wts = self.settings.k_plan.weights
        # reshaping k array to work with array inputs
        shape = np.shape(pts) + tuple(1 for _ in np.broadcast_shapes(np.shape(r), np.shape(z)))
        pts = np.reshape(pts, shape)
        wts = np.reshape(wts, shape)

        # normalization constant
        NORM = self._POWERSPECTRUM_NORM / (2*np.pi**2)
        if normalize:
            NORM *= self.sigma8**2

        k  = np.exp(pts)
        r  = np.asfarray(r)
        kr = k*r
        # variance
        f1 = NORM * self.matterPowerSpectrum(k, z, deriv = 0, normalize = None, nonlinear = nonlinear) * k**(3 + j)
        f3 = self.window_function(kr, deriv = 0)
        f2 = f1*f3
        res1 = np.sum( f2*f3*wts, axis = 0 )
        if deriv == 0:
            return res1
        # first derivative
        f3   = 2 * self.window_function(kr, deriv = 1) * k
        res2 = np.sum( f2*f3*wts, axis = 0 )
        if deriv == 1:
            return r / res1 * res2
        # second derivative
        res3 = 0.5*f1 * f3**2 + 2*f2 * self.window_function(kr, deriv = 2) * k**2
        res3 = np.sum( res3*wts, axis = 0 )
        if deriv == 2:
            res1 = r / res1
            res2 = res1 * res2
            res2 = res2 * (1. - res2)
            return res2 + res1 * res3 * r
        
        raise ValueError(f"invalid value for argument 'deriv': {deriv}")
      
    def nu(self, 
           r: Any, 
           z: Any = 0., 
           normalize: bool = True, 
           nonlinear: bool = False, ) -> Any:
        r"""
        Related to matter variance.

        Parameters
        ----------
        r: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the log derivative of that order (values = 0, 1 or 2).
        normalize: bool, default = True
            If true, normalize the result using value of :math:`\sigma_8`.
        nonlinear: bool, default = False    
            Not used.

        Returns
        -------
        res: array_like
            
        """
        var = self.matterVariance(r, z, deriv = 0, normalize = normalize, nonlinear = nonlinear)
        res = self.collapseDensity(z) / np.sqrt(var)
        return res
    
    def collapseDensity(self, z: Any) -> Any:
        r"""
        Return the critical over-density for spherical collapse at redshift z.

        Parameters
        ----------
        z: array_like

        Returns
        -------
        res: array_like

        """
        return DELTA_SC * np.ones_like(z, dtype = 'float')
    
    def collapseRadius(self, 
                       z: Any, 
                       **kwargs: Any, ) -> Any:
        r"""
        Return the collapse radius of a halo at redshift z.

        Parameters
        ----------
        z: array_like

        Returns
        -------
        res: array_like
            Collapse radius in Mpc/h.

        """
        lnra, lnrb = np.log(1e-03), np.log(1e+03)

        # cost function: function to find root
        def cost(lnr: float, z: float) -> float:
            res = self.nu(np.exp(lnr), z, normalize = True, nonlinear = False) - 1.
            return res

        # vectorised function returning collapse radius
        @np.vectorize
        def result(z: float) -> float: 
            lnr = brentq(cost, lnra, lnrb, args = (z, ), disp = False, **kwargs)
            return np.exp( lnr )

        res = result(z)
        return res
    
    def collapseRedshift(self, 
                         r: Any, 
                         **kwargs: Any, ) -> Any:
        r"""
        Return the collapse redshift for a halo of radius r Mpc/h.

        Parameters
        ----------
        r: array_like

        Returns
        -------
        res: array_like
            Collapse redshift.

        """
        raise NotImplementedError()
    
    ###########################################################################################################
    #                                               Halo statistics                                           #
    ###########################################################################################################

    def lagrangianM(self, 
                    r: Any, 
                    overdensity: float | None = None, ) -> Any:
        r"""
        Return the mass corresponding to a halo of lagrangian radius r Mpc/h.

        Parameters
        ----------
        r: array_like
        overdensity: float, default = None
            Halo over-density value w.r.to mean density (for spherical overdensity halos).

        Returns
        -------
        res: array_like
            Mass in astrophysical units (Msun/h).

        """
        halo_density = self.Om0 * RHO_CRIT0_ASTRO
        if overdensity is not None:
            halo_density *= overdensity
        res = 4*np.pi/3. * np.asfarray(r)**3 * halo_density
        return res
    
    def lagrangianR(self, 
                    m: Any, 
                    overdensity: float | None = None, ) -> Any:
        r"""
        Return the lagrangian radius corresponding to a halo of mass Msun/h.

        Parameters
        ----------
        m: array_like
        overdensity: float, default = None
            Halo over-density value w.r.to mean density (for spherical overdensity halos).

        Returns
        -------
        res: array_like
            Radius in astrophysical units (Mpc/h).

        """
        halo_density = self.Om0 * RHO_CRIT0_ASTRO
        if overdensity is not None:
            halo_density *= overdensity
        res = np.cbrt( 0.75*np.asfarray(m) / np.pi / halo_density )
        return res
    
    def haloMassFunction(self, 
                         m: Any, 
                         z: float = 0., 
                         overdensity: float | None = 200, 
                         retval: str = 'dndlnm',        ) -> Any:
        r"""
        Returns the halo mass-function, number density of halos of mass m (Msun/h) at redshift z.

        Parameters
        ----------

        m: array_like
        z: array_like, default = 0
        overdensity: float, default = 200
        retval: str, default = `dndlnm`
            Specify the return value. Allowed values are `f`, `dndm`, `dndlnm` and `full`.

        Returns
        -------
        res: array_like 

        """
        if self.mass_function is None:
            raise CosmologyError("no mass-function model is linked with this model")
        m = np.asfarray(m)
        # size of the halo
        r = self.lagrangianR(m, overdensity)
        # average matter variance inside the halo
        s = np.sqrt(self.matterVariance(r, z, deriv = 0, normalize = True, nonlinear = False))
        # mass-function f(s)
        f = self.mass_function.call(self, s, z, overdensity)
        if retval in ['f', 'fsigma']:
            return f
        density  = self.Om0 * RHO_CRIT0_ASTRO * np.add(z, 1.)**3
        dlnsdlnm = self.matterVariance(r, z, deriv = 1, normalize = True, nonlinear = False) / 6.
        # number density
        dndm = f * np.abs(dlnsdlnm) * density / m**2
        if retval in ['dndm']:
            return dndm
        if retval in ['dndlnm']:
            return dndm * m
        if retval != 'full':
            raise ValueError(f"invalid value for argument 'retval': {retval}")
        return (m, s, dlnsdlnm, f, dndm) # TODO: as dataclass
    
    def haloBias(self, 
                 m: Any, 
                 z: Any = 0., 
                 overdensity: float | None = 200, ) -> Any:
        r"""
        Returns the halo bias function value for a halo of mass m (Msun/h) at redshift z. 

        Parameters
        ----------
        m: array_like
        z: array_like, default = 0
        overdensity: float, default = 200

        Returns
        -------
        res: array_like 

        """
        if self.halo_bias is None:
            raise CosmologyError("no halo bias model is linked with this model")
        m = np.asfarray(m)
        # size of the halo
        r   = self.lagrangianR(m, overdensity)
        nu  = self.nu(r, z, normalize = True, nonlinear = False)
        res = self.halo_bias.call(self, nu, z, overdensity)
        return res 
    
    ###########################################################################################################
    #                                               Halo structure                                            #
    ###########################################################################################################

    def haloConcentration(self, 
                          m: Any, 
                          z: Any = 0., 
                          overdensity: float | None = None, ) -> Any:
        r"""
        Return the halo concentration for mass m (Msun/h) at redshift z.

        Parameters
        ----------
        m: array_like
        z: array_like, default = 0
        overdensity: float, default = 200

        Returns
        -------
        res: array_like 

        """
        if self.halo_cmreln is None:
            raise CosmologyError("no c-m relation model is linked with this model")
        res = self.halo_cmreln.call(self, m, z, overdensity)
        return res
    
    def haloProfile(self, 
                    arg: Any, 
                    m: Any, 
                    z: Any = 0., 
                    overdensity: float | None = None, 
                    fourier_transform: bool = False, 
                    trancate: bool = False,        ) -> Any:
        r"""
        Return the density profile of a halo of mass m (Msun/h) at redshift z. 

        Parameters
        ----------
        arg: array_like
            Argument: distance (Mpc/h) from the halo center for real space profiles and 
            wavenumber (h/Mpc) for fourier space profiles. 
        m: array_like
        z: array_like
        overdensity: float, default = None
        fourier_transform: bool, default = False
            If true, return the fourier space profile, otherwise real space profile.
        trancate: bool, default = False
            If true, truncate the real space profile at virial radius.

        Returns
        -------
        res: array_like

        """
        if self.halo_profile is None:
            raise CosmologyError("no halo profile model is linked with this model")
        # reshape inputs to work with arrays
        m   = np.asfarray(m)
        zp1 = np.add(z, 1.)
        # concentration
        c = self.haloConcentration(m, z, overdensity)
        # virial radius
        rvir = self.lagrangianR(m, overdensity) / zp1
        # scale the argument in units of virial radius
        arg = arg * rvir if fourier_transform else arg / rvir
        res = self.halo_profile.call(arg, c, fourier_transform) * c**3 / self.halo_profile.A(c)
        if not fourier_transform:
            density = self.Om0 * zp1**3
            if overdensity is not None: 
                density = density * overdensity
            res = res * density
            # truncating at virial radius
            if trancate:
                res = np.where( np.less_equal(arg, 1.), res, 0. )
        return res
    
#########################################################################################
#                               Other related models                                    #
#########################################################################################

class PowerSpectrum:
    r"""
    Base class representing a (linear) matter power spectrum model. 
    """

    # a database of available power spectrums
    available: ModelDatabase = None

    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      
             deriv: int = 0, 
             **kwargs: Any,     ) -> Any:
        r"""
        Returns the value of linear transfer function for wavenumber k (in Mpc/h) at 
        redshift z.

        Paremeters
        ---------- 
        model: Cosmology
            Cosmology model to use.
        k: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first log derivative w.r.to k, otherwise the function.

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, 
                 model: Cosmology, 
                 k: Any, 
                 z: Any,      
                 deriv: int = 0, 
                 **kwargs: Any, ) -> Any:
        r"""
        Returns the value of linear transfer function for wavenumber k (in Mpc/h) at 
        redshift z.

        Paremeters
        ---------- 
        model: Cosmology
            Cosmology model to use.
        k: array_like
        z: array_like
        deriv: int, default = 0
            If non-zero, return the first log derivative w.r.to k, otherwise the function.

        Returns
        -------
        res: array_like

        """
        return self.call(model, k, z, deriv, **kwargs)

PowerSpectrum.available = ModelDatabase('power_spectra', PowerSpectrum)
        
class WindowFunction:
    r"""
    Base class representing a smoothing window model. These are used smooth matter density 
    field in calculating variance. 
    """

    # a database of available window functions
    available: ModelDatabase = None

    def call(self, x: Any, deriv: int = 0) -> Any:
        r"""
        Returns the value of the window function or its derivatives.

        Parameters
        ----------
        x: array_like
        deriv: int, default = 0
            If non-zero, return the log derivative of that order (values = 0, 1 or 2).

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, x: Any, deriv: int = 0) -> Any:
        r"""
        Returns the value of the window function or its derivatives.

        Parameters
        ----------
        x: array_like
        deriv: int, default = 0
            If non-zero, return the derivative of that order (values = 0, 1 or 2).

        Returns
        -------
        res: array_like

        """
        return self.call(x, deriv)
    
WindowFunction.available = ModelDatabase('window_functions', WindowFunction)

class MassFunction:
    r"""
    Base class representing a halo mass-function model. 
    """

    # a database of available mass functions
    available: ModelDatabase = None

    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        r"""
        Return the halo mass-function as a function of the mass variable s at redshift z.

        Parameters
        ----------
        model: Cosmology
            Cosmology model to use.
        s: array_like
        z: array_like
        overdensity: float

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, 
                 model: Cosmology, 
                 s: Any, 
                 z: Any, 
                 overdensity: float | None = None, ) -> Any:
        r"""
        Return the halo mass-function as a function of the mass variable s at redshift z.

        Parameters
        ----------
        model: Cosmology
            Cosmology model to use.
        s: array_like
        z: array_like
        overdensity: float

        Returns
        -------
        res: array_like

        """
        return self.call(model, s, z, overdensity)

MassFunction.available = ModelDatabase('mass_functions', MassFunction)

class HaloBias:
    r"""
    Base class representing a halo bias model. 
    """

    # a database of available halo bias functions
    available: ModelDatabase = None

    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        r"""
        Return the halo mass-function as a function of the mass variable :math:`\nu` at redshift z.

        Parameters
        ----------
        model: Cosmology
            Cosmology model to use.
        nu: array_like
        z: array_like
        overdensity: float

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, 
                 model: Cosmology, 
                 nu: Any,   
                 z: Any, 
                 overdensity: float | None = None, ) -> Any:
        r"""
        Return the halo mass-function as a function of the mass variable :math:`\nu` at redshift z.

        Parameters
        ----------
        model: Cosmology
            Cosmology model to use.
        nu: array_like
        z: array_like
        overdensity: float

        Returns
        -------
        res: array_like

        """
        return self.call(model, nu, z, overdensity)
    
HaloBias.available = ModelDatabase('halo_bias', HaloBias)

class HaloConcentrationMassRelation:
    r"""
    Base class representing the concentration-mass relation of a dark-matter halo.
    """

    # a database of available c-m relations
    available: ModelDatabase = None

    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        r"""
        Returns the halo concentration as function of mass m, at redshift z.

        Parameters
        ----------
        model: Cosmology
        m: array_like
        z: array_like
        overdensity: float, None

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, 
                 model: Cosmology, 
                 m: Any, 
                 z: Any, 
                 overdensity: float | None = None, ) -> Any:
        r"""
        Returns the halo concentration as function of mass m, at redshift z.

        Parameters
        ----------
        model: Cosmology
        m: array_like
        z: array_like
        overdensity: float, None

        Returns
        -------
        res: array_like

        """
        return self.call(model, m, z, overdensity)
    
HaloConcentrationMassRelation.available = ModelDatabase('halo_cmrelns', HaloConcentrationMassRelation)

class HaloDensityProfile:
    r"""
    Base class representing a density profile for the halos.
    """

    # a database of available c-m relations
    available: ModelDatabase = None

    def A(self, c: Any) -> Any:
        r"""
        A function relating halo concentration c to its virial mass.

        Parameters
        ----------
        c: array_like

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def call(self, 
             arg: Any, 
             c: Any, 
             fourier_transform: bool = False, ) -> Any:
        r"""
        Returns the halo profile in real or fourier transform space. 

        Parameters
        ----------
        arg: array_like
            For real space profile, distance from the center in Mpc/h. For fourier space 
            profile, wavenumber in h/Mpc.
        c: array_like
            Concentration parameter. 
        fourier_transform: bool, default = False
            If true, return fourier space profile, other wise real space.

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
    def __call__(self, 
                 arg: Any, 
                 c: Any, 
                 fourier_transform: bool = False, ) -> Any:
        r"""
        Returns the halo profile in real or fourier transform space. 

        Parameters
        ----------
        arg: array_like
            For real space profile, distance from the center in Mpc/h. For fourier space 
            profile, wavenumber in h/Mpc.
        c: array_like
            Concentration parameter. 
        fourier_transform: bool, default = False
            If true, return fourier space profile, other wise real space.

        Returns
        -------
        res: array_like

        """
        return self.call(arg, c, fourier_transform)
    
HaloDensityProfile.available = ModelDatabase('halo_profiles', HaloDensityProfile)

