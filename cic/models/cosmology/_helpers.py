#!/usr/bin/python3

from dataclasses import dataclass
from typing import Any, Callable

from .. import power_spectrum as cps
from ..halos import mass_function as cmf
from ..halos import bias as cbf
from ..utils import typestr, Interpolator1D, Interpolator2D


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

    def get_window(self) -> None:
        win = self.smoothWindow
        if isinstance( win, str ):
            if win not in cps.builtin_windows: 
                raise ValueError("window function '%s' is not available" % win)
            win = cps.builtin_windows[win]
        if not isinstance( win, cps.WindowFunction ): 
            raise TypeError("window function must be an instance of 'WindowFunction'")
        return win


@dataclass
class _InterpolationTables:
    _lnDistance: Interpolator1D      = None # ln of comoving distance: args = ln(z+1)
    _lnDplus: Interpolator1D         = None # ln of linear growth: args = ln(z+1)
    _lnPowerSpectrum: Interpolator2D = None # ln of power spectrum: args = log(k), ln(z+1)
    _lnVariance: Interpolator2D      = None # ln of variance: args = ln(r), ln(z+1)

    @property
    def lnDistance(self) -> Interpolator1D: return self._lnDistance

    @property
    def lnDplus(self) -> Interpolator1D: return self._lnDplus

    @property
    def lnPowerSpectrum(self) -> Interpolator2D: return self._lnPowerSpectrum

    @property
    def lnVariance(self) -> Interpolator2D: return self._lnVariance

    def create_lnDistance(self, 
                          func: Callable, 
                          lnzap1: float,
                          lnzbp1: float,
                          zpts: int, 
                          **kwargs,    ) -> None:
        self._lnDistance = Interpolator1D(func, xa = lnzap1, xb = lnzbp1, xpts = zpts, fkwargs = kwargs)
        return

    def create_lnDplus(self, 
                       func: Callable, 
                       lnzap1: float,
                       lnzbp1: float,
                       zpts: int, 
                       **kwargs,    ) -> None:
        self._lnDplus = Interpolator1D(func, xa = lnzap1, xb = lnzbp1, xpts = zpts, fkwargs = kwargs)
        return

    def create_lnPowerSpectrum(self, 
                               func: Callable,
                               lnka: float, 
                               lnkb: float, 
                               kpts: int, 
                               lnzap1: float, 
                               lnzbp1: float, 
                               zpts: int, 
                               **kwargs,    ) -> None:
        self._lnPowerSpectrum = Interpolator2D(func, 
                                               xa = lnka,   xb = lnkb,   xpts = kpts,
                                               ya = lnzap1, yb = lnzbp1, ypts = zpts, 
                                               fkwargs = kwargs, )
        return
    
    def create_lnVariance(self, 
                          func: Callable, 
                          lnra: float, 
                          lnrb: float, 
                          rpts: int, 
                          lnzap1: float, 
                          lnzbp1: float, 
                          zpts: int, 
                          **kwargs,    ) -> None:
        self._lnVariance = Interpolator2D(func, 
                                          xa = lnra,   xb = lnrb,   xpts = rpts,
                                          ya = lnzap1, yb = lnzbp1, ypts = zpts, 
                                          fkwargs = kwargs, )
        return


@dataclass
class _ModelsTable:
    _power_spectrum: cps.PowerSpectrum = None # model for matter power spectrum
    _mass_function: cmf.MassFunction = None # model for halo mass function
    _halo_bias: cbf.HaloBias         = None # model for halo bias

    @property
    def power_spectrum(self): return self._power_spectrum

    @property 
    def mass_function(self): return self._mass_function

    @property
    def halo_bias(self): return self._halo_bias

    @power_spectrum.setter
    def power_spectrum(self, value: Any):
        if isinstance( value, str ):
            if value not in cps.builtin_power_spectrums:
                raise ValueError(f"power spectrum model '{value}' is not available")
            value = cps.builtin_power_spectrums.get( value )
        if not isinstance( value, cps.PowerSpectrum ):
            raise TypeError(f"cannot use a '{typestr(value)}' object as power spectrum model")
        self._power_spectrum = value
        return

    @mass_function.setter
    def mass_function(self, value: Any):
        if isinstance( value, str ):
            if value not in cmf.builtin_massfunctions:
                raise ValueError(f"mass function model '{value}' is not available")
            value = cmf.builtin_massfunctions.get( value ) 
        if not isinstance( value, cmf.MassFunction ):
            raise TypeError(f"cannot use a '{typestr(value)}' object as mass function model")
        self._mass_function = value
        return

    @halo_bias.setter
    def halo_bias(self, value: Any):
        if isinstance( value, str ):
            if value not in cbf.builtin_linear_biases:
                raise ValueError(f"halo bias model '{value}' is not available")
            value = cbf.builtin_linear_biases.get( value ) 
        if not isinstance( value, cbf.HaloBias ):
            raise TypeError(f"cannot use a '{typestr(value)}' object as halo bias model")
        self._halo_bias = value
        return
    

