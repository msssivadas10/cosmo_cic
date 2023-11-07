#!/usr/bin/python3

import numpy as np
from scipy.integrate import simpson
from abc import ABC, abstractmethod
from typing import Any, Callable
from ..utils.constants import ONE_OVER_FOUR_PI2


class WindowFunction(ABC):
    r"""
    Base class representing a smoothing window function. These represesnt filters used to smooth the density 
    fluctuations and as kernal in convolving power spectrum in the matter variance calculations.   
    """

    @abstractmethod
    def f(self, x: Any, nu: int = 0) -> Any:
        r"""
        Returns the value of the window function in fourier space or its first two derivaties (`nu = 1, 2`).

        Parameters
        ----------
        x: array_like
        nu: int, default = 0

        Returns
        -------
        res: array_like 

        """
        ...

    __call__ = f
    
    def convolve(self, 
                 func: Callable,
                 lnr: Any, 
                 nu: int = 0, 
                 ka: float = 1e-08,
                 kb: float = 1e+08,
                 pts: int = 10001, 
                 *args, 
                 **kwargs,         ) -> Any:
        r"""
        Returns the value of colvolution of given function `func` with window.

        .. math:

            f(r) := \int {\rm d}k f(k) \vert w(kr) \vert^2

        Parameters
        ----------
        func: callable
            The argument should be log of the actual function argument.
        lnr: array_like
        nu: int, default = 0
            Order of the log derivative. Allowed values are 0, 1, and 2.
        ka, kb: float, optional
            Integration limit. Default is 1e-8 and 1e+8 respectively.
        pts: int, default = 10001
            Number of points used for a simpson rule integration.
        *args, **kwargs: Any
            Other arguments and keywords are passed to the function.

        Returns
        -------
        res: array_like
            Log of the convolution value.

        """

        lnr = np.asfarray( lnr )

        assert nu in [0, 1, 2], "nu must be 0, 1 or 2"

        lnk, dlnk = np.linspace( np.log(ka), np.log(kb), pts, retstep = True )
        lnk       = np.reshape( lnk, lnk.shape + tuple(1 for _ in lnr.shape) )

        f1 = func(lnk, *args, **kwargs)

        # NOTE: from now, lnk and lnr are k and r...
        lnr = np.exp( lnr )
        lnk = np.exp( lnk )
        kr  = lnr * lnk 
        f3  = self.__call__( kr, nu = 0 ) 

        # variance, s2
        f2   = 2 * f1 * f3
        res1 = simpson( f2 * f3, dx = dlnk, axis = 0 ) 
        if nu == 0:
            res1 = np.log( res1 ) 
            return res1
        
        # first log derivative, dln(s2)/dln(r)
        res1 = 2 * lnr / res1
        f3   = self.__call__( kr, nu = 1 ) * lnk
        res2 = simpson( f2 * f3, dx = dlnk, axis = 0 ) * res1
        if nu == 1:
            return res2

        # second log derivative, d2ln(s2)/d2ln(r)
        f2   = f2 * lnk * lnk * self.__call__( kr, nu = 2 ) + 2 * f1 * f3 * f3
        res2 = simpson( f2, dx = dlnk, axis = 0 ) * res1 * lnr + res2 * (1 - res2)
        return res2
    

class PowerSpectrum(ABC):
    r"""
    Base class representing a power spectrum object. These, combined with a cosmology model object are used 
    to calculate the values of matter power spectrum, variance, correlation etc.
    """

    @abstractmethod
    def lnt(self, 
            model: object, 
            lnk: Any, 
            lnzp1: Any,  ) -> Any:
        r"""
        Returns the log of linear transfer function :math:`T(k, z)`, based on the given cosmology model.

        Parameters
        ----------
        model: Cosmology
        lnk: array_like
        lnzp1: array_like

        Returns
        -------
        lnt: array_like

        """
        ...

    def __call__(self, 
                 model: object, 
                 lnk: Any, 
                 lnzp1: Any, 
                 der: bool = False, 
                 grid: bool = False ) -> Any:
        r"""
        Returns the log of matter power spectrum :math:`P(k, z)`, based on the given cosmology model.

        Parameters
        ----------
        model: Cosmology
        lnk: array_like
        lnzp1: array_like
        der: bool, default = False
            If set to true, returns the log derivative.
        grid: bool, default = False
            If set true, evaluate the function on a grid of input arrays. Otherwise, they must be broadcastable.

        Returns
        -------
        lnp: array_like

        """
        
        # flatten any multi-dimensional arrays, if given
        # lnk   = np.ravel( lnk ) if np.ndim( lnk ) > 1 else lnk
        # lnzp1 = np.ravel( lnzp1 ) if np.ndim( lnzp1 ) > 1 else lnzp1

        if not der:
            lnk, lnzp1 = np.asfarray( lnk ), np.asfarray( lnzp1 )
            if grid:
                # assert np.ndim( lnk ) <= 1 and np.ndim( lnzp1 ) <= 1
                lnk, lnzp1 = np.ravel( lnk )[:, None], np.ravel( lnzp1 )

            res  = 2 * self.lnt( model, lnk, lnzp1 ) + model.ns * lnk 
            return res
        
        # derivative calculation with finite difference
        k  = np.exp( lnk ) # k0
        h  = 0.01 * k
        
        k     = k - 2*h # k0 - 2h
        lnk   = np.log( k )
        dlnpk = self.__call__( model, lnk, lnzp1, False, grid )

        k     = k + h   # k0 - h
        lnk   = np.log( k )
        dlnk  = lnk
        dlnpk = dlnpk - 8 * self.__call__( model, lnk, lnzp1, False, grid )

        k     = k + 2*h   # k0 + h
        lnk   = np.log( k )
        dlnk  = 6 * ( lnk - dlnk )
        dlnpk = dlnpk + 8 * self.__call__( model, lnk, lnzp1, False, grid )

        k     = k + h   # k0 + 2h
        lnk   = np.log( k )
        dlnpk = dlnpk - self.__call__( model, lnk, lnzp1, False, grid )

        dlnk = np.ravel( dlnk )[:, None] if grid else dlnk
        res  = dlnpk / dlnk 
        return res

    def matterVariance(self, 
                       model: object, 
                       lnr: Any, 
                       lnzp1: Any, 
                       window: WindowFunction,
                       nu: int = 0, 
                       ka: float = 1e-08,
                       kb: float = 1e+08,
                       pts: int = 10001, 
                       grid: bool = False,   ) -> Any:
        r"""
        Returns the log of matter variance (`nu = 0`) or its first two log derivatives (`nu = 1, 2`).

        Parameters
        ----------
        model: Cosmology
        lnr: array_like
        lnzp1: array_like
        window: WindowFunction
            Smoothing window to use.
        nu: int, default = 0
            Order of the log derivative. Allowed values are 0, 1, and 2.
        ka, kb: float, optional
            Integration limit. Default is 1e-8 and 1e+8 respectively.
        pts: int, default = 10001
            Number of points used for a simpson rule integration.
        grid: bool, default = False
            If set true, evaluate the function on a grid of input arrays. Otherwise, they must be broadcastable.

        Returns
        -------
        res: array_like

        """

        # dimensionless power spectrum k^3 * P(k)
        def func(lnk, lnzp1, model):
            res = self.__call__(model, lnk, lnzp1, der = False, grid = False )
            res = np.exp( res + 3 * lnk )
            return res

        lnr, lnzp1 = np.asfarray( lnr ), np.asfarray( lnzp1 )
        if grid:
            # assert np.ndim( lnr ) <= 1 and np.ndim( lnzp1 ) <= 1
            lnr, lnzp1 = np.ravel( lnr )[:, None], np.ravel( lnzp1 )

        lnr, lnzp1 = np.broadcast_arrays( lnr, lnzp1 )

        if not isinstance(window, WindowFunction):
            raise TypeError("window must be a 'WindowFunction' instance")

        res = window.convolve(func, lnr, nu, ka, kb, pts, grid, lnzp1, model)
        return res if nu else res + np.log( ONE_OVER_FOUR_PI2 )

    def matterCorrelation(self, 
                          model: object, 
                          lnr: Any, 
                          lnzp1: Any, 
                          nu: int = 0, 
                          ka: float = 1e-08,
                          kb: float = 1e+08,
                          pts: int = 10001, 
                          grid: bool = False,   ) -> Any:
        r"""
        Returns the matter correlation function.

        Parameters
        ----------
        model: Cosmology
        lnr: array_like
        lnzp1: array_like
        nu: int, default = 0
            Order of the log derivative. Allowed values are 0, 1, and 2.
        ka, kb: float, optional
            Integration limit. Default is 1e-8 and 1e+8 respectively.
        pts: int, default = 10001
            Number of points used for a simpson rule integration.
        grid: bool, default = False
            If set true, evaluate the function on a grid of input arrays. Otherwise, they must be broadcastable.

        Returns
        -------
        res: array_like

        """
        raise NotImplementedError()
    
