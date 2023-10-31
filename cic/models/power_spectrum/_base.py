#!/usr/bin/python3

import numpy as np
from scipy.integrate import simpson
from abc import ABC, abstractmethod
from typing import Any, Callable
from ..utils.constants import ONE_OVER_FOUR_PI2


class WindowFunction(ABC):
    r"""
    Base class representing a smoothing window function.
    """

    @abstractmethod
    def f(self, x: Any, nu: int = 0) -> Any:
        ...

    def __call__(self, x: Any, nu: int = 0) -> Any:
        return self.f(x, nu)
    
    def convolve(self, 
                 func: Callable,
                 lnr: Any, 
                 nu: int = 0, 
                 ka: float = 1e-08,
                 kb: float = 1e+08,
                 pts: int = 10001, 
                 grid: bool = False,   
                 *args, 
                 **kwargs,         ) -> Any:
        r"""
        Return the value of colvolution of given function with window.
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
    Base class representing a matter power spectrum object.
    """

    @abstractmethod
    def lnt(self, 
            model: object, 
            lnk: Any, 
            lnzp1: Any,  ) -> Any:
        r"""
        Calculate the log of linear transfer function.
        """
        ...

    def __call__(self, 
                 model: object, 
                 lnk: Any, 
                 lnzp1: Any, 
                 der: bool = False, 
                 grid: bool = False ) -> Any:
        r"""
        Calculate the log of linear matter power spectrum.
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
                       nu: int = 0, 
                       window: WindowFunction = None,
                       ka: float = 1e-08,
                       kb: float = 1e+08,
                       pts: int = 10001, 
                       grid: bool = False,   ) -> Any:
        r"""
        Calculate the linear matter variance or its first two derivatives.
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
                          r: Any, 
                          z: Any, 
                          exact: bool = True, 
                          integral_pts: int = 10001, ) -> Any:
        r"""
        Calculate the linear matter correlation function.
        """
        raise NotImplementedError()
    
