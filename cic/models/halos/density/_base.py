#!/usr/bin/python3

import numpy as np
from abc import ABC, abstractmethod
from typing import Any 


class CMRelation(ABC):
    r"""
    Base calss representing a halo concentration - mass relation. 
    """

    def set(self, **kwargs):
        r"""
        Set parameters for the c-M relation.
        """
        
        for attr, value in kwargs.items(): setattr(self, attr, value) 
        return self

    @abstractmethod
    def cmreln(self, 
               model: object, 
               m: Any, 
               z: Any,
               overdensity: Any = None,
               grid: bool = False,
               **kwargs,              ) -> Any:
        r"""
        Return the halo concentration as function of mass m.
        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any = None, 
                 grid: bool = False,
                 **kwargs,              ) -> Any:
        r"""
        Return the halo concentration as function of mass m.
        """
        
        return self.cmreln(model, m, z, overdensity, grid, **kwargs)


class DensityProfile(ABC):
    r"""
    Base class representing a halo density profile.
    """

    __slots__ = 'cm', 

    def __init__(self, cm: CMRelation) -> None:
        self.setCMRelation( cm )
        
    def setCMRelation(self, cm: CMRelation) -> None:
        r"""
        Set a c-M relation.
        """

        if not isinstance(cm, CMRelation):
            raise TypeError("cm must be 'CMRelation' instance")
        self.cm = cm
        return

    def c(self, 
          model: object, 
          m: Any, 
          z: Any, 
          overdensity: Any = None, 
          grid: bool = False, 
          **kwargs,              ) -> Any:
        r"""
        Return the halo consentration as function of mass m.
        """
        
        return self.cm(model, m, z, overdensity, grid, **kwargs)
    
    def __call__(self, 
                 model: object,
                 arg: Any,
                 m: Any, 
                 z: Any, 
                 overdensity: Any = None,
                 ft: bool = False,
                 truncate: bool = False,
                 grid: bool = False,
                 **kwargs,              ) -> Any:
        r"""
        Return the halo profile function in real or fourier space. 
        """
        
        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )
        arg  = np.reshape(arg, np.shape(arg) + tuple(1 for _ in m.shape))
        
        # concentration 
        c = self.c(model, m, z, overdensity, grid = False, **kwargs)
        
        # virial radius
        rvir = model.lagrangianR(m, overdensity) / (1 + z)

        # fourier transform
        if ft:
            res = self.u(arg * rvir, c) * c**3 / self.A(c)
            return res
        
        # real density
        res = self.f(arg / rvir, c)
        res = overdensity * model.Om0 * np.add(z, 1)**3 * c**3 / self.A(c) * res
        if truncate:
            res = np.where(np.less_equal(arg, rvir), res, 0.0)
        return res

    @abstractmethod
    def A(self, c: Any) -> Any:
        r"""
        Function related to the virial mass calculations.
        """
        ...

    @abstractmethod
    def f(self, x: Any, c: Any) -> Any:
        r"""
        Real space profile of the halo as function of distance from centre. 
        """
        ...

    @abstractmethod
    def u(self, q: Any, c: Any) -> Any:
        r"""
        Fourier space profile of the halo as function of wavenumber.
        """
        ...    
