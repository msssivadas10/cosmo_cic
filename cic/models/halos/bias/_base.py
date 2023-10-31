#!/usr/bin/python3

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class HaloBias(ABC):
    r"""
    Base class representing a halo bias function object.
    """

    @abstractmethod
    def f(self, 
          model: object, 
          nu: Any, 
          z: Any, 
          overdensity: Any, ) -> Any:
        r"""
        Calculate the halo bias function values.
        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any, 
                 grid: bool = False, ) -> Any:
        r"""
        Calculate the halo bias function values as function of mass.
        """

        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )

        r  = model.lagrangianR( m, overdensity )
        nu = model.peakHeight(r, z, 
                              grid = False, 
                              exact = not model.settings.useInterpolation ) # delta_c / sigma(r)

        res = self.f( model, nu, z, overdensity )
        return res