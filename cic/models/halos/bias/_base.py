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
        Returns the halo bias value as function of :math:`\nu = \delta_c/\sigma`, based on the given 
        cosmology model.

        Parameters
        ----------
        model: Cosmology
        nu: array_like
        z: array_like
        overdensity: int

        Returns
        -------
        res: array_like

        """
        ...

    def __call__(self, 
                 model: object, 
                 m: Any, 
                 z: Any, 
                 overdensity: Any, 
                 grid: bool = False, ) -> Any:
        r"""
        Returns the halo bias values as function of mass, based on the given cosmology model.

        Parameters
        ----------
        model: Cosmology
        m: array_like
        z: array_like
        overdensity: float, None
        grid: bool, default = False
            If set true, evaluate the function on a grid of input arrays. Otherwise, they must be broadcastable.

        Returns
        -------
        res: array_like

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