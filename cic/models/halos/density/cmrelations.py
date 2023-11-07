#!/usr/bin/python3

r"""

Some relations connecting halo concentration parameter :math:`c` to its mass.

"""

import numpy as np
from typing import Any
from ._base import CMRelation

available_models = {}

class Bullock01(CMRelation):
    r"""
    Concentration mass relation given by Bullock et. al. (2001). 
    """

    __slots__ = 'F', 'K'

    def __init__(self, F: float = 0.01, K: float = 4.0) -> None:
        self.F, self.K = F, K

    def cmreln(self, 
               model: object, 
               m: Any, 
               z: Any, 
               overdensity: Any = None, 
               grid: bool = False,
               **kwargs,              ) -> Any:
        
        F, K = np.cbrt( self.F ), self.K

        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )
        
        r  = model.lagrangianR(m, overdensity)

        # collapse redshift
        zc = model.collapseRedshift( np.ravel( F*r ), **kwargs ).reshape( r.shape ) 
        zc = np.where( zc < z, z, zc )
        return K * (1 + zc) / (1 + z)
    
bullock01 = Bullock01()
available_models['bullock01'] = bullock01
    

class Zheng07(CMRelation):
    r"""
    Concentration mass relation given by Zheng et. al. (2017). Specified for redshifts 0 and 1.
    """

    __slots__ = 'c0', 'beta'

    def __init__(self, c0: float = 11., beta: float = -0.13) -> None:
        self.c0, self.beta = c0, beta

    def cmreln(self, 
               model: object, 
               m: Any, 
               z: Any, 
               overdensity: Any = None, 
               grid: bool = False, 
               **kwargs,              ) -> Any:
        

        if grid:
            assert np.ndim( m ) <= 1 and np.ndim( z ) <= 1
            m, z = np.ravel( m )[:, None], np.ravel( z )
        m, z = np.broadcast_arrays( m, z )
        
        # non-linear mass at z = 0
        mstar = model.lagrangianM(model.collapseRadius(0., **kwargs), overdensity)

        c = self.c0 * (m / mstar)**self.beta * (1 + z)**-1 
        return c

zheng07 = Zheng07()
available_models['zheng07'] = zheng07
