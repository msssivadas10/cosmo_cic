#!/usr/bin/python3

import numpy as np
from typing import Any
from ._base import CMRelation

builtinCMRelations = {}

class Bullock01(CMRelation):

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
builtinCMRelations['bullock01'] = bullock01
    

class Zheng07(CMRelation):

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
builtinCMRelations['zheng07'] = zheng07
