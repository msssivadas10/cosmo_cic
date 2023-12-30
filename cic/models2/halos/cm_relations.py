#!/usr/bin/python3

import numpy as np
from typing import Any 
from .._base import Cosmology, HaloConcentrationMassRelation

class Zheng07(HaloConcentrationMassRelation):
    r"""
    Concentration mass relation given by Zheng et. al. (2017). Specified for redshifts 0 and 1.
    """

    def __init__(self, c0: float = 11.0, beta: float = -0.13) -> None:
        super().__init__()
        # model parameters
        self.c0, self.beta = c0, beta

    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        m   = np.asfarray(m)
        zp1 = np.add(z, 1)
        # non-linear mass at z = 0
        mstar = model.lagrangianM(model.collapseRadius(0.), overdensity)
        # concentration
        res = self.c0 * (m / mstar)**self.beta / zp1 
        return res

# initialising models to be readily used
zheng07 = Zheng07()

_available_models__ = {'zheng07': zheng07, }

