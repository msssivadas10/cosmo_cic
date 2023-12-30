#!/usr/bin/python3

import numpy as np
from scipy.special import eval_hermite, hyp0f1
from typing import Any 
from .._base import WindowFunction

class SphericalTophat(WindowFunction):
    r"""
    Spherical top-hat filter.
    """
    def call(self, x: Any, deriv: int = 0) -> Any:
        # tophat function in terms of hypergeometric function 0f1 (to avoid zero division)
        def _g(x: Any, nu: int = 0, a: float = 2.5) -> Any:
            # top-hat function: w(x) = 3*( sin(x) - x * cos(x) ) / x**3
            if nu == 0: return hyp0f1(a, -0.25*x**2)
            # first derivative
            if nu == 1: return -x * hyp0f1(a + 1, -0.25*x**2) / (2*a)
            # second derivative
            if nu == 2: return (2*a - 1) / (2*a) * hyp0f1(a + 1, -0.25*x**2) - hyp0f1(a, -0.25*x**2)
            # general derivative. TODO: check this relation
            return (2*a - 1) / (2*a) * _g(x, nu - 2, a + 1) - _g(x, nu - 2, a)	
        
        if deriv < 0:
            raise ValueError(f"invalid value for argument 'deriv': {deriv}")
        res = _g(np.asfarray(x), deriv, 2.5)
        return res
 
class Gaussian(WindowFunction):
    r"""
    Gaussian function.
    """
    def call(self, x: Any, deriv: int = 0) -> Any:
        SQRT_2 = 1.4142135623730951
        if deriv < 0:
            raise ValueError(f"invalid value for argument 'deriv': {deriv}")
        x   = np.divide(x, SQRT_2)
        res = np.exp(-x**2)
        if deriv > 0:
            # derivatives in terms of hermite polynomials
            res = res * eval_hermite(deriv, x) / (-SQRT_2)**deriv
        return res


# initialising models to be readily used
tophat   = SphericalTophat()
gaussian = Gaussian()

_available_models__ = {'tophat'  : tophat,
                       'gaussian': gaussian, }
