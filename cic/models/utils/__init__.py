#!/usr/bin/python3

import re
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from random import choice
from abc import ABC, abstractmethod
from typing import Any, Callable


class Interpolator:
    r"""
    Base class for an interpolation table. 
    """

    __slots__ = 'spline', 'x', 'data', 'func'

    def __inti__(self, func: Callable, *args, **kwargs) -> None: ...
    def __call__(self, *args, **kwargs) -> Any: ...


class Interpolator1D(Interpolator):
    r"""
    A one variate interpolation table.
    """
    
    def __init__(self, 
                 func: Callable, 
                 xa: float, 
                 xb: float, 
                 xpts: int, 
                 fargs: tuple = (),
                 fkwargs: dict = {}, 
                 **kwargs          ) -> None:
        
        kwargs = { 's': 0, **kwargs } # set default s = 0 

        assert callable(func), "'func' must be a callable object"
        self.func = func

        # generate and save the data
        x = np.linspace(xa, xb, xpts)
        f = func(x, *fargs, **fkwargs)
        
        self.x, self.data = x, f

        # create a univariate spline with the data
        self.spline = UnivariateSpline(x, f, **kwargs)

    def __call__(self, x: Any, nu: int = 0) -> Any:
        return self.spline.__call__(x, nu = nu)


class Interpolator2D(Interpolator):
    r"""
    A two variable interpolations table. Used to interpolate values on a grid. 
    """
    
    def __init__(self, 
                 func: Callable, 
                 xa: float, 
                 xb: float, 
                 xpts: int,
                 ya: float, 
                 yb: float, 
                 ypts: int, 
                 fargs: tuple = (),
                 fkwargs: dict = {}, 
                 **kwargs          ) -> None:
        
        kwargs = { 's': 0, **kwargs } # set default s = 0 

        assert callable(func), "'func' must be a callable object"
        self.func = func

        # generate and save the data
        x = np.linspace(xa, xb, xpts)
        y = np.linspace(ya, yb, ypts)
        f = func(x, y, *fargs, **fkwargs)
        
        self.x, self.data = (x, y), f

        # create a univariate spline with the data
        self.spline = RectBivariateSpline(x, y, f, **kwargs)

    def __call__(self, x: Any, y: Any, dx: int = 0, dy: int = 0, grid: bool = False) -> Any:
        
        shape = None
        if grid:
            x, y  = np.meshgrid(x, y, indexing = 'ij')
            shape = x.shape
            x, y  = x.flatten(), y.flatten()

        res   = self.spline.__call__(x, y, dx = dx, dy = dy, grid = False)    
        if grid:
            res = res.reshape(shape)
        return res


def typestr(o: object, full: bool = False) -> str:
    r"""
    Return the class (type) name of the object.
    """
    
    m = re.search( '(?<=\<class \')[\.\w]+(?=\'>)', repr(type(o)) )
    if not m:
        raise ValueError("failed to get the class name")
    
    m = m.group(0)
    return m if full else m.rsplit('.', maxsplit = 1)[-1]

def randomString(size: int = 16) -> str:
    r"""
    Generate a random string of given size.
    """

    charecters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join([ choice( charecters ) for _ in range(size) ]) 
