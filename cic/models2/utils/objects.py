#!/usr/bin/python3

import warnings
import numpy  as np
from scipy.optimize import brentq
from scipy.fftpack import dct
from scipy.fft import irfft
from scipy.special import gamma
from dataclasses import dataclass
from typing import Any, Callable

class ModelDatabase:
    r"""
    An object to store some models as database. This then allow the user to map 
    models with names and get then easily!

    Parameters
    ----------
    name: str
    baseclass: type, tuple of type
        Type of the objects in the database.

    """
    __slots__ = '_db', 'baseclass', 'name'

    def __init__(self, name: str, baseclass: type | tuple[type]) -> None:
        # name of the database
        assert isinstance(name, str)
        self.name = name
        # base class of the objects in the database
        self.baseclass = baseclass
        # models are stored as key-value pairs in a dictionary
        self._db = {}

    def exists(self, key: str) -> bool:
        r"""
        Check if an item with given key `key` exist or not. 
        """
        return key in self._db

    def keys(self) -> list[str]:
        r"""
        Return the keys in the database.
        """
        return list( self._db.keys() )

    def add(self, key: str, value: object) -> None:
        r"""
        Add a model to the database and map to the given `key`. If `key` already exist, 
        an error is raised. Argument `value` must be of correct type.
        """
        if key in self.keys():
            raise ValueError(f"key already exists: '{key}'")
        if not isinstance(value, self.baseclass):
            raise TypeError(f"incorrect type: '{type(value)}'")
        self._db[key] = value
        return
    
    def remove(self, key: str) -> None:
        r"""
        Remove an item  with given key from the database.
        """
        if key not in self.keys(): return
        warnings.warn(f"removing item from the database: '{self.name}'")
        value = self._db.pop(key)
        return 
    
    def get(self, key: str) -> object:
        r"""
        Get an item with given key, from the database. If key not exist, return None.
        """
        return self._db.get(key, None)
    
###################################################################################################
#									     Numerical Tools										  #
###################################################################################################

# Itergration
class IntegrationRule:
    r"""
    Base class representing an integration rule.
    """
    __slots__ = 'a', 'b', 'pts', '_nodes', '_weights',

    def __init__(self) -> None:
        # integration nodes and weights
        self._nodes, self._weights = None, None
        # integration limits
        self.a: float = None
        self.b: float = None
        # number of points
        self.pts: int = None
    
    @property
    def nodes(self) -> Any: 
        r""" 
        Intergration points for this rule. 
        """
        return self._nodes
    
    @property
    def weights(self) -> Any: 
        r""" 
        Weight of this rule. They will have sum `(b-a)/2` 
        """
        return self._weights

    def __add__(self, other: object) -> 'IntegrationRule':
        if not isinstance(other, IntegrationRule):
            return NotImplemented
        return CompositeIntegrationRule(self, other)
    
    def integrate(self, 
                     func: Callable, 
                  args: tuple = None, 
                  axis: int = 0,    ) -> Any:
        r"""
        Integrate a function `func`.

        Parameters
        ----------
        func: callable
        args: tuple, default = None
            Additional arguments to function call.
        axis: int, default = 0
            Axis to use fot summation.

        Returns
        -------
        res: array_like

        """
        args = args or ()
        return np.sum( self.weights * func(self.nodes, *args), axis = axis )

class CompositeIntegrationRule(IntegrationRule):
    r"""
    A class representing a composite integration rule as left and right rules.

    Parameters
    ----------
    left, right: IntegrationRule

    """
    __slots__ = '_left', '_right', 'connected' 

    def __init__(self, left: IntegrationRule, right: IntegrationRule) -> None:
        super().__init__()
        if right.a < left.a: left, right = right, left
        if left.a < right.b and left.b > right.a: 
            raise ValueError(f"overlapping intervals: [{left.a}, {left.b}] and [{right.a, right.b}]")
        self._left, self._right = left, right	
        self.a, self.b = left.a, right.b
        self.connected = False
        self.pts = left.pts + right.pts
        if abs( left.b - right.a ) < 1e-08: 
            self.connected = True
            self.pts -= 1
    
    @property
    def nodes(self) -> Any:
        _nodes = np.zeros(self.pts)
        _nodes[:self._left.pts] = self._left.nodes
        offset = 1 if self.connected else 0
        _nodes[self._left.pts:] = self._right.nodes[offset:]
        return _nodes
    
    @property
    def weights(self) -> Any:
        _weights = np.zeros(self.pts)
        _weights[:self._left.pts] = self._left.weights
        offset = 0
        if self.connected:
            offset = 1
            _weights[self._left.pts-1] += self._right.weights[0]
        _weights[self._left.pts:] = self._right.weights[offset:]
        return _weights
    
    def integrate(self, func, args: tuple = None, axis: int = 0) -> Any:
        res = self._left.integrate(func, args, axis) + self._right.integrate(func, args, axis)
        return res

class DefiniteUnweightedCC(IntegrationRule):
    r"""
    A class representing an unweighted clenshaw-curtis integration rule over the interval 
    `[a, b]`. This can be used for integrations in general, but not useful when in the 
    function to integrate is highly oscillating.

    Parameters
    ----------
    a, b: float
        Limits of integration
    pts: int
        Number of points to use for integration

    """
    
    def __init__(self, a: float, b: float, pts: int) -> None:
        super().__init__()
        # make number of points even number greater than or equal to 2
        pts = 2 * (max(2, pts) // 2) # will be pts + 1 nodes
        # finding nodes and weights:
        n = np.arange(pts/2 + 1)
        # weights
        __w = dct( 1./( 1 - 4.*n**2 ), type = 1, ) * (2. / pts)
        __w[0] *= 0.5
        # nodes
        __x = np.cos(n*np.pi / pts)
        
        self._nodes   = np.concatenate([ -__x[:-1], __x[-1::-1] ]) 
        self._weights = np.concatenate([  __w[:-1], __w[-1::-1] ])
        # transform the nodes and weights into range [a, b]
        if b < a: a, b = b, a
        self._nodes   = 0.5*(b-a) * self._nodes + 0.5*(b+a)
        self._weights = 0.5*(b-a) * self._weights 
        self.a, self.b, self.pts = a, b, pts + 1 
        return

class SphericalHankelTransform(IntegrationRule):
    r"""
    A special integration rule to calculate spherical hankel transforms.

    Parameters
    ----------
    L: float
        Used to specify the integration limits and points. For `N` points, the points are 
        log spaced from `exp(-L/2)...exp(L/2)`, with spacing `exp(L/N)`.
    pts: int
        Number of points to use for integration
    k_max: int, default = 1
        Maximum order of transforms.  

    """

    def __init__(self, L: float, pts: int, k_max: int = 1) -> None:
        super().__init__()
        assert L > 0.
        assert isinstance(k_max, int) and k_max >= 0
        # log-spaced points
        __x = np.arange( pts )
        __x[ __x > pts // 2 ] -= pts
        __x = np.exp( __x*L/pts )
        self._nodes, self._weights = __x, []
        for k in range(k_max + 1):
            # weights for order k transform TODO: check the equations
            __w = np.arange( pts//2 + 1 ) * np.pi/L
            __w = np.exp(1.3862943611198906j*__w) * gamma(1. + 1j*__w) / gamma((2*k + 1)*0.5 - 1j*__w)
            __w = 2*(2*k+1) / 2**(k + 0.5) / (2*np.pi)**1.5  * irfft(__w)
            self._weights.append(__w)
        # integration limits
        self.a, self.b, self.pts = np.exp(-0.5*L), np.exp(0.5*L), pts

# Root finding
class RootFinder:
    r"""
    Base class to represent a root finding method. 

    Parameters
    ----------
    reltol: float, default = 1e-6
        Relative tolerance for convergenece.
    maxiter: int, default = 10000
        Maximum iterations to use.
    
    """
    __slots__ = 'default_a', 'default_b', 'reltol', 'maxiter', 'extra_args'

    def __init__(self, 
                 reltol: float = 1e-06, 
                 maxiter: int = 10_000, ) -> None:
        # error tolerance
        self.reltol = reltol
        # maximum iteration
        self.maxiter = maxiter

    def rootof(self, 
               func: Callable, 
               a: float | None = None, 
               b: float | None = None, 
               args: tuple = None, ) -> Any:
        r"""
        Return the root of a function.

        Parameters
        ----------
        func: callable
        a, b: float, optional
            Search interval. If not given use the default values used in the constructor.
        args: tuple, optional 
            Additional arguments passed to the function.

        Returns
        -------
        res: array_like
            Root of the function.

        """
        raise NotImplementedError()
    
class BrentQ(RootFinder):
    r"""
    A root finder based on `scipy.optimize.brentq`.

    Parameters
    ----------
    a, b: float, optional
        Search interval. If not given, must be specified in the `rootof` call.
    reltol: float, default = 1e-6
        Relative tolerance for convergenece.
    maxiter: int, default = 10000
        Maximum iterations to use.
    kwargs: dict, optional
        Other keyword arguments to the root finder
    
    """
    __slots__ = 'default_a', 'default_b', 'extra_args'

    def __init__(self, 
                   a: float | None = None, 
                 b: float | None = None, 
                 reltol: float = 1e-06, 
                 maxiter: int = 10_000, 
                 kwargs: dict = None, ) -> None:
        super().__init__(reltol, maxiter)
        # search interval
        self.default_a, self.default_b = a, b
        # additional keyword arguments for the root-finder
        self.extra_args = kwargs or {} 

    def rootof(self, 
               func: Callable, 
               a: float | None = None, 
               b: float | None = None, 
               args: tuple = None, ) -> Any:
        if a is None: a = self.default_a
        if b is None: b = self.default_b
        args = args or ()
        # vectorised function returning the roots (in case of array arguments)
        @np.vectorize
        def _rootof(*args):
            res = brentq(func, a, b, args = args, rtol = self.reltol, maxiter = self.maxiter, **self.extra_args)
            return res
        
        res = _rootof(*args)
        return res

@dataclass
class Settings:
    r"""
    A table of various settings for calculations.
    """
    reltol: float = 1e-06
    # redshift integration rule
    z_quad: IntegrationRule = DefiniteUnweightedCC(a = -1., b = 1., pts = 128)
    # k integration rule (points in log(k) space)
    k_quad: IntegrationRule = (DefiniteUnweightedCC(a = -14., b = -7., pts = 64 ) + # 1e-06 to 1e-03, approx. 
                               DefiniteUnweightedCC(a =  -7., b =  7., pts = 512) + # 1e-03 to 1e+03, approx. 
                               DefiniteUnweightedCC(a =   7., b = 14., pts = 64 ) ) # 1e+03 to 1e+06, approx. 
    # special integration rule for correlation function calculation (experimental feature)
    corr_quad: SphericalHankelTransform = SphericalHankelTransform(L = 38., pts = 128, k_max = 1)
    # mass integration rule (points in log(m) space)
    m_quad: IntegrationRule = (DefiniteUnweightedCC(a = 12., b = 42., pts = 512)) # 1e+6 to 1e+18, approx
    # root finder object solving for scale (in log(r) space)
    r_finder: RootFinder = BrentQ(a = -7., b = 7.) #1e-03 to 1e+03 approx