#!/usr/bin/python3

import warnings
import numpy  as np
from scipy.fftpack import dct
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

class IntegrationRule:
	r"""
	Base class representing an integration rule.
	"""
	__slots__ = 'a', 'b', 'pts', '_nodes', '_weights',

	def __init__(self) -> None:
		# integration nodes and weights
		self._nodes, self._weights = None, None
		# integration limits
		self.a, self.b = None, None
		# number of points
		self.pts = None
	
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
	