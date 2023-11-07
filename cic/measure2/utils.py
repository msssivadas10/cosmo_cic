#!/usr/bin/python3

import os
import json, gzip
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any

EPS = 1e-06 # tolerence to match two floats
UNIT_DEGREE_IN_RADIAN = 0.017453292519943295 # 1 deg in rad: degree to radian conversion factor
UNIT_RADIAN_IN_DEGREE = 57.29577951308232    # 1 rad in deg: radian to degree conversion factor

class MeasurementError( Exception ):
    r"""
    Base class of exceptions raised during failure in measurements.    
    """

_ParalellProcessInfo = namedtuple('_ParalellProcessInfo', ['comm', 'rank', 'size', 'err'])

def get_parellel_process_info(use_mpi: bool = True) -> _ParalellProcessInfo:
     r"""
     Get the parellel process communicator, rank and size, if mpi is enabled.
     """
     if not use_mpi: return _ParalellProcessInfo(None, 0, 1, 0)
     try: 
         from mpi4py import MPI
         comm = MPI.COMM_WORLD
         return _ParalellProcessInfo(comm, comm.rank, comm.size, 0)
     except ModuleNotFoundError: return _ParalellProcessInfo(None, 0, 1, 1)

def getNearestPrettyNumber(value: float, round: bool = False) -> float:
    r"""
    Get a nice number closer to the given number.

    Parameters
    ----------
    value: float
    round: bool

    Returns
    -------
    res: float

    """

    exponent = np.floor( np.log10( value ) )
    fraction = value / 10**exponent

    # move to some good number near to the fraction
    if round:
        fraction = 1. if fraction < 1.5 else ( 2. if fraction < 3. else ( 5. if fraction < 7. else 10. ) )
    else:
        fraction = 1. if fraction < 1. else ( 2. if fraction < 2. else ( 5. if fraction < 5. else 10. ) )

    return fraction * 10**exponent


#############################################################################################################
# Shapes
#############################################################################################################

class ShapeError(Exception):
    r"""
    Base class of exceptions raised by shape objects.
    """

class Shape(ABC):
    r"""
    Base class representing a 2D shape.
    """

    @property 
    @abstractmethod
    def dim(self) -> int: 
        r"""
        Number of dimensions the box existing.
        """
    
    def samedim(self, other: 'Shape') -> bool:
        r"""
        Check if the other shape has the same number of dimensions as this one.

        Parameters
        ----------
        other: subclass of Shape

        Returns
        -------
        res: bool

        """
        return self.dim == other.dim

    @abstractmethod
    def intersect(self, other: 'Shape') -> bool:
        r"""
        Check if the shape intersects with another shape. 

        Parameters
        ----------
        other: subclass of Shape

        Returns
        -------
        res: bool

        """
        
    @abstractmethod
    def join(self, other: 'Shape', cls: type = None) -> 'Shape':
        r"""
        Return the shape containing the two shapes.

        Parameters
        ----------
        other: subclass of Shape
        cls: type, optional
            Type of the returning shape. If not given, same as this shape.

        Returns
        -------
        res: subclass of Shape
            A shape containing both shapes.

        """

    @abstractmethod
    def volume(self) -> float:
        r"""
        Return the volume of the shape.

        Returns
        -------
        res: float

        """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        r"""
        Chack if the two shapes are same. 
        """

    @abstractmethod
    def to_json(self) -> object:
        r"""
        Return the JSON equivalent of this shape object.
        """


class Box(Shape):
    r"""
    A class representing a box in a general N-D space.

    Parameters
    ----------
    min, max: array_like
        Coordinates of the minimum and maximum corners of the box.

    Examples
    --------
    >>> b = Box([0., 0.], [1., 1.])
    >>> b
    Box(min=array([0., 0.]), max=array([1., 1.]))

    """

    __slots__ = 'min', 'max', 

    def __init__(self, min: Any, max: Any) -> None:

        if np.ndim( min ) > 1 or np.ndim( max ) > 1:
            raise ShapeError("min and max should be scalar or 1d arrays")
        elif np.ndim( min ) != np.ndim( max ):
            raise ShapeError("min and max should have same dimensions")
        elif np.size( min ) != np.size( max ):
            raise ShapeError("min and max should have same size")
        
        non_positive_axis, = np.where(np.less_equal( max, min ))
        if len(non_positive_axis):
            raise ShapeError(f"box has zero or negative width along direction { non_positive_axis[0] }")
        
        self.min = np.asfarray( min ) # minimum corner
        self.max = np.asfarray( max ) # maximum corner

    def __repr__(self) -> str: return f"Box(min={self.min!r}, max={self.max!r})"
    
    @property
    def dim(self) -> int: return np.size( self.min )

    def intersect(self, other: 'Shape') -> bool:
        if not isinstance(other, Shape): raise ShapeError("other must be an instance of 'Shape'")
        if not isinstance(other, Box): raise NotADirectoryError()
        if not self.samedim( other ): raise ShapeError("both boxes should have same dimensions")
        return not np.any( np.logical_or( self.min >= other.max, self.max <= other.min ) )
    
    def join(self, other: 'Box', cls: type = None) -> 'Box':
        if not isinstance(other, Box): raise ShapeError("other must be an instance of 'Box'")
        if not self.samedim( other ): raise ShapeError("both boxes should have same dimensions")
        
        _min = np.min([self.min, other.min], axis = 0)
        _max = np.max([self.max, other.max], axis = 0)

        if cls is None: cls = self.__class__
        elif not issubclass(cls, Box): raise TypeError("cls must be a subclass of 'Box'")
        
        # only works if the subclass __init__ method take arguments as __init__(self, min, max) 
        # or __init__(self, *min, *max) otherwise, re-define the method in the subclass
        try: return cls( _min, _max )
        except Exception: return cls( *_min, *_max )

    def volume(self) -> float: return np.prod( self.max - self.min )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box): return NotImplemented
        if not self.samedim( other ): return False
        return np.all(np.logical_and(np.abs(self.min - other.min) <= EPS, 
                                     np.abs(self.max - other.max) <= EPS, ))
    
    def to_json(self) -> object: return dict( min = self.min.tolist(), max = self.max.tolist() )
    
    @classmethod
    def create(cls, obj: Any) -> 'Box':
        r"""
        Create a box from another box or box-like object (array or dict). Used for creating 
        an instance from a JSON representation.

        Parameters
        ----------
        obj: object
            A list or dict with minimum and maximum coordinates, or another `Box` instance. 

        Returns
        -------
        res: subclass of Box

        Examples
        --------
        >>> b1 = Box.create([[0., 0.], [1., 1.]])
        >>> b2 = Box.create({'min': [0., 0.], 'max':[1., 1.]})

        """
        
        if isinstance(obj, dict): return cls(**obj)
        if np.ndim( obj ) == 2 and np.size( obj ) == 2: return cls(*obj)
        if isinstance( obj, Box ): return cls(obj.min, obj.max)
        raise ShapeError(f"cannot convert object to Box")
    

class Rectangle(Box):
    r"""
    A class representing a 2D rectangle on a flat or spherical space.

    Parameters
    ----------
    x_min, y_min: float
        Coordinates of the lower left corner of the rectangle.
    x_max, y_max: float
        Coordinates of the upper right corner of the rectangle.

    Attributes
    ----------
    Same as the input parameters.

    Examples
    --------
    >>> Rectangle([0., 0.], [1., 1.])
    Rectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    or, 

    >>> Rectangle.c(0., 0., 1., 1.)
    Rectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    """

    @classmethod
    def c(cls, 
          x_min: float, 
          y_min: float, 
          x_max: float, 
          y_max: float, ) -> None: return cls([x_min, y_min], [x_max, y_max])

    @property
    def x_min(self) -> float: return self.min[0]

    @property
    def y_min(self) -> float: return self.min[1]

    @property
    def x_max(self) -> float: return self.max[0]

    @property
    def y_max(self) -> float: return self.max[1]

    def __repr__(self) -> str:
        return f"Rectangle(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"
        
    def volume(self, 
               sphere: bool = False, 
               degree: bool = True, ) -> float:
        r"""
        Return the area of the rectangle. `area` is an alternate name for this method.

        Parameters
        ----------
        sphere: bool, optional
            Tells the rectangle is defined on a unit sphere. Default is false (flat).
        degree: bool, optional
            Tells the coordinates are degrees. Default is true. Only used if `sphere = True`.

        Returns
        -------
        res: float

        Examples
        --------
        >>> b = Rectangle(0., 0., 1., 1.)
        >>> b.volume()
        1.0
        >>> b.area() # alternate name
        1.0

        """
        
        x_width, y_width = self.x_max - self.x_min, self.y_max - self.y_min

        # area of flat rectangle: width * length
        if not sphere: return x_width * y_width
        
        # converting degree to radian
        if degree: 
            x_width, y_width = x_width * UNIT_DEGREE_IN_RADIAN, y_width * UNIT_DEGREE_IN_RADIAN
        
        res = 4 * np.arcsin( np.tan( 0.5 * x_width ) * np.tan( 0.5 * y_width ) ) # in rad**2

        # converting area back to deg**2
        if degree: 
            res = res * UNIT_RADIAN_IN_DEGREE**2

        return res
    
    area = volume # volume means area 2d

    
class Box3D(Box):
    r"""
    A class representing a 3D box.

    Parameters
    ----------
    x_min, y_min, z_min: float
        Coordinates of the lower left corner of the rectangle.
    x_max, y_max, z_max: float
        Coordinates of the upper right corner of the rectangle.

    Attributes
    ----------
    Same as the input parameters.

    Examples
    --------
    >>> Box3D([0., 0., 0.], [1., 1., 1.])
    Box3D(x_min=0.0, y_min=0.0, z_min=0.0, x_max=1.0, y_max=1.0, z_max=1.0)

    or,

    >>> Box3D.c(0., 0., 0., 1., 1., 1.)
    Box3D(x_min=0.0, y_min=0.0, z_min=0.0, x_max=1.0, y_max=1.0, z_max=1.0)

    """

    @classmethod
    def c(cls, 
          x_min: float, 
          y_min: float, 
          z_min: float, 
          x_max: float, 
          y_max: float, 
          z_max: float, ) -> None: return cls([x_min, y_min, z_min], [x_max, y_max, z_max])

    @property
    def x_min(self) -> float: return self.min[0]

    @property
    def y_min(self) -> float: return self.min[1]

    @property
    def z_min(self) -> float: return self.min[2]

    @property
    def x_max(self) -> float: return self.max[0]

    @property
    def y_max(self) -> float: return self.max[1]

    @property
    def z_max(self) -> float: return self.max[2]

    def __repr__(self) -> str:
        return f"Box3D(x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max})"


    
#############################################################################################################
# Storage 
#############################################################################################################

class CountResult:
    r"""
    An object to store count-in-cells data. This stores the counting set-up such as the region 
    used and cellsize. This is created without any data, but labelled data are later added using 
    the `CountResult.add` method. All added data should have the same shape.

    Parameters
    ----------
    region: subclass of Box
        The region (usually a box or rectangle) where the counting is done.
    patches: list[Shape]
        List of patches (sub-regions).
    patch_flags: list[bool]
        Flag telling if the sub-region is good (that is, not overlapping with any unwanted 
        areas in the region). 
    pixsize: array_like of float
        Cellsize used for counting.
    patchsize: array_like of float
        Size of the sub-regions.
    shape: type
        Specify the class of the region and patches. Default is a general `Box`.

    """

    __slots__ = 'region', 'patchsize', 'pixsize', 'patches', 'shape', 'patch_flags', 'data', 'extra_bins'

    def __init__(self, 
                 region: Rectangle, 
                 patches: list[Shape], 
                 patch_flags: list[bool],
                 pixsize: float,
                 patchsize: list[float], 
                 **extra_bins: list[float], ) -> None:
        
        self.region  = Box.create( region )
        self.patches = []
        for patch in patches:
            patch = Box.create( patch )
            if not patch.samedim( self.region ):
                raise ShapeError("patches should have same dimension as region")
            self.patches.append( patch )

        if len( patch_flags ) != len( patches ):
            raise TypeError( "patch_flags should have same size as the patches list" )
        self.patch_flags = list( map( bool, patch_flags ) )

        dim = self.region.dim

        if np.ndim( patchsize ) == 0:
            patchsize = np.repeat( patchsize, region.dim )
        elif np.ndim( patchsize ) > 1:
            raise TypeError("patchsize must be a scalar or 1d array")
        elif np.size( patchsize ) != dim:
            raise TypeError(f"patchsize must be a scalar or array of size {dim}")
        if np.any( np.less_equal( patchsize, 0. ) ):
            raise TypeError( f"all patchsizes must be positive" )
        self.patchsize = np.asfarray( patchsize )

        if np.ndim( pixsize ) == 0:
            pixsize = np.repeat( pixsize, region.dim )
        elif np.ndim( pixsize ) > 1:
            raise TypeError("pixsize must be a scalar or 1d array")
        elif np.size( pixsize ) != dim:
            raise TypeError(f"pixsize must be a scalar or array of size {dim}")
        if np.any( np.less_equal ( pixsize, 0. ) ):
            raise TypeError( f"all pixsizes must be positive" )
        self.pixsize = np.asfarray( pixsize )

        self.extra_bins = {}
        for feature, bins in extra_bins.items():
            if not np.ndim(bins) != 1: raise TypeError("bin edges should be 1d arrays")
            if np.size(bins) < 3: raise TypeError("bin edges must have at least 3 values")
            self.extra_bins[feature] = np.asfarray(bins)

        self.data  = {}   # dict to store labelled data
        self.shape = None # shape of the data

    def copy(self, include_data: bool = False) -> 'CountResult':
        r"""
        Return a copy of the results.

        Parameters
        ----------
        include_data: bool, optional
            Tells if to include data and extra bins info to the result (default = False).

        Returns
        -------
        res: CountResult

        """
        res = CountResult(region      = self.region, 
                          patches     = self.patches,
                          patch_flags = self.patch_flags, 
                          pixsize     = self.pixsize, 
                          patchsize   = self.patchsize, )
        
        if include_data:
            res.shape = self.shape
            for label, value in self.data.items(): res.data[label] = np.copy(value)
            for label, value in self.extra_bins.items(): res.extra_bins[label] = np.copy(value)

        return res

    def add(self, value: Any, label: str, replace: bool = False) -> 'CountResult':
        r"""
        Add new data to the result and return the object itself.

        Parameters
        ----------
        value: array_like
            Data to be added.
        label: str
            Name or label of the data.
        replace: bool, optional
            If true, replace the data if exist, else raise an error (default).

        Returns
        -------
        res: CountResult
            A reference to the result object.

        """
        
        if label in self.data.keys() and not replace:
            raise ValueError( f"label '{ label }' already exist" )
        
        value = np.asfarray( value )
        if self.shape is None: 
            self.shape = value.shape
        elif value.shape != self.shape: 
            raise TypeError( f"data has incorrect shape {value.shape}. must be { self.shape }" )

        self.data[ label ] = value
        return self
    
    def clear(self) -> None:
        r"""
        Remove all data from the object.
        """
        self.data.clear()
        self.shape = None

    def save(self, path: str, compress: bool = True) -> None:
        r"""
        Save the results to a file `path` in a JSON format.

        Parameters
        ----------
        path: str
        compress: bool, default = True

        """     

        # creating a JSON object...
        obj = {
                "region"     : self.region.to_json(), 
                "patches"    : [ patch.to_json() for patch in self.patches ],
                "patch_flags": self.patch_flags,
                "pixsize"    : self.pixsize.tolist(),
                "patchsize"  : self.patchsize.tolist(), 
                "shape_cls"  : type( self.region ).__name__,
              }
        
        obj["shape"] = self.shape        
        obj["data"]  = { label: value.tolist() for label, value in self.data.items() }

        obj['extra_bins'] = { label: value.tolist() for label, value in self.extra_bins.items() }

        if compress:
            with gzip.open( path, 'wb' ) as file: 
                file.write( json.dumps(obj, separators = (',',':')).encode('utf-8') )
        else:
            with open( path, 'w' ) as file: 
                json.dump(obj, file, indent = 4)

        return  

    @classmethod
    def load(cls, path: str, compress: bool = True) -> 'CountResult':
        r"""
        Load a saved result from a JSON file `path`.

        Parameters
        ----------
        path: str
        compress: bool, default = True

        """

        if not os.path.exists( path ):
            raise FileNotFoundError( f"path '{ path }' does not exist" )
        
        # load in JSON format
        if compress:
            with gzip.open( path, 'rb' ) as file: 
                obj = json.loads( file.read().decode('utf-8') )
        else:
            with open( path, 'r' ) as file:
                obj = json.load( file )

        res = CountResult(region      = obj['region'], 
                          patches     = obj['patches'],
                          patch_flags = obj['patch_flags'], 
                          pixsize     = obj['pixsize'],
                          patchsize   = obj['patchsize'], )
        
        if obj['shape'] is not None: res.shape = tuple( obj['shape'] )

        for label, values in obj['data'].items():
            if res.shape is not None: values = np.reshape( values, res.shape )
            res.data[label] = values

        for label, values in obj['extra_bins'].items():
            res.extra_bins[label] = np.asfarray(values)

        return res
    
    @property
    def labels(self) -> list[str]: return list( self.data.keys() )

    def get(self, label: str, *i: int) -> Any:
        r"""
        Get the data with given label. Return None if not exist.

        Parameters
        ----------
        label: str
        i: int, optional
            If given tell data along which axis are returned. Last axis is the patch.

        Returns
        -------
        res: array_like

        """
        data = self.data.get( label )
        axis = (..., *i)
        return data if i is None else data[axis]
    
    def __getitem__(self, __key: str | tuple) -> Any:
        if isinstance(__key, tuple): return self.get(*__key)
        return self.get(__key)
    
    
