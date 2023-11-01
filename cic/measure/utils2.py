#!/usr/bin/python3

import json, os
import numpy as np, pandas as pd
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any


# error codes: 
WARNING = 2 # warnings 
ERROR   = 1 # error / failure 
SUCCESS = 0 # success or other flags

EPS = 1e-06 # tolerence to match two floats
UNIT_DEGREE_IN_RADIAN = 0.017453292519943295 # 1 deg in rad: degree to radian conversion factor
UNIT_RADIAN_IN_DEGREE = 57.29577951308232    # 1 rad in deg: radian to degree conversion factor

class MeasurementError( Exception ):
    r"""
    Base class of exceptions raised during failure in measurements.    
    """

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

    def __repr__(self) -> str:
        return f"Box(min={self.min!r}, max={self.max!r})"
    
    @property
    def dim(self) -> int: return np.size( self.min )

    def intersect(self, other: 'Shape') -> bool:

        if not isinstance(other, Shape):
            raise ShapeError("other must be an instance of 'Shape'")

        if not isinstance(other, Box):
            raise NotADirectoryError()
        
        if not self.samedim( other ):
            raise ShapeError("both boxes should have same dimensions")

        return not np.any( np.logical_or( self.min >= other.max, self.max <= other.min ) )
    
    def join(self, other: 'Box', cls: type = None) -> 'Box':

        if not isinstance(other, Box):
            raise ShapeError("other must be an instance of 'Box'")
        
        if not self.samedim( other ):
            raise ShapeError("both boxes should have same dimensions")
        
        _min = np.min([self.min, other.min], axis = 0)
        _max = np.max([self.max, other.max], axis = 0)

        if cls is None:
            cls = self.__class__
        elif not issubclass(cls, Box):
            raise TypeError("cls must be a subclass of 'Box'")
        
        # only works if the subclass __init__ method take arguments as __init__(self, min, max) 
        # or __init__(self, *min, *max) otherwise, re-define the method in the subclass
        try:
            return cls( _min, _max )
        except Exception:
            return cls( *_min, *_max )

    def volume(self) -> float:
        return np.prod( self.max - self.min )
    
    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Box):
            return NotImplemented
        
        if not self.samedim( other ):
            return False
        
        return np.all(np.logical_and(np.abs(self.min - other.min) <= EPS, 
                                     np.abs(self.max - other.max) <= EPS, ))
    
    def to_json(self) -> object:
        return dict( min = self.min.tolist(), max = self.max.tolist() )
    
    @classmethod
    def create(cls, obj: Any) -> 'Box':
        r"""
        Create a box from another box or box-like object (array or dict). Used for creating 
        an instance from a JSON representation.

        Parameters
        ----------
        obj: Subclass of Box, array_like, dict
            If obj is an array, it should be an array of input arguments as they appear in 
            the class' __init__ method, if it is a dict, it must a mapping of keyword name 
            and value. 

        Returns
        -------
        res: subclass of Box

        Examples
        --------
        >>> b1 = Box.create([[0., 0.], [1., 1.]])
        >>> b2 = Box.create({'min': [0., 0.], 'max':[1., 1.]})

        """
        
        if not isinstance( obj, Box ):
            try:
                if isinstance( obj, dict ):
                    return cls( **obj )
                return cls( *obj )
            except Exception as e:
                raise ShapeError(f"cannot create box, {e}")
        
        return obj
    

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
    >>> b = Rectangle(0., 0., 1., 1.)
    Rectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    """

    def __init__(self, 
                 x_min: float, 
                 y_min: float, 
                 x_max: float, 
                 y_max: float, ) -> None:

        if any( map( lambda x: np.ndim(x) > 0, [x_min, x_max, y_min, y_max] ) ):
            raise ShapeError("arguments must be scalars")
        
        super().__init__([x_min, y_min], [x_max, y_max])

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
        if not sphere:
            return x_width * y_width
        
        # converting degree to radian
        if degree:
            x_width, y_width = x_width * UNIT_DEGREE_IN_RADIAN, y_width * UNIT_DEGREE_IN_RADIAN
        
        res = 4 * np.arcsin( np.tan( 0.5 * x_width ) * np.tan( 0.5 * y_width ) ) # in rad**2

        # converting area back to deg**2
        if degree:
            res = res * UNIT_RADIAN_IN_DEGREE**2

        return res
    
    area = volume # volume means area 2d

    @classmethod
    def create(cls, obj: Any) -> 'Rectangle':
        if isinstance(obj, Box):
            if obj.dim != 2:
                raise ShapeError("obj is not 2 dimensional")
            return Rectangle( *obj.min, *obj.max )
        return super().create(obj)

    
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
    >>> b = Box3D(0., 0., 0., 1., 1., 1.)
    Box3D(x_min=0.0, y_min=0.0, z_min=0.0, x_max=1.0, y_max=1.0, z_max=1.0)

    """

    def __init__(self, 
                 x_min: float, 
                 y_min: float, 
                 z_min: float, 
                 x_max: float, 
                 y_max: float, 
                 z_max: float, ) -> None:

        if any( map( lambda x: np.ndim(x) > 0, [x_min, x_max, y_min, y_max, z_min, z_max] ) ):
            raise ShapeError("arguments must be scalars")
        
        super().__init__([x_min, y_min, z_min], [x_max, y_max, z_max])

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
    
    @classmethod
    def create(cls, obj: Any) -> 'Box3D':
        if isinstance(obj, Box):
            if obj.dim != 3:
                raise ShapeError("obj is not 3 dimensional")
            return Box3D( *obj.min, *obj.max )
        return super().create(obj)

    
#############################################################################################################
# Storage 
#############################################################################################################

_HistogramResult = namedtuple( 'HistogramResult', ['bins', 'hist'] )

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

    __slots__ = 'region', 'patchsize', 'pixsize', 'patches', 'shape', 'patch_flags', 'data'

    def __init__(self, 
                 region: Rectangle, 
                 patches: list[Shape], 
                 patch_flags: list[bool],
                 pixsize: float,
                 patchsize: list[float], ) -> None:
        
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

        self.data  = {}   # dict to store labelled data
        self.shape = None # shape of the data

    def add(self, value: Any, label: str) -> 'CountResult':
        r"""
        Add new data to the result and return the object itself.

        Parameters
        ----------
        value: array_like
            Data to be added.
        label: str
            Name or label of the data.

        Returns
        -------
        res: CountResult
            A reference to the result object.

        """
        
        if label in self.data.keys():
            raise ValueError( f"label '{ label }' already exist" )
        
        value = np.asfarray( value )
        if self.shape is None:
            self.shape = value.shape

        # check if the new data have similar shape 
        elif value.shape != self.shape:
            raise TypeError( f"data has incorrect shape. must be { self.shape }" )

        self.data[ label ] = value
        return self

    def save(self, path: str) -> None:
        r"""
        Save the results to a file `path` in a JSON format.
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

        with open( path, 'w' ) as file:
            json.dump(obj, file, indent = 4)

        return  

    @classmethod
    def load(cls, path: str) -> 'CountResult':
        r"""
        Load a saved result from a JSON file `path`.
        """

        if not os.path.exists( path ):
            raise FileNotFoundError( f"path '{ path }' does not exist" )
        
        # load in JSON format
        with open( path, 'r' ) as file:
            obj = json.load( file )

        res = CountResult(region      = obj['region'], 
                          patches     = obj['patches'],
                          patch_flags = obj['patch_flags'], 
                          pixsize     = obj['pixsize'],
                          patchsize   = obj['patchsize'], )
        
        res.shape = tuple( obj['shape'] )

        for label, values in obj['data'].items():
            res.data[label] = np.reshape( values, res.shape )

        return res
    
    @property
    def labels(self) -> list[str]: return list( self.data.keys() )
    
    def get(self, label: str) -> Any:
        r"""
        Get the data with given label. Return None if not exist.

        Parameters
        ----------
        label: str

        Returns
        -------
        res: array_like

        """

        return self.data.get( label )
    
    def statistics(self, label: str) -> pd.DataFrame:
        r"""
        Return the patchwise descriptive statistics of a data. 

        Parameters
        ----------
        label: str
            Name of the data. If not exist, raises a `KeyError`.

        Returns
        -------
        stats: pandas.DataFrame

        """

        data = self.get( label )
        if data is None:
            raise KeyError( "missing data '%s'" % label )
        
        from scipy.stats import describe

        stats = []
        for i in range( data.shape[-1] ):
            xi  = data[..., i].flatten() 
            res = describe( xi )
            stats.append([res.nobs, 
                          res.mean, 
                          res.variance, 
                          res.skewness, 
                          res.kurtosis,
                          *np.percentile( xi, q = [0., 25., 50., 75., 100.] ) ])

        stats = pd.DataFrame(stats, 
                             columns = ['count', 'mean', 'variance', 'skewness', 'kurtosis',
                                        'min', '25%', '50%', '75%', 'max'                  ])
        return stats

    def histogram(self, 
                  label: str, 
                  bins: Any = 21, 
                  xrange: Any = None, 
                  density: bool = False ) -> _HistogramResult:
        r"""
        Return the patchwise histogram of a data.

        Parameters
        ----------
        label: str
            Name of the data to use. If not exist, raise exception.
        bins: int or 1D array_like
            Number of bins (int) or bins edges (1D array of floats) to use.
        xrange: bool, optional
            Range to use for counting. If not given and bins is a int, range is taken from 
            data limits.
        density: bool, optional
            If set true, normalise the histogram to a probability density function. Default 
            is false.

        Returns
        -------
        res: _HistogramResult
            A tuple of bin edges and histogram values.

        """

        data = self.get( label )
        if data is None:
            raise KeyError( "missing data '%s'" % label )
        
        if isinstance(bins, int): # find bin edges with given number of bins

            if xrange is None: # get bounds from the data

                # bounds of the data
                lower, upper = np.min( data ), np.max( data )

                # a nice value for the width
                width = getNearestPrettyNumber( upper - lower )

                # a nice bin size
                binsize = getNearestPrettyNumber( width / bins )

                # nice values for the bounds
                lower = np.floor( lower / binsize ) * binsize
                upper = np.ceil( upper / binsize ) * binsize

            else:
                lower, upper = xrange 
        
            bins = np.linspace( lower, upper, bins + 1 )

        if np.ndim( bins ) != 1:
            raise TypeError("bins must be an int or 1d array")
        
        hist = []
        bins = np.asfarray( bins )
        for i in range( data.shape[-1] ):
            hist_i, _ = np.histogram( data[..., i].flatten(), bins = bins, density = density )
            hist.append( hist_i )
        hist = np.stack( hist, axis = -1 )

        return _HistogramResult( bins = bins, hist = hist )



x = []
for i in range(3):
    x.append([])
    for j in range(4):
            x[i].append([])
            for k in range(5):
                    x[i][j].append(0) # k + 5*(j + 4*i)

# s = (3,4,5)
# for i in range(3):
#     for j in range(4):
#             for k in range(5):
#                     print( i,j,k,k + 5*(j + 4*i) )

s = np.array([3,4,5])
c = np.array([0,0,0])
for p in range(70):
    print(c, c[2] + 5*(c[1] + 4*c[0]))
    c[2] += 1
    for a in reversed(range(3)):
        if c[a] == s[a]:
            c[a] = 0
            c[a-1] += 1
        else:
            break



# print(np.array(x))
# print(np.arange(3*4*5).reshape((3,4,5)))