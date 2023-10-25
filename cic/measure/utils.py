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

EPSILON = 1e-06 # tolerence to match two floats
UNIT_DEGREE_IN_RADIAN = 0.017453292519943295 # 1 deg in rad: degree to radian conversion factor
UNIT_RADIAN_IN_DEGREE = 57.29577951308232    # 1 rad in deg: radian to degree conversion factor


def getNearestPrettyNumber(value: float, round: bool = False) -> float:
    r"""
    Get a nice number closer to the given number.
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

class ShapeError( Exception ):
    r"""
    Base class of exceptions raised by shape objects.
    """

class Shape2D( ABC ):
    r"""
    Base class representing a 2D shape.
    """

    @abstractmethod
    def intersect(self, other: 'Shape2D') -> bool:
        r"""
        Check if the shape intersects with another shape. 
        """
        
    @abstractmethod
    def join(self, other: 'Shape2D') -> 'Shape2D':
        r"""
        Return the shape containing the two shapes.
        """

    @abstractmethod
    def area(self) -> float:
        r"""
        Return the area of the shape.
        """

    @abstractmethod
    def __eq__(self, other: 'Shape2D') -> bool:
        r"""
        Chack if the two shapes are same. 
        """

    @abstractmethod
    def to_json(self) -> object:
        r"""
        Return the JSON equivalent of this shape object.
        """


class Rectangle( Shape2D ):
    r"""
    A 2D rectangle class.

    Parameters
    ----------
    x_min, y_min: float
        Lower left coordinates of the rectangle region.
    x_max, y_max: float
        Upper right coordinates of the rectangle region.

    Examples
    --------
    >>> Rectangle(0.0, 0.0, 1.0, 1.0) # create a square 
    Rectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    """

    __slots__ = 'x_min', 'y_min', 'x_max', 'y_max'

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        
        # check if arguments are correct 
        if x_max - x_min <= 0:
            raise ShapeError( "shape has non-positive width along x direction" )
        elif y_max - y_min <= 0.:
            raise ShapeError( "shape has non-positive width along y direction" )
        
        self.x_min, self.x_max = float( x_min ), float( x_max )
        self.y_min, self.y_max = float( y_min ), float( y_max )

    def intersect(self, other: 'Rectangle') -> bool:

        if not isinstance(other, Rectangle):
            raise ShapeError("other is not a 'Rectangle' object")

        return not ( self.x_min >= other.x_max 
                        or self.x_max <= other.x_min 
                        or self.y_max <= other.y_min 
                        or self.y_min >= other.y_max )
    
    def join(self, other: 'Rectangle') -> 'Rectangle':
        
        if not isinstance(other, Rectangle):
            raise ShapeError("other is not a 'Rectangle' object")
        
        return Rectangle(x_min = min( self.x_min, other.x_min ), 
                         y_min = min( self.y_min, other.y_min ), 
                         x_max = max( self.x_max, other.x_max ), 
                         y_max = max( self.y_max, other.y_max ))
    
    def area(self, sphere: bool = False, degree: bool = True) -> float:
        r"""
        Return the area of the shape. The optional argument `sphere` tells if the 
        rectangle is on a unit sphere (default = False). The other optional argument 
        `degree` tells if the coordinates are in degree units (defaul = True) and is 
        only used is `sphere=True`.
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
    
    def __eq__(self, other: Shape2D) -> bool:

        if not isinstance(other, Rectangle):
            return NotImplemented
        
        return ( abs( self.x_min - other.x_min ) <= EPSILON 
                    and abs( self.x_max - other.x_max ) <= EPSILON 
                    and abs( self.y_min - other.y_min ) <= EPSILON 
                    and abs( self.y_max - other.y_max ) <= EPSILON )
    
    def to_json(self) -> object:
        
        obj = { "x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max }
        return obj
    
    def __repr__(self) -> str:

        return f"Rectangle(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"
    
    @classmethod
    def create(cls, obj: Any) -> 'Rectangle':
        r"""
        Create a `Rectangle` from any rectangle-like objects (array fo four floats).
        """

        if not isinstance( obj, Rectangle ):

            # try to unpack the object as 4 numbers, otherwise raise error
            try:
                return Rectangle( *obj )
            except Exception as _e:
                raise ShapeError( f"cannot create 'Rectangle' object. {_e}" )
            
        return obj


# TODO: circle shape


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
    region: Rectangle
        The rectangular region where the counting is done.
    patches: list[Rectangle]
        List of patches (sub-regions).
    patch_flags: list[bool]
        Flag telling if the sub-region is good (that is, not overlapping with any unwanted 
        areas in the region). 
    pixsize: float  
        Cellsize used for counting.
    patchsize_x, patchsize_y: float
        Size of the sub-regions.

    """

    __slots__ = 'region', 'patchsize_x', 'patchsize_y', 'pixsize', 'patches', 'shape', 'patch_flags', 'data'

    def __init__(self, 
                 region: Rectangle, 
                 patches: list[Rectangle], 
                 patch_flags: list[bool],
                 pixsize: float,
                 patchsize_x: float, 
                 patchsize_y: float,     ) -> None:
        
        self.region  = Rectangle.create( region )
        self.patches = [ Rectangle.create( patch ) for patch in patches ]

        if patchsize_x <= 0 or patchsize_y <= 0:
            raise TypeError( f"patch sizes must be positive" )
        self.patchsize_x, self.patchsize_y = float( patchsize_x ), float( patchsize_y ) 

        if pixsize <= 0:
            raise TypeError( f"pixsiz must be positive" )
        self.pixsize = pixsize

        if len( patch_flags ) != len( patches ):
            raise TypeError( "patch_flags should have same size as the patches list" )
        self.patch_flags = list( map( bool, patch_flags ) )

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
                "pixsize"    : self.pixsize,
                "patchsize_x": self.patchsize_x, 
                "patchsize_y": self.patchsize_y, 
              }
        
        obj["shape"] = self.shape

        data = {}
        for label, value in self.data.items():
            data[label] = list( map( float, value.flatten() ) )
        obj["data"] = data

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

        res = CountResult(region      = Rectangle( **obj['region'] ), 
                          patches     = [ Rectangle( **rect ) for rect in obj['patches'] ],
                          patch_flags = obj['patch_flags'], 
                          pixsize     = obj['pixsize'],
                          patchsize_x = obj['patchsize_x'],
                          patchsize_y = obj['patchsize_y'],   )
        
        res.shape = tuple( obj['shape'] )

        for label, values in obj['data'].items():
            res.data[label] = np.reshape( values, res.shape )

        return res
    
    @property
    def labels(self) -> list[str]:
        return list( self.data.keys() )
    
    def get(self, label: str) -> Any:
        r"""
        Get the data with given label. Return None if not exist.
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
            Patchwise statistics.

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
                binsize = getNearestPrettyNumber( width / (bins -1) )

                # nice values for the bounds
                lower = np.floor( lower / binsize ) * binsize
                upper = np.ceil( upper / binsize ) * binsize

            else:
                lower, upper = xrange 
        
            bins = np.linspace( lower, upper, bins )

        if np.ndim(bins) != 1:
            raise TypeError("bins must be an int or 1d array")
        
        hist = []
        bins = np.asfarray( bins )
        for i in range( data.shape[-1] ):
            hist_i, _ = np.histogram( data[..., i].flatten(), bins = bins, density = density )
            hist.append( hist_i )
        hist = np.stack( hist, axis = -1 )

        return _HistogramResult( bins = bins, hist = hist )





