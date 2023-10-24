#!/usr/bin/python3

import json, os
import numpy as np, pandas as pd
import numpy.random as rnd
from abc import ABC, abstractmethod
from typing import Any


# error codes: 
WARNING = 2 # warnings 
ERROR   = 1 # error / failure 
SUCCESS = 0 # success or other flags

EPSILON = 1e-06 # tolerence to match two floats
UNIT_DEGREE_IN_RADIAN = 0.017453292519943295 # 1 deg in rad: degree to radian conversion factor
UNIT_RADIAN_IN_DEGREE = 57.29577951308232    # 1 rad in deg: radian to degree conversion factor


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
        Create a `Rectangle` from rectangle-like objects.
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

class CountResult:
    r"""
    An object to store count-in-cells data. 
    """

    __slots__ = 'region', 'patchsize_x', 'patchsize_y', 'pixsize', 'patches', 'shape', 'patch_flags', 'data'

    def __init__(self, 
                 region: Rectangle, 
                 patches: list[Rectangle], 
                 patch_flags: list[bool],
                 pixsize: float,
                 patchsize_x: float, 
                 patchsize_y: float,) -> None:
        
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
        Save the results in a JSON format.
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
        Load a saved result from a JSON file.
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

#############################################################################################################
# Helper functions
#############################################################################################################

def generateMockCatalog(size: int = 10_000):
    r"""
    Generate a mock catalog of objects in the rectangular region [0, 0, 1, 1]. This will 
    have two types of masks (u, v), each with 3-10 mask objects of radius between 0.1 and 2; 
    two types of magnitudes with standard normal distribution; and a redshift with standard 
    exponential distribution. Scale and shift the values to get catalogs of different values.  
    """

    # position
    x = rnd.uniform( low = 0., high = 1., size = size ) # x coordinate
    y = rnd.uniform( low = 0., high = 1., size = size ) # y coordinate

    # magnitudes
    mag_u = rnd.normal( loc = 0., scale = 1., size = size ) # type u
    mag_v = rnd.normal( loc = 0., scale = 1., size = size ) # type v

    # redshift 
    redshift = rnd.exponential( scale = 1., size = size )

    # type u mask
    n_masks = rnd.randint( 3, 10 )
    mask_xy = rnd.uniform( low = 0.,   high = 1.,  size = [ n_masks, 2 ] ) # mask positions
    mask_r  = rnd.uniform( low = 0.01, high = 0.1, size = n_masks )        # mask size  
    mask_u  = np.zeros( size, dtype = 'bool' ) # type u
    for i in range( n_masks ):
        mask_u = mask_u | ( ( x - mask_xy[i,0] )**2 + ( y - mask_xy[i,1] )**2 < mask_r[i]**2 )

    # type v mask
    n_masks = rnd.randint( 3, 10 )
    mask_xy = rnd.uniform( low = 0.,   high = 1.,  size = [ n_masks, 2 ] ) # mask positions
    mask_r  = rnd.uniform( low = 0.01, high = 0.1, size = n_masks )        # mask size  
    mask_v  = np.zeros( size, dtype = 'bool' ) # type u
    for i in range( n_masks ):
        mask_v = mask_v | ( ( x - mask_xy[i,0] )**2 + ( y - mask_xy[i,1] )**2 < mask_r[i]**2 )

    df = pd.DataFrame({'x'          : x,
                       'y'          : y,
                       'redshift'   : redshift,
                       'u_magnitude': mag_u, 
                       'v_magnitude': mag_v, 
                       'u_mask'     : mask_u, 
                       'v_mask'     : mask_v,  })
    return df
