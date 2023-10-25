#!/usr/bin/python3

r"""
Generate a 2D Poisson point process with given features.
"""

import sys, time
import numpy as np
import numpy.random as rnd
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

###############################################################################################
# Masks
###############################################################################################

class Mask( ABC ):
    r"""
    A class representing a mask object. These masks are used to mark a masked region 
    of some shape.

    Parameters
    ----------
    loc: tuple[float]
        Position of the mask.
    size: float or tuple[float]
        Size of the mask.

    """

    __slots__ = 'loc', 'size', 

    def __init__(self, loc: tuple[float], size: Any) -> None:

        self.loc  = loc
        self.size = size
    
    @abstractmethod
    def isMasked(self, objects: list) -> list[bool]:
        r"""
        Tell if the points are inside the masked region or not.

        Parameters
        ----------
        objects: list of tuple[float]
            Positions of the points or objects.
        
        Returns
        -------
        res: list of bool
            A flag indicating whether the point is masked or not.

        """
        ...
    
    def __repr__(self) -> str:
        return f"Mask(loc={ self.loc }, size={ self.size })"


class CircularMask( Mask ):
    r"""
    A circular mask object.

    Parameters
    ----------
    loc: tuple[float]
        Center of the mask.
    size: float
        Radius of the mask.

    Examples
    --------
    >>> m = CircularMask(loc=(0.0, 0.0), size=1.0)
    >>> m.isMasked([(0.5, 0.0), (0.0, 0.0), (1.5, 1.2)])
    [True, True, False]

    """

    def isMasked(self, objects: list) -> list[bool]:
        
        assert np.ndim( objects ) == 2
        assert np.shape( objects )[1] == len( self.loc )

        # is inside circle := dist( object - center ) <= radius
        res = np.sum( np.subtract( objects, self.loc )**2., axis = -1 ) <= self.size**2
        return res
    
    def __repr__(self) -> str:
        return 'Circular' + super().__repr__()
    

class RectangularMask( Mask ):
    r"""
    A rectangular mask object.

    Parameters
    ----------
    loc: tuple[float]
        Lower left corner of the mask.
    size: float or tuple[float]
        Widths of the mask.

    Examples
    --------
    >>> m = RectangularMask(loc=(0.0, 0.0), size=(1.0, 1.0))
    >>> m.isMasked([(0.5, 0.0), (0.0, 0.0), (1.5, 1.2)])
    [True, True, False]

    """

    def isMasked(self, objects: list) -> list[bool]:
        
        assert np.ndim( objects ) == 2
        assert np.shape( objects )[1] == len( self.loc )

        # is inside rectangle := AND { start_x <= object_x  <= start_x + size_x } FOR ALL x
        res = np.logical_and(
                                np.less_equal( self.loc, objects ), 
                                np.less_equal( objects, np.add( self.loc, self.size ) ) 
                            ).min( axis = -1 )
        return res
    
    def __repr__(self) -> str:
        return 'Rectangular' + super().__repr__()
    

###############################################################################################
# Features
###############################################################################################

class FeatureGenerator( ABC ):
    r"""
    A class representing a feature of the objects. These feature generator objects 
    sample random values from a distribution and assign it to the object.

    Parameters
    ----------
    name: str
        Name of the feature.

    """

    __slots__ = 'name', 
    
    def __init__(self, name: str) -> None:

        assert isinstance( name, str ), "name must be a str"
        self.name = name

    @abstractmethod
    def generateValues(self, objects: list) -> list:
        r"""
        Generate a values for the given objects.

        Parameters
        ----------
        objects: list of tuple[floats]
            Position of the points or objects.

        Returns
        -------
        res: list[Any]
            Generated values.

        """
        ...


class NormalFeatureGenerator( FeatureGenerator ):
    r"""
    A feature with a normal distribution.

    Parameters
    ----------
    name: str
        Name of the feature.
    loc: float
        Mean of the distribution (default = 0).
    scale: float
        Spread of the distribution (default = 1).
        
    """

    __slots__ = 'loc', 'scale', 

    def __init__(self, name: str, loc: float = 0.0, scale: float = 1.0) -> None:

        self.loc, self.scale = loc, scale
        super().__init__( name )

    def generateValues(self, objects: list) -> list[float]:
        return rnd.normal( loc = self.loc, scale = self.scale, size = len( objects ) )
    

class ExponentialFeatureGenerator( FeatureGenerator ):
    r"""
    A feature with an exponential distribution.

    Parameters
    ----------
    name: str
        Name of the feature.
    scale: float
        Spread of the distribution (default = 1).
        
    """

    __slots__ = 'scale', 

    def __init__(self, name: str, scale: float = 1.0) -> None:
        
        self.scale = scale
        super().__init__( name )

    def generateValues(self, objects: list) -> list[float]:
        return rnd.exponential( scale = self.scale, size = len( objects ) )
    

class BooleanFeatureGenerator( FeatureGenerator ):
    r"""
    A feature with a boolean values. This usually tells the points in a masked 
    area or not.

    Parameters
    ----------
    name: str
        Name of the feature.
    masks: list[Mask]
        Masks used for generating the values.
        
    """

    __slots__ = 'masks', 

    def __init__(self, name: str, masks: list[Mask]) -> None:

        self.masks = []
        for mask in masks:
            assert isinstance( mask, Mask )
            self.masks.append( mask )

        super().__init__( name )

    def generateValues(self, objects: list) -> list[bool]:
        
        res = False
        for mask in self.masks:
            res = np.logical_or( res, mask.isMasked( objects ) )
        return res


###############################################################################################
# Catalog generation
###############################################################################################

def generateP3Catalog(x_min: float = 0.0, 
                      y_min: float = 0.0, 
                      x_max: float = 1.0, 
                      y_max: float = 1.0, 
                      density: float = 100.0, 
                      x_coord: str = 'x',
                      y_coord: str = 'y',
                      spherical: bool = False, 
                      degree: bool = True,
                      features: list[FeatureGenerator] = []) -> pd.DataFrame:
    r"""
    A random catalog with given specifications and features. The catalog will be a 
    2D poisson point process.

    Parameters
    ----------
    x_min, y_min, x_max, y_max: float
        Lower and upper bounds of the region where the points are generated. Default is 
        the unit square x_min = 0, y_min = 0, x_max = 1 and y_max = 1. 
    density: float
        Density of points (number of points per unit area). Default is 100.
    x_coord, y_coord: str
        Name of the x and y coordinates, as appeared in the result. Default is `x` and `y`.
    spherical: bool
        Tell if the points are on a unit sphere or not (default is false).
    degree: bool
        Tell if degree units are used (default is true). This used only if `spherical=True`.
    features: list[FeatureGenerator]
        List of generator objects corresponding to additional features / columns in the 
        data.

    Returns
    -------
    df: pandas.DataFrame
        Point process catalog. 

    """
    
    assert isinstance( x_coord, str )
    assert isinstance( y_coord, str )
    assert x_min < x_max
    assert y_min < y_max
    assert density > 0.
    for fetaure in features:
        assert isinstance( fetaure, FeatureGenerator )

    # area of the region
    if spherical:
        x_size, y_size = x_max - x_min, y_max - y_min 
        if degree: # convert to radian
            x_size, y_size = x_size * np.pi / 180. , y_size * np.pi / 180.
        
        area = 4 * np.arcsin( np.tan( 0.5 * x_size ) * np.tan( 0.5 * y_size ) ) # radian**2
        if degree: # convert to degree**2
            area = area * ( 180. / np.pi )**2
    else:
        area = ( x_max - x_min ) * ( y_max - y_min ) 
        

    # number of objects / points
    size = rnd.poisson( density * area )   

    # positions
    sys.stderr.write( f"Generating {size} points..." )       
    __start   = time.time()
    objectPos = rnd.uniform( low = [x_min, y_min], high = [x_max, y_max], size = [size, 2] )
    sys.stderr.write( f"completed in { time.time() - __start :.3g} sec.\n" )

    # other features
    df = {}
    for feature in features:
        sys.stderr.write( f"Generating {feature.name} values..." )
        df[ feature.name ] = feature.generateValues( objectPos )
        sys.stderr.write( f"completed in { time.time() - __start :.3g} sec.\n" )

    df = pd.DataFrame({ x_coord: objectPos[:,0], y_coord: objectPos[:,1], **df })

    sys.stderr.write( "Catalog generation complete!\n" )       
    return df
    

def p3Generator1(ra1: float, 
                 ra2: float, 
                 dec1: float, 
                 dec2: float, 
                 density: float = 100.0, 
                 degree: bool = True   ) -> pd.DataFrame:
    r"""
    Generates a catalog that simulate a galaxy survey in three bands: g, r ans i. The 
    resulting catalog contains the magnitudes and mask values in these bands and the 
    redshift information. 

    - Magnitudes are normally distributed with mean and spread g(22, 5), r(21, 4) and
      i(20, 4). Feature name is `g/r/i_magnitude`.
    - 20 random circular masks with an average size of 5% of the smallest region size. 
      Feature name is `g/r/i_mask`.
    - Exponentially distributed redshift feature `redshift` with scale 1.5.

    Parameters
    ----------
    ra1, ra2: float
        Limits for the right ascension in degree.
    dec1, dec2: float
        Limits for the declination in degree.
    density: float, optional
        Number of points per unit area (default is 100).
    degree: bool
        Tell if degree units are used (default is true).

    Returns
    -------
    df: pandas.DataFrame
        Point process catalog.

    Examples
    --------
    Generate a catalog with number density 500/deg^2 and save as CSV file:

    >>> df = p3Generator1(ra1 = 0., ra2 = 40., dec1 = 0., dec2 = 10., density = 500.)
    >>> df.to_csv('p3catalog.csv', index = False) 

    """

    
    # create masks
    av_size = min( ra2 - ra1, dec2 - dec1 ) * 0.05
    masks   = [CircularMask(loc  = rnd.uniform( low = [ra1, dec1], high = [ra2, dec2] ), 
                            size = abs( rnd.normal(loc = av_size, scale = 0.01 * av_size) ) 
                        ) for i in range(20)
            ]

    # features
    features = [NormalFeatureGenerator('g_magnitude', loc = 22., scale = 5.0),
                NormalFeatureGenerator('r_magnitude', loc = 21., scale = 4.0),
                NormalFeatureGenerator('i_magnitude', loc = 20., scale = 4.0), 
                BooleanFeatureGenerator('g_mask', masks),
                BooleanFeatureGenerator('r_mask', masks),
                BooleanFeatureGenerator('i_mask', masks),
                ExponentialFeatureGenerator('redshift', scale = 1.5)
            ]
    
    # catalog
    df = generateP3Catalog(x_min = ra1,
                           x_max = ra2,
                           y_min = dec1,
                           y_max = dec2,
                           density = density,
                           x_coord = 'ra',
                           y_coord = 'dec',
                           spherical = True,
                           degree = degree, 
                           features = features, )
    
    return df    
