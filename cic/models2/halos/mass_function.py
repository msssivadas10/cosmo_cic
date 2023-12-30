#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Any 
from .._base import MassFunction, Cosmology 

class Press74(MassFunction):
    r"""
    Halo mass function model by Press & Schechter (1974). It is based on spherical collapse.
    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None) -> Any:
        nu  = model.collapseDensity(z) / np.asfarray(s)
        res = np.sqrt( 2 / np.pi ) * nu * np.exp( -0.5 * nu**2 )
        return res
    
class Sheth01(MassFunction):
    r"""
    Halo mass function model by Sheth et al (2001). It is based on ellipsoidal collapse.
    """
    def __init__(self) -> None:
        super().__init__()
        # parameters
        self.A, self.a, self.p = 0.3222, 0.707, 0.3 

    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None) -> Any:
        # parameters
        A, a, p = self.A, self.a, self.p
        nu  = model.collapseDensity(z) / np.asarray(s)
        res = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
        return res

class Tinker08(MassFunction):
    r"""
    Halo mass function model by Tinker et al (2008). This model is redshift dependent.
    """

    def __init__(self) -> None:
        super().__init__()    
        # interpolation tables for finding overdensity dependent parameters    
        self.A  = CubicSpline([200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                              [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],)
        self.a  = CubicSpline([200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                              [1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66 ],)
        self.b  = CubicSpline([200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                              [2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41 ],)
        self.c  = CubicSpline([200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                              [1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44 ],)

    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None) -> Any:
        s     = np.asfarray(s)
        zp1   = np.add(z, 1)
        # eqn 8
        alpha = 10.0**( -( 0.75 / np.log10( overdensity / 75 ) )**1.2 )  
        A = self.A( overdensity ) / zp1**0.14  # eqn 5
        a = self.a( overdensity ) / zp1**0.06  # eqn 6     
        b = self.b( overdensity ) / zp1**alpha # eqn 7 
        c = self.c( overdensity )
        # eqn 3
        res = A * ( 1 + ( b / s )**a ) * np.exp( -c / s**2 ) 
        return res
    

# initialising models to be readily used
press74  = Press74()
sheth01  = Sheth01()
tinker08 = Tinker08() 

_available_models__ = {'press74' : press74,
                       'sheth01' : sheth01,
                       'tinker08': tinker08, }
