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
    
class Jenkins01(MassFunction):
    r"""
    Halo mass function by Jenkins et al (2001). It is valid over the range `-1.2 <= -log(sigma) <= 1.05`[1]_.

    References
    ----------
    .. [1] A. Jenkins et al. The mass function of dark matter halos. <http://arxiv.org/abs/astro-ph/0005260v2>
    
    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None) -> Any:
        # parameters
        s = np.asarray(s)
        res = 0.315*( -np.abs(0.61 - np.log(s))**3.8 )
        return res

class Reed03(Sheth01):
    r"""
    Halo mass function by Reed et al (2003)[1]_.

    References
    ----------
    .. [1] Zarija Lukić et al. The halo mass function: high-redshift evolution and universality. 
            <http://arXiv.org/abs/astro-ph/0702360v2>.
    
    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None) -> Any:
        s   = np.asfarray(s)
        res = super().call(model, s, z, overdensity) * np.exp(-0.7 / (s*np.cosh(2*s)**5.))
        return res
    
class Warren06(MassFunction):
    r"""
    Halo mass function by Warren et al (2006)[1]_.

    References
    ----------
    .. [1] Zarija Lukić et al. The halo mass function: high-redshift evolution and universality. 
            <http://arXiv.org/abs/astro-ph/0702360v2>.
    
    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float = 200., ) -> Any:
        # parameters
        A, a, b, c = 0.7234, 1.625, 0.2538, 1.1982
        s   = np.asfarray(s)
        res = A*( s**-a + b ) * np.exp( -c / s**2 )
        return res
    
class Reed07(MassFunction):
    r"""
    Halo mass function by Reed et al (2007)[1]_. It is aredshift and cosmology dependent model.

    References
    ----------
    .. [1] Reed et al. The halo mass function from the dark ages through the present day. Mon. Not. R. Astron. Soc. 374, 
            2-15 (2007)

    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        # parameters
        A, c, ca, p = 0.310, 1.08, 0.764, 0.3
        s  = np.asfarray(s)
        w  = np.sqrt( ca ) * model.collapseDensity( z ) / s
        G1 = np.exp( -0.5*( np.log(w) - 0.788 )**2 / 0.6**2 )
        G2 = np.exp( -0.5*( np.log(w) - 1.138 )**2 / 0.2**2 )
        # scale corresponding to this value of s
        r  = model.scale(s, r, normalize = True)
        # effective index
        neff = -2.*model.matterVariance(r, z, deriv = 1, normalize = True) - 3.
        res  = A * w * np.sqrt( 2.0/np.pi ) 
        res *= np.exp( -0.5*w - 0.0325*w**p / ( neff + 3 )**2 )
        res *= ( 1.0 + 1.02*w**( 2*p ) + 0.6*G1 + 0.4*G2 )
        return res

class Tinker08(MassFunction):
    r"""
    Halo mass function model by Tinker et al (2008)[1]_. This model is redshift dependent.

    References
    ----------
    .. [1] Jeremy Tinker et al. Toward a halo mass function for precision cosmology: The limits of universality. 
            <http://arXiv.org/abs/0803.2706v1> (2008).

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
             overdensity: float = 200., ) -> Any:
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
    
class Crocce10(MassFunction):
    r"""
    Halo mass function by Crocce et al (2010)[1]_.

    References
    ----------
    .. [1] Martín Crocce et al. Simulating the Universe with MICE : The abundance of massive clusters. 
            <http://arxiv.org/abs/0907.0019v2>
    
    """
    def call(self, 
             model: Cosmology, 
             s: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        s   = np.asfarray(s)
        zp1 = np.add(z, 1.)
        # redshift dependent parameters
        Az  = 0.580 * zp1**-0.130
        az  = 1.370 * zp1**-0.150
        bz  = 0.300 * zp1**-0.084
        cz  = 1.036 * zp1**-0.024
        res = Az * ( s**-az + bz ) * np.exp( -cz / s**2 )
        return res
    
class Courtin10(Sheth01):
    r"""
    Halo mass function by Courtin et al (2010)[1]_.

    References
    ----------
    .. [1] J. Courtin et al. Imprints of dark energy on cosmic structure formation-II. Non-universality of the halo 
            mass function. Mon. Not. R. Astron. Soc. 410, 1911-1931 (2011)
    
    """
    def __init__(self) -> None:
        super().__init__()
        # parameters
        self.A, self.a, self.p = 0.348, 0.695, 0.1 
    

# initialising models to be readily used
press74   = Press74()
sheth01   = Sheth01()
jenkins01 = Jenkins01()
reed03    = Reed03()
warren06  = Warren06()
reed07    = Reed07()
tinker08  = Tinker08()
crocce10  = Crocce10()
courtin10 = Courtin10() 

_available_models__ = {'press74'  : press74, 
                       'sheth01'  : sheth01, 
                       'jenkins01': jenkins01, 
                       'reed03'   : reed03, 
                       'warren06' : warren06, 
                       'reed07'   : reed07, 
                       'tinker08' : tinker08, 
                       'crocce10' : crocce10, 
                       'courtin10': courtin10, }
