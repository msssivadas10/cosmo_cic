#!/usr/bin/python3

import numpy as np
from typing import Any 
from .._base import HaloBias, Cosmology, DELTA_SC

class Cole89(HaloBias):
    r"""
    Linear bias model given by Cole & Kaiser (1989)[1]_ and Mo & White (1996)[2]_.

    References
    ----------
    .. [1] Shaun Cole and Nick Kaiser. Biased clustering in the cold dark matter cosmogony. Mon. Not.R. astr. Soc. 
            237, 1127-1146 (1989).
    .. [2] H. J. Mo, Y. P. Jing and S. D. M. White. High-order correlations of peaks and haloes: a step towards
            understanding galaxy biasing. Mon. Not. R. Astron. Soc. 284, 189-201 (1997).

    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        nu  = np.asfarray( nu )
        res = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        return res
    
class Sheth01(HaloBias):
    r"""
    Linear bias model given by Sheth et al. (2001). For the functional form, see, for example [1]_. 

    References
    ----------
    .. [1] Jeremy L. Tinker et al. The large scale bias of dark matter halos: Numerical calibration and model tests. 
            <http://arxiv.org/abs/1001.3162v2> (2010).
    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        # parameters
        a, b, c = 0.707, 0.5, 0.6
        sqrt_a  = np.sqrt(a)
        nu = np.asfarray( nu )
        anu2 = a*nu**2
        res  = sqrt_a * anu2 + sqrt_a * b * anu2**( 1-c ) - anu2**c / ( anu2**c + b * (1. - c) * (1. - 0.5*c) )
        res  = 1. + 1. / sqrt_a / DELTA_SC * res 
        return res
    
class Jing98(HaloBias):
    r"""
    Linear bias model by Jing (1998)[1]_.

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Y. P. Jing. Accurate fitting formula for the two-point correlation function of the dark matter halos. 
            <http://arXiv.org/abs/astro-ph/9805202v2> (1998).

    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        nu = np.asfarray( nu )
        r  = model.scale(DELTA_SC / nu, z, normalize = True)
        neff = model.matterPowerSpectrum(2*np.pi/r, z, deriv = True)
        res  = (0.5 / nu**4 + 1)**(0.06 - 0.02*neff) * (1. + (nu**2 - 1) / DELTA_SC) 
        return res
    
class Seljak04(HaloBias):
    r"""
    Linear bias model by Seljak et al (2004)[1]_.

    References
    ----------
    .. [1] Uroš Seljak & Michael S. Warren. Large scale bias and stochasticity of halos and dark matter. 
            <http://arxiv.org/abs/astro-ph/0403698v3> (2004).

    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, 
             enable_correction: bool = False, ) -> Any:
        # redshift dependent collapse density
        delta_c = model.collapseDensity(z)
        nu = np.asfarray(nu)
        x  = ( model.scale(delta_c / nu, z, normalize = True) / model.scale(1., z, normalize = True) )**3 
        res = 0.53 + 0.39*x**0.45 + 0.13 / ( 40.0*x + 1 ) + 5E-4*x**1.5
        if not enable_correction:
            return res
        h, Om0, ns, sigma8 = model.h, model.Om0, model.ns, model.sigma8
        alpha_s = 0.
        res    += np.log10(x) * ( 0.4*( Om0 - 0.3 + ns - 1.0 ) + 0.3*( sigma8 - 0.9 + h - 0.7 ) + 0.8*alpha_s )
        return res

class Tinker10(HaloBias):
    r"""
    Linear bias model given by Tinker et al. (2010)[1]_.

    References
    ----------
    .. [1] Jeremy L. Tinker et al. The large scale bias of dark matter halos: Numerical calibration and model tests. 
            <http://arxiv.org/abs/1001.3162v2> (2010).

    """
    def call(self, 
             model: Cosmology, 
             nu: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        nu = np.asfarray( nu )
        z  = np.asfarray( z )
        y  = np.log10( overdensity )
        A  = 1.0 + 0.24 * y * np.exp( -( 4. / y )**4 )
        a  = 0.44 * y - 0.88
        B  = 0.183
        b  = 1.5
        C  = 0.019 + 0.107 * y + 0.19 * np.exp( -( 4. / y )**4 )
        c  = 2.4
        res = 1.0 - A * nu**a / ( nu**a + DELTA_SC**a ) + B * nu**b + C * nu**c
        return res


# initialising models to be readily used
cole89   = Cole89()
tinker10 = Tinker10()
sheth01  = Sheth01()
jing98   = Jing98()
seljak04 = Seljak04()

_available_models__ = {'cole89'  : cole89,
                       'sheth01' : sheth01,
                       'jing98'  : jing98,
                       'seljak04': seljak04, 
                       'tinker10': tinker10, }

