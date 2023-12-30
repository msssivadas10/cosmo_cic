#!/usr/bin/python3

import numpy as np
from typing import Any 
from scipy.special import hyp2f1 # hypergeometric function
from ._base import Cosmology, CosmologyError

class FlatLambdaCDM(Cosmology):
    r"""
    A class representing a flat Lambda-CDM cosmlogy.
    """
    
    def __init__(self, 
                 h: float, 
                 Om0: float, 
                 Ob0: float, 
                 Onu0: float = 0, 
                 Nnu: float = 3, 
                 ns: float = 1, 
                 sigma8: float = 1, 
                 Tcmb0: float = 2.725, 
                 name: str | None = None, ) -> None:
        super().__init__(h, Om0, Ob0, 
                         Ode0 = None, 
                         Onu0 = Onu0, 
                         Nnu = Nnu, 
                         ns = ns, 
                         sigma8 = sigma8, 
                         w0 = -1., 
                         wa = 0., 
                         Tcmb0 = Tcmb0, 
                         name = name, )
        return
    
    def darkEnergyModel(self, 
                        z: Any, 
                        deriv: int = 0, ) -> Any:
        return np.zeros_like(z, dtype = 'float') if deriv else np.ones_like(z, dtype = 'float')
    
    def comovingDistance(self, 
                         z: Any, 
                         deriv: int = 0, ) -> Any:
        FACT = Cosmology.UNIT_DISTANCE
        zp1 = np.add(z, 1.)
        res = self.Om0 * zp1**3
        # first derivative
        if deriv:
            res = FACT / np.sqrt(res + self.Ode0)
            return res
        # function (in terms of hypergeometric series)
        res = zp1 / np.sqrt(res) * hyp2f1(1./6., 0.5, 7./6., -self.Ode0/res)
        res = 1. / np.sqrt(self.Om0) * hyp2f1(1./6., 0.5, 7./6., -self.Ode0/self.Om0) - res
        res = 2*FACT * res
        return res
    
    def time(self, 
             z: Any, 
             deriv: int = 0, ) -> Any:
        zp1 = np.add(z, 1.)
        res = self.Om0 * zp1**3
        # first derivative
        if deriv:
            res = -1. / ( np.sqrt( res + self.Ode0 ) * zp1 )
            return res
        # function (in terms of hypergeometric series)
        res = 2./(3*np.sqrt(res)) * hyp2f1(0.5, 0.5, 1.5, -self.Ode0/res) 
        return res
    
    def dplus(self, 
              z: Any, 
              deriv: int = 0, ) -> Any:
        zp1 = np.add(z, 1.)
        res = self.Om0 * zp1**3
        # first log derivative
        if deriv:
            arg = self.Ode0/res
            res = -2.5 + 1.5/(1 + arg) + (45./22.) * hyp2f1(11./6., 2.5, 17./6., -arg) / hyp2f1(5./6., 1.5, 11./6., -arg) * arg
            return -res
        # function (in terms of hypergeometric series)
        res = 2./5./ self.Om0**1.5 * zp1**-2.5 * np.sqrt(res + self.Ode0) * hyp2f1(5./6., 1.5, 11./6., -self.Ode0/res)
        return res

#########################################################################################
#                               Built-in models + constructor                           #
#########################################################################################

def cosmology(name: str, *args, **kwargs) -> Cosmology:
    r"""
    Return a cosmology model.

    Parameters
    ----------
    name: str
        If a predefined name, return that cosmology.Otherwise, create a cosmology with 
        this name.
    *args, **kwargs: Any
        Other arguments are passed to `Cosmology` object constructor.

    Returns
    -------
    cm: Cosmology

    See Also
    --------
    Cosmology

    """
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be an 'str' or None")
    # cosmology with parameters from Plank et al (2018)
    if name == 'plank18':
        return Cosmology(h = 0.6790, Om0 = 0.3065, Ob0 = 0.0483, Ode0 = 0.6935, sigma8 = 0.8154, ns = 0.9681, Tcmb0 = 2.7255, name = 'plank18')
    # cosmology with parameters from Plank et al (2015)
    if name == 'plank15':
        return Cosmology(h = 0.6736, Om0 = 0.3153, Ob0 = 0.0493, Ode0 = 0.6947, sigma8 = 0.8111, ns = 0.9649, Tcmb0 = 2.7255, name = 'plank15')
    # cosmology with parameters from WMAP survay
    if name == 'wmap08':
        return Cosmology(h = 0.719, Om0 = 0.2581, Ob0 = 0.0441, Ode0 = 0.742, sigma8 = 0.796, ns = 0.963, Tcmb0 = 2.7255, name = 'wmap08')
    # cosmology for millanium simulation
    if name == 'millanium':
        return Cosmology(h = 0.73, Om0 = 0.25, Ob0 = 0.045, sigma8 = 0.9, ns = 1.0, Tcmb0 = 2.7255, name = 'millanium')
    if not args and not kwargs:
        raise CosmologyError(f"model not available: '{name}'")
    # create a new model with given name
    if kwargs.get( 'flat', False ): return FlatLambdaCDM(*args, **kwargs, name = name)
    return Cosmology(*args, **kwargs, name = name)