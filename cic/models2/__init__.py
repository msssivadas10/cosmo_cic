#!/usr/bin/python3

# for basic cosmology calculations
from ._base import Cosmology, CosmologyError
# for calculations related to power spectrums
from ._base import PowerSpectrum, WindowFunction
# for halo statistics
from ._base import MassFunction, HaloBias
# for halo structure
from ._base import HaloDensityProfile, HaloConcentrationMassRelation

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
        raise KeyError(f"model not available: '{name}'")
    # create a new model with given name
    return Cosmology(*args, **kwargs, name = name)

def _init_module() -> None:
    # initialise power spectrum module
    from .power_spectrum import _init_module
    _init_module()
    # initialise halos module
    from .halos import _init_module
    _init_module()
    return

# initialising the module
_init_module()

__all__ = ['power_spectrum', 
           'halos', 
           'cosmology', 
           'Cosmology', 
           'CosmologyError',
           'PowerSpectrum', 
           'WindowFunction', 
           'MassFunction', 
           'HaloBias', 
           'HaloDensityProfile', 
           'HaloConcentrationMassRelation', ]
