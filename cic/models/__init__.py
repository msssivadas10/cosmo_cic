#!/usr/bin/python3

r"""

`models` - For Theoretical Cosmology Calculations
=================================================

This module defines various classes and functions for cosmology calculations. Main thing in the 
module is the class `Cosmology` - which, when initialised properly, can be used for almost all 
calculations related cosmology and astrophysics (using the module). 

Sub-modules are loaded with some buit-in models of quantities:

- `power_spectrum.linear_models` - linear matter power spectrums
- `halos.mass_function.models` - halo mass-function models
- `halos.bias.linear_models` - linear halo bias functions
- `halos.density.profiles`- halo density profiles

"""

from .cosmology import Cosmology
from .power_spectrum import PowerSpectrum, WindowFunction
from .halos.mass_function import MassFunction
from .halos.bias import HaloBias
from .halos.density import DensityProfile, CMRelation

__all__ = ['constants', 'utils', 'cosmology', 'power_spectrum', 'halos',]
