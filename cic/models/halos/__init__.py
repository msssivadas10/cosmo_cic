#!/usr/bin/python3

r"""

Calculations Related to `halos`
===============================

Various utilities for halo related calculations, such as mass-function and bias. This defines 
the base classes: `MassFunction`, `HaloBias`, `DensityProfile` and `CMRelation`. 

and tables of available models:

- `builtin_massfunctions` - a dict of built-in mass-function models
- `builtin_linear_biases` - a dict of built-in linear bias models
- `builtin_profiles` - a dict of buil-in halo density profiles
- `builtin_cmrelations` - a dict of buil-in halo c-M relations

"""

from .mass_function import MassFunction, builtin_massfunctions
from .bias import HaloBias, builtin_linear_biases
from .density import DensityProfile, CMRelation, builtin_profiles, builtin_cmrelations

__all__ = ['bias', 'mass_function', 'density']
