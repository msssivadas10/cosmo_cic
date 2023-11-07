#!/usr/bin/python3

r"""

Halo Density Profiles and Concentration-Mass Relations
======================================================

Define the base class for halo density profile (`DensityProfile`) and :math:`c-M` relation (`CMRelation`).
Also contain some pre-defined models.

Available profile models are (defined in the sub-module `profiles`)

- `nfw` - NFW profile by Navarro, Frenk and White (1997), pre-loaded with `bullock01` c-M relation.

Available c-M relations are (defined in sub-module `cmrelations`)

- `bullock01` - Bullock et. al. (2001)
- `zheng07` - Zheng et. al. (2017). Redshift ~ 0 and 1

and `builtinProfiles` and `builtinCMRelations` are dict of built-in profile and c-M relation models.

"""

from ._base import DensityProfile, CMRelation
from .cmrelations import builtinCMRelations
from .profiles import builtinProfiles

__all__ = ['cmrelations', 'profiles']

