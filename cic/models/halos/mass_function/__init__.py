#!/usr/bin/python3

r"""

Halo Mass-function models
=========================

Define the base class for halo mass-function (`MassFunction`) and some built-in models.

Available models are 

- `press74` - Press & Schechter (1974), based on spherical collapse.
- `sheth01` - Sheth et al (2001), based on ellipsoidal collapse
- `tinker08` - a redshift dependent model by Tinker et al (2008)

and `builtin_massfunctions` is a dict of built-in models

"""

from ._base import MassFunction
from .models import available_models as builtin_massfunctions
from .models import press74, sheth01, tinker08