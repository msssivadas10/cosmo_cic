#!/usr/bin/python3

r"""

Halo Bias models
================

Define the base class for halo bias function (`HaloBias`) and some built-in models.

Available linear models are (defined in the sub-module `linear_models`)

- `cole89` - Cole & Kaiser (1989)
- `tinker10` - Tinker et al (2008)

and `builtinLinearBiases` is a dict of built-in models

"""

from ._base import HaloBias
from .linear_models import builtinLinearBiases

