#!/usr/bin/python3

r"""

Power Spectrum and Related Calculations
=======================================

Define the base classes for power spectrum (`PowerSpectrum`) and smoothing window (`WindowFunction`) 
used in calculations related to structure formation. Also contains

- `builtinWindows` - a dict of built-in windows
- `builtinLinearSpectrums` - a dict of built-in linear power spectrum models
- `builtinPowerSpectrums` - a dict of all available power spectrum models

"""

from ._base import PowerSpectrum, WindowFunction
from .windows import builtinWindows
from .linear_models import builtinPowerSpectrums as builtinLinearSpectrums

# union of all available power spectrum models: linear models will have the same 
# name and non-linear models will have subscript `nl` in their name.
builtinPowerSpectrums = { **builtinLinearSpectrums }

__all__ = ['linear_models', 'windows']
