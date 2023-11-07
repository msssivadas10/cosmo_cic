#!/usr/bin/python3

r"""

Power Spectrum and Related Calculations
=======================================

Define the base classes for power spectrum (`PowerSpectrum`) and smoothing window (`WindowFunction`) 
used in calculations related to structure formation. Also contains

- `builtin_windows` - a dict of built-in windows
- `builtin_linear_spectrums` - a dict of built-in linear power spectrum models
- `builtin_power_spectrums` - a dict of all available power spectrum models

"""

from ._base import PowerSpectrum, WindowFunction
from .windows import available_models as builtin_windows
from .linear_models import available_models as builtin_linear_spectrums

# union of all available power spectrum models: linear models will have the same 
# name and non-linear models will have subscript `nl` in their name.
builtin_power_spectrums = { **builtin_linear_spectrums }

__all__ = ['linear_models', 'windows']
