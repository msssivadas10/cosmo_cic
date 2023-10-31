#!/usr/bin/python3

from ._base import PowerSpectrum, WindowFunction
from .windows import builtinWindows
from .linear_models import builtinPowerSpectrums as builtinLinearSpectrums

# union of all available power spectrum models: linear models will have the same 
# name and non-linear models will have subscript `nl` in their name.
builtinPowerSpectrums = { **builtinLinearSpectrums }

__all__ = ['linear_models', 'windows']
