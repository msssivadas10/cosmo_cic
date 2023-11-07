#!/usr/bin/python3

r"""

Estimate Cosmological information Using `measure2`
==================================================

This module can be used for making cosmological measurements from possibly large data. These measurements 
include estimation of *count-in-cells*, a one-point statistic and the *two-point correlation function*. 
All measurement routines has support for parellel processing using MPI by `mpi4py` module.

Use these sub-modules for different uses:

- `counting` for count-in-cells (currently, only rectangular cells are used).
- `correlation` for pair counting and correlation function **will be added in future**.
- `stats` for (general) re-sampling and error estimation using e.g., jackknife resampling.

"""

__all__ = ['counting', 'stats', 'utils']

