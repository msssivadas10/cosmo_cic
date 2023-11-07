#!/usr/bin/python3

r"""

A `cosmology` module, for all Cosmological Calculations
=======================================================

Main thing in the module is the class `Cosmology`, which is the represesntation of a practical cosmology 
model and is used for all related calculations. To use, import the class from the module

>>> from models.cosmology import Cosmology

and create an instance

>>> cm = Cosmology(h = 0.7, Om0 = 0.3, Ob0 = 0.05, ns = 1., sigma8 = 0.8, name = 'test_cosmology')
>>> cm
Cosmology(name=test_cosmology, h=0.7, Om0=0.3, Ob0=0.05, Ode0=0.7, ns=1.0, sigma8=0.8, Tcmb0=2.725)

or, use any of the built-in models from `plank18`, `plank15`, `wmap08` or `millanium`

>>> from models.cosmology import plank18 as cm

Then, configure it by setting models and normalizing power spectrum (without normalizing the power spectrum, 
mass-function, bias etc calculations does not work properly).

>>> cm.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
>>> cm.createInterpolationTables() 
>>> cm.normalizePowerSpectrum() 

And, use for any calculations!

"""

from ._base import Cosmology, CosmologyError
from .models import available_models as builtin_cosmologies
from .models import plank18, plank15, wmap08, millanium


