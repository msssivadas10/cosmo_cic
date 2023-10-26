#!/usr/bin/python3

__all__ = ['constants', 
           'cosmology',
           'halo_bias', 
           'halo_model',
           'mass_function',
           'power_spectrum', 
           ]

from cic.models.cosmology import Cosmology, predefined as predefinedCosmology
from cic.models.power_spectrum import available as availablePowerSpectrumModels
from cic.models.mass_function import available as availableMassfunctionModels
from cic.models.halo_bias import available as availableBiasModels


#
# Constant names for various models
#

# predefined cosmologies
PLANK18_COSMOLOGY              = 'plank18'
PLANK15_COSMOLOGY              = 'plank15'
WMAP08_COSMOLOGY               = 'wmap08'
MILLANIUM_COSMOLOGY            = 'millanium'

# window functions
TOPHAT_WINDOW                  = 'tophat'
GAUSSIAN_WINDOW                = 'gauss'

# power spectrum models
EISENSTEIN98_ZB_POWER_SPECTRUM = 'eisenstein98_zb'
EISENSTEIN98_NU_POWER_SPECTRUM = 'eisenstein98_nu'

# halo mass-functions
PRESS74_MASS_FUNCTION          = 'press74'
SHETH01_MASS_FUNCTION          = 'sheth01'
TINKER08_MASS_FUNCTION         = 'tinker08'

# linear halo bias functions
COLE89_LINEAR_BIAS             = 'cole89'
TINKER10_LINEAR_BIAS           = 'tinker10'
