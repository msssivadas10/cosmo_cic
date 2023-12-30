#!/usr/bin/python3

# for basic cosmology calculations
from ._base import Cosmology, CosmologyError
# for calculations related to power spectrums
from ._base import PowerSpectrum, WindowFunction
# for halo statistics
from ._base import MassFunction, HaloBias
# for halo structure
from ._base import HaloDensityProfile, HaloConcentrationMassRelation
# specialised models and constructors
from .cosmology import cosmology, FlatLambdaCDM

# initialise models
def _initialise_models() -> None:
    from .power_spectrum.linear_models import _available_models__
    for __name, __model in _available_models__.items():
        PowerSpectrum.available.add( __name, __model )
    from .power_spectrum.window_functions import _available_models__
    for __name, __model in _available_models__.items():
        WindowFunction.available.add( __name, __model )
    from .halos.mass_function import _available_models__
    for __name, __model in _available_models__.items():
        MassFunction.available.add( __name, __model )
    from .halos.bias import _available_models__
    for __name, __model in _available_models__.items():
        HaloBias.available.add( __name, __model )
    from .halos.cm_relations import _available_models__
    for __name, __model in _available_models__.items():
        HaloConcentrationMassRelation.available.add( __name, __model )
    from .halos.density_profiles import _available_models__
    for __name, __model in _available_models__.items():
        HaloDensityProfile.available.add( __name, __model )
    return

# initialising the module
_initialise_models()

__all__ = ['power_spectrum', 
           'halos', 
           'specials',
           'cosmology', 
           'FlatLambdaCDM',
           'Cosmology', 
           'CosmologyError',
           'PowerSpectrum', 
           'WindowFunction', 
           'MassFunction', 
           'HaloBias', 
           'HaloDensityProfile', 
           'HaloConcentrationMassRelation', ]
