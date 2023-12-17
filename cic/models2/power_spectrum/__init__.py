#!/usr/bin/python3

__all__ = ['linear_models', 'window_functions']

def _init_module() -> None:
    # initialise the linear power spectrum module
    from .linear_models import _init_module
    _init_module()
    # initialise the window functions module
    from .window_functions import _init_module
    _init_module()
    return


