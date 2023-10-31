#!/usr/bin/python3

from ._base import Cosmology

builtinCosmology = {}

# cosmology with parameters from Plank et al (2018)
plank18 =  Cosmology(h = 0.6790, Om0 = 0.3065, Ob0 = 0.0483, Ode0 = 0.6935, sigma8 = 0.8154, ns = 0.9681, Tcmb0 = 2.7255, name = 'plank18')
plank18.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['plank18'] = plank18

# cosmology with parameters from Plank et al (2015)
plank15 =  Cosmology(h = 0.6736, Om0 = 0.3153, Ob0 = 0.0493, Ode0 = 0.6947, sigma8 = 0.8111, ns = 0.9649, Tcmb0 = 2.7255, name = 'plank15')
plank15.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['plank15'] = plank15

# cosmology with parameters from WMAP survay
wmap08 =  Cosmology(h = 0.719, Om0 = 0.2581, Ob0 = 0.0441, Ode0 = 0.742, sigma8 = 0.796, ns = 0.963, Tcmb0 = 2.7255, name = 'wmap08')
wmap08.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['wmap08'] = wmap08

# cosmology for millanium simulation
millanium = Cosmology(h = 0.73, Om0 = 0.25, Ob0 = 0.045, sigma8 = 0.9, ns = 1.0, Tcmb0 = 2.7255, name = 'millanium')
millanium.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
builtinCosmology['millanium'] = millanium
