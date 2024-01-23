#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from cic.models2 import cosmology
from cic.models2.stats.hod import Zheng07


cm = cosmology('plank18')
cm.link(power_spectrum = 'eisenstein98_zb', 
        window         = 'tophat', 
        mass_function  = 'tinker08', 
        halo_bias      = 'tinker10', 
        cmreln         = 'bullock01_powerlaw',
        halo_profile   = 'nfw',   )
# print( cm )

Muv  = -20.5
zbar = 1.7
hm = Zheng07.harikane22(Muv, zbar, cm, 200.)
print( hm )

print( ( hm.effectiveBias( zbar ) ) )
x = 12.72
# print( 3.16 * x - 24.33 )

# plt.figure()
# x = np.logspace(6, 18, 11)
# plt.loglog()
# y = cm.haloMassFunction(x, zbar, 200.)
# plt.plot(x, y, '-')
# y = hm.massFunction(x, zbar)
# plt.plot(x, y, 's')
# plt.show()
