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

Muv  = -20.5
zbar = 1.7
hm = Zheng07.predefined('harikane22', Muv, zbar, cm, 200.)

print( ( hm.effectiveBias( zbar ) ) )

# plt.figure()
# r = np.logspace(-3, 3, 11)
# y  = cm.matterCorrelation(r, 0, 1)
# y1 = cm.matterCorrelation(r, 0, 1, ht = 1)
# plt.loglog()
# plt.plot(r, y, '-s')
# plt.plot(r, y1, '-o')
# plt.show()
