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

def test2():
    MUV_TH = -22.5 # uv magnitude threshold
    ZBAR   =  3.8  # redshift
    def z_distribution(z, cm = None): # for z = 3.8 (~ 4)
        res = np.interp(z, 
                        [3.00, 3.20, 3.40, 3.60, 4.00, 4.20, 4.40, 4.50],
                        [0.00, 0.02, 0.55, 0.75, 0.60, 0.55, 0.05, 0.00],
                        left  = 0.,
                        right = 0., )
        return res
    hm = Zheng07.harikane22(mag = MUV_TH, z = ZBAR, cosmology = cm, overdensity = 200)
    hm.setRedshiftDistribution(z_distribution, z_min = 3.0, z_max = 4.5)
    print(f"log Mmin    : { hm.logm_min :10.3f}")
    print(f"log Msat    : { hm.logm1    :10.3f}")
    print(f"avg. Density: { hm.averageDensity :10.3f}")
    print(f"Eff. bg     : { hm.effectiveBias() :10.3f}")
    print(f"log <Mh>    : { np.log10(hm.averageHaloMass()) :10.3f}")
    return

if __name__ =='__main__':
    test2()
