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

def test1():
    Muv  = -20.5
    zbar = 1.7
    hm = Zheng07.harikane22(Muv, zbar, cm, 200.)
    # print( hm )

    plt.figure()
    x = np.logspace(6, 18, 11)
    plt.loglog()
    y = cm.haloMassFunction(x, zbar, 200.)
    plt.plot(x, y, '-')
    y = hm.massFunction(x, zbar)
    plt.plot(x, y, 's')
    plt.show()
    return

def test2():
    from scipy.interpolate import CubicSpline
    hm = Zheng07.harikane22(mag = -22.5, 
                            z   =  3.8, 
                            cosmology   = cm, 
                            overdensity = 200, )
    # for z = 3.8 (~ 4)
    complteness_table = CubicSpline([3.00, 3.20, 3.40, 3.60, 4.00, 4.20, 4.40, 4.50],
                                    [0.00, 0.02, 0.55, 0.75, 0.60, 0.55, 0.05, 0.00])
    hm.setRedshiftDistribution(lambda z, cm: complteness_table( z ), 
                               z_min = 3.0,
                               z_max = 4.5, )
    print(f"log Mmin: { hm.logm_min :10.3f}")
    print(f"log Msat: { hm.logm1    :10.3f}")
    print(f"Eff. bg : { hm.effectiveBias() :10.3f}")
    print(f"log <Mh>: { np.log(hm.averageHaloMass()) :10.3f}")
    
    return

if __name__ =='__main__':
    test2()
