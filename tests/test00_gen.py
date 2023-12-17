#!/usr/bin/python3

# add module location to path
import sys, os.path as path
from typing import Any
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
from scipy.special import erf 
import matplotlib.pyplot as plt
from cic.models2 import cosmology
from cic.models2.hod import HaloModel

class Model1(HaloModel):
	def __init__(self, 
				 mmin: float = 1e+08, 
				 msat: float = 1e+12,
				 mscale: float = 0.2, 
				 alpha: float = 1., ) -> None:
		super().__init__()
		self.mmin = mmin
		self.msat = msat
		self.mscale = mscale
		self.alpha  = alpha
		self.mcut   = 1 / np.sqrt(mmin)
		return

	def centralCount(self, m: Any) -> Any:
		x = ( np.log(m) - np.log(self.mmin) ) / ( np.sqrt(2) * self.mscale )
		return 0.5 * ( 1. + erf(x) )
	
	def satelliteFraction(self, m: Any) -> float:
		return ( ( np.subtract(m, self.mcut) ) / self.msat )**self.alpha    


# cm = cosmology('plank18')
cm = cosmology('test_cosmology', 0.7, 0.3, 0.05, sigma8 = 0.8)
cm.link(power_spectrum = 'eisenstein98_zb', 
        window         = 'tophat', 
        mass_function  = 'press74', 
        halo_bias      = 'cole89', 
        cmreln         = 'zheng07',
        halo_profile   = 'nfw',   )
# print(cm)

hm = Model1()
hm.link(cm)
plt.figure()
r = np.logspace(-3, 3, 51)
m = np.logspace(6, 14, 5)
z = np.linspace(0, 3, 3)
y = hm.galaxyPowerSpectrum(r, 0)
# print(np.shape(y), np.shape(r[:,None,None]))
plt.loglog(r, y.T, '-')
plt.show()