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


cm = cosmology('plank18')
cm = cosmology('test_cosmology', 0.7, 0.3, 0.05, sigma8 = 0.8)
cm.link(power_spectrum = 'eisenstein98_zb', 
        window         = 'tophat', 
        mass_function  = 'press74', 
        halo_bias      = 'cole89', 
        cmreln         = 'zheng07',
        halo_profile   = 'nfw',   )
# print(cm)

# hm = Model1()
# hm.link(cm)
# from scipy.interpolate import CubicSpline
plt.figure()
# x = np.linspace(-1, 1, 21)
# y*x**2 + x - y = 0 --> x = -1 +- sqrt(1 + 4*y**2) / 2*y
# y = 5*x / (1 - x**2)
# plt.plot(y, np.ones_like(y), '-s')
r = np.logspace(-6, 6, 11)
# m = np.logspace(6, 14, 5)
# z = np.linspace(0, 3, 3)
y = cm.matterVariance(r, 0, 0)
# f = CubicSpline(np.log(r), np.log(y))
# y = cm.matterVariance(r, 0, 2)
# y = cm.matterPowerSpectrum(r, 0, 1) 
plt.loglog()
# plt.semilogx()
plt.plot(r, (y), '-s')
# plt.plot(r, (f(np.log(r), 1)), '-')
# y = hm.galaxyPowerSpectrum(r, 0)
# print(np.shape(y), np.shape(r[:,None,None]))
# plt.loglog(r, y.T, '-')
plt.show()

# from cic.models2._base import IntegrationPlan
# p = IntegrationPlan().create([[-1, 0, 5], [0., 1., 11],[1., 1.5, 3]])
# print(p.generatePoints())