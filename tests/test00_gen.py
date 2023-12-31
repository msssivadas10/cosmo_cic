#!/usr/bin/python3

# add module location to path
import sys, os.path as path
from typing import Any
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
from scipy.special import erf 
import matplotlib.pyplot as plt
from cic.models2 import cosmology
from cic.models2.stats.hod import HaloModel

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
# plt.figure()
# x = np.linspace(-1, 1, 21)
# y*x**2 + x - y = 0 --> x = -1 +- sqrt(1 + 4*y**2) / 2*y
# y = 5*x / (1 - x**2)
# plt.plot(y, np.ones_like(y), '-s')
r = np.logspace(-3, 3, 11)
# m = np.logspace(6, 14, 5)
# z = np.linspace(0, 10, 11)
# y = cm.dplus(z, 1)
y  = cm.matterCorrelation(r, 0, 1)
y1 = cm.matterCorrelation(r, 0, 1, ht = 1)
# f = CubicSpline(np.log(r), np.log(y))
# y = cm.matterVariance(r, 0, 2)
# y = cm.matterPowerSpectrum(r, 0, 1) 
plt.loglog()
# plt.semilogx()
plt.plot(r, y, '-s')
plt.plot(r, y1, '-o')
# plt.plot(r, (f(np.log(r), 1)), '-')
# y = hm.galaxyPowerSpectrum(r, 0)
# print(np.shape(y), np.shape(r[:,None,None]))
# plt.loglog(r, y.T, '-')
plt.show()

# from cic.models2._base import IntegrationPlan
# p = IntegrationPlan().create([[-1, 0, 5], [0., 1., 11],[1., 1.5, 3]])
# print(p.generatePoints())


# from scipy.fft import irfft
# from scipy.special import gamma
# k = 1
# L = -2*np.log(1e-8)
# print(L)
# N = 32
# m = np.arange(N//2 + 1)
# print(m)
# w = np.pi*m/L
# w = np.exp(1j*2*np.log(2)*w) * gamma(1. + 1j*w) / gamma((2*k+1)*0.5 - 1j*w)
# print(w.shape)
# w = irfft(w) * 2 / 2**(k + 0.5) * (2*k+1) / (2*np.pi)**1.5
# print(w)
# m = np.arange(N)
# m[m > N//2] -= N
# x = np.exp(m*L/N)
# print(x)
# print( np.sum(w) )
# print( 2**(0.5-k) * (2*k+1) / (2*np.pi)**1.5 / gamma(k+0.5) )
# plt.figure()
# plt.semilogx()
# plt.plot(x, w, 's')
# plt.show()