#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy  as np 
import matplotlib.pyplot as plt 
from cic.models2.cosmology import Cosmology

cm = Cosmology(0.7, 0.30, 0.05, ns = 1, sigma8 = 0.8)
cm.link(power_spectrum = 'eisenstein98_zb', window = 'tophat')
# print(cm)

plt.figure()

# z, y1, y2, y3, y4 = np.loadtxt('tests/z_out.csv', skiprows = 1, delimiter = ',', unpack = 1)

# x1 = cm.comovingDistance(z) / cm.h
# plt.plot(z, x1, '-o', ms = 10)
# plt.plot(z, y1, 'o')

# x2 = cm.comovingVolumeElement(z) / cm.h**3
# plt.plot(z, x2, '-o', ms = 10)
# plt.plot(z, y2, 'o')

# x3 = cm.dplus(z)
# plt.plot(z, x3, '-o', ms = 10)
# plt.plot(z, y3, 'o')

# x4 = cm.dplus(z, deriv=1)
# plt.plot(z, x4, '-o', ms = 10)
# plt.plot(z, y4, 'o')

k, y1, y2, y3, y4, y5, y6 = np.loadtxt('tests/power.csv', skiprows = 1, delimiter = ',', unpack = 1)

# x1 = cm.matterTransferFunction(k/cm.h, 0.)
# plt.loglog(k, x1, '-o', ms = 10)
# plt.loglog(k, y1, 'o')

# x2 = cm.matterPowerSpectrum(k/cm.h, 0., normalize=1) / cm.h**3
# plt.loglog(k, x2, '-o', ms = 10)
# plt.loglog(k, y2, 'o')

# x3 = cm.matterPowerSpectrum(k/cm.h, 0., 1) 
# plt.semilogx(k, x3, '-o', ms = 10)
# plt.semilogx(k, y3, 'o')

# x4 = cm.matterVariance(k*cm.h, 0., normalize=1)
# plt.loglog(k, x4, '-o', ms = 10)
# plt.loglog(k, y4, 'o')

# x5 = cm.matterVariance(k*cm.h, 0., 1)
# plt.semilogx(k, x5, '-o', ms=10)
# plt.semilogx(k, y5, 'o')

# x6 = cm.matterVariance(k*cm.h, 0., 2)
# plt.semilogx(k, x6, '-o', ms=10)
# plt.semilogx(k, y6, 'o')

plt.show()


# x = [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ]
# y = [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260]
# Delta = 300.
# ts = len(y)
# A1 = 0.
# # if Delta < 200. :
# #     A1 = y[0]
# # elif Delta > 3200.:
# #     A1 = y[ts-1]
# # else:
# i = 0
# while i < ts-1:
#     i = i + 1
#     if not (Delta > x[i]):  
#         break
# t  = ( Delta - x[i-1] ) / ( x[i] - x[i-1] )
# print(i, x[i-1], Delta, x[i], t)
# A1 = y[i]* t + y[i-1] * (1 - t) 
# print(A1, i)