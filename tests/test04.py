#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy  as np 
import matplotlib.pyplot as plt 
from cic.models2.cosmology import Cosmology

cm = Cosmology(0.7, 0.30, 0.05, ns = 1, sigma8 = 0.8)
cm.link(power_spectrum = 'eisenstein98_zb', window = 'tophat', mass_function='tinker08', halo_bias='tinker10')
# print(cm)

plt.figure()

# z, y1, y2, y3, y4 = np.loadtxt('tests/z_out.csv', skiprows = 1, delimiter = ',', unpack = 1)

# x1 = cm.comovingDistance(z) / cm.h * 1e-3
# plt.plot(z, x1, '-o', ms = 10)
# plt.plot(z, y1, 'o')

# x2 = cm.comovingVolumeElement(z) / cm.h**3 * 1e-9
# plt.plot(z, x2, '-o', ms = 10)
# plt.plot(z, y2, 'o')

# x3 = cm.dplus(z)
# plt.plot(z, x3, '-o', ms = 10)
# plt.plot(z, y3, 'o')

# x4 = cm.dplus(z, deriv=1)
# plt.plot(z, x4, '-o', ms = 10)
# plt.plot(z, y4, 'o')

# z = 2
# k, y1, y2, y3 = np.loadtxt('tests/power_z%.3f.csv' % z, skiprows = 1, delimiter = ',', unpack = 1)

# x1 = cm.matterTransferFunction(k/cm.h, z)
# plt.loglog(k, x1, '-o', ms = 10)
# plt.loglog(k, y1, 'o')

# x2 = cm.matterPowerSpectrum(k/cm.h, z, normalize=1) / cm.h**3
# plt.loglog(k, x2, '-o', ms = 10)
# plt.loglog(k, y2, 'o')

# x3 = cm.matterPowerSpectrum(k/cm.h, z, 1) 
# plt.semilogx(k, x3, '-o', ms = 10)
# plt.semilogx(k, y3, 'o')

z = 0
m, y0, y1, y2, y4, y3, y5 = np.loadtxt('tests/massfunc_z%.3f.csv' % z, skiprows = 1, delimiter = ',', unpack = 1)

# x1 = cm.matterVariance(cm.lagrangianR(m*cm.h), z, normalize=1)**0.5
# plt.loglog(m, x1, '-o', ms = 10)
# plt.loglog(m, y1, 'o')

# x2 = cm.matterVariance(cm.lagrangianR(m*cm.h), z, 1) / 6.
# plt.semilogx(m, x2, '-o', ms=10)
# plt.semilogx(m, y2, 'o')

# x3 = cm.haloMassFunction(m*cm.h, z, 200., retval='dndlnm') * cm.h**3
# plt.loglog(m, x3, '-o', ms=10)
# plt.loglog(m, y3, 'o')

# x4 = cm.haloMassFunction(m*cm.h, z, 200., retval='f') 
# plt.loglog(m, x4, '-o', ms=10)
# plt.loglog(m, y4, 'o')

x5 = cm.haloBias(m*cm.h, z, 200.) 
plt.loglog(m, x5, '-o', ms=10)
plt.loglog(m, y5, 'o')

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