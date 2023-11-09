#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from cic.models.cosmology import plank18
from cic.models.halo_model import HaloModel
from test03_cosmo import PlotData, plot

n = 5

# create interpolation tables
plank18.createInterpolationTables()

# normalisation
plank18.normalizePowerSpectrum()

# test halo model
hm = HaloModel(plank18, m_min = 1e+08, sigma = 0.2, m_sat = 1e+12, alpha = 1.)


# find the n-sigma mass at redshift z 
@np.vectorize
def nsigmaMass(z, n, cm):
    lnr = brentq(lambda lnr, z: cm.peakHeight(np.exp(lnr), z) - n, np.log(1e-03), np.log(1e+03), (z, ))
    return cm.lagrangianM(np.exp(lnr))

z = np.asfarray([0., 2., 5.])
m = np.logspace(6., 20., 31)
if np.ndim( z ) >= 1: m = m[:, None]


res1 = hm.totalCount( m ) * hm.massFunction(m, z, 'dndlnm', False)

# integrand for avg. halo mass
res2 = res1 * m

# integrand for effective galaxy bias
res3 = res1 * hm.biasFunction( m, z, False ) 

mm = m #/ nsigmaMass(z, n, plank18)

fig, axs = plt.subplots(3, 1, sharex = 1)
plt.subplots_adjust(hspace=0)
axs[0].loglog(mm, res1, 's-', ms = 4)
axs[0].set_ylim((1e-16, 1e+06))
axs[0].set_ylabel('galaxy density')

axs[1].loglog(mm, res2, 's-', ms = 4)
axs[1].set_ylim((1e-16, 1e+16))
axs[1].set_ylabel('avg. halo mass')

axs[2].loglog(mm, res3, 's-', ms = 4)
axs[2].set_ylim((1e-16, 1e+06))
axs[2].set_ylabel('effective bias')
# axs[2].set_xlabel('$m/m_{%g\\sigma}$' % n)
axs[2].set_xlabel('$m$')
axs[2].legend(z, title = 'redshift, z:')

plt.show()



