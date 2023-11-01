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


# find the n-sigma mass at redshift z 
def nsigmaMass(n, z, cm):
    lnr = brentq(lambda lnr, z: cm.peakHeight(np.exp(lnr), z) - n, np.log(1e-03), np.log(1e+03), (z, ))
    return plank18.lagrangianM(np.exp(lnr))

# integrand for galaxy density
@plot
def galaxyDensityIntegrand(hm, z = [0., 2., 5.], start = 6., stop = 20., pts = 31):
    z = np.asfarray(z)
    m = np.logspace(start, stop, pts)
    if np.ndim( z ) >= 1: m = m[:, None]
    res = PlotData(np.divide( m, [nsigmaMass(5., zi, hm.cosmo) for zi in z] ),
                   hm.totalCount( m ) * hm.massFunction(m, z, 'dndlnm', False),
                   xlabel = '$m/m_{5\\sigma}$', ylabel = '$f(m)N(m)\\frac{dn(m)}{dln(m)}$', 
                   title = 'galaxy density integrand, $f(m) = 1$',
                   logx = True, logy = True, legend = z.tolist(), legend_title = 'redshift, z',
                   cm = hm.cosmo, ylim = (1e-08, 1e+06) )
    return res

# integrand for avg. halo mass
@plot
def averageMassIntegrand(hm, z = [0., 2., 5.], start = 6., stop = 20., pts = 31):
    z = np.asfarray(z)
    m = np.logspace(start, stop, pts)
    if np.ndim( z ) >= 1: m = m[:, None]
    res = PlotData(np.divide( m, [nsigmaMass(5., zi, hm.cosmo) for zi in z] ),
                   hm.totalCount( m ) * hm.massFunction(m, z, 'dndlnm', False) * m,
                   xlabel = '$m/m_{5\\sigma}$', ylabel = '$f(m)N(m)\\frac{dn(m)}{dln(m)}$', 
                   title = 'avg. halo mass integrand, $f(m) = m$',
                   logx = True, logy = True, legend = z.tolist(), legend_title = 'redshift, z',
                   cm = hm.cosmo, ylim = (1e-08, 1e+16) )
    return res

# integrand for effective galaxy bias
@plot
def effectiveBiasIntegrand(hm, z = [0., 2., 5.], start = 6., stop = 20., pts = 31):
    z = np.asfarray(z)
    m = np.logspace(start, stop, pts)
    if np.ndim( z ) >= 1: m = m[:, None]
    res = PlotData(np.divide( m, [nsigmaMass(5., zi, hm.cosmo) for zi in z] ),
                   hm.totalCount( m ) * hm.massFunction(m, z, 'dndlnm', False) * hm.biasFunction( m, z, False ) ,
                   xlabel = '$m/m_{5\\sigma}$', ylabel = '$f(m)N(m)\\frac{dn(m)}{dln(m)}$', 
                   title = 'effective bias integrand, $f(m) = b_h(m)$',
                   logx = True, logy = True, legend = z.tolist(), legend_title = 'redshift, z',
                   cm = hm.cosmo, ylim = (1e-08, 1e+06) )
    return res



# create interpolation tables
plank18.createInterpolationTables()

# normalisation
plank18.normalizePowerSpectrum()

# test halo model
hm = HaloModel(plank18, m_min = 1e+08, sigma = 0.2, m_sat = 1e+12, alpha = 1.)

galaxyDensityIntegrand(hm)
averageMassIntegrand(hm)
effectiveBiasIntegrand(hm)

