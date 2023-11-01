#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def testCatelogGeneration():

    from cic.misc.generate_p3 import p3Generator1

    df = p3Generator1(ra1 = 0., ra2 = 40., dec1 = 0., dec2 = 10., density = 500.)
    df.to_csv('p3catalog.csv', index = False)
    return

def testCounting():

    import logging

    logging.basicConfig(level    = logging.INFO,
                        format   = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [ logging.StreamHandler() ])
    # logging.disable()

    from cic.measure.utils import Rectangle
    from cic.measure.counting import prepareRegion, countObjects

    prepareRegion(path         = 'p3catalog.csv', 
                  output_path  = 'test_counts.json',
                  region       = Rectangle(0., 0., 40., 10.),
                  patchsize_x  = 10., 
                  patchsize_y  = 10., 
                  pixsize      = 1.0,
                  bad_regions  = [],
                  use_masks    = ['g_mask', 'r_mask', 'i_mask'], 
                  data_filters = [], 
                  expressions  = [], 
                  x_coord      = 'ra', 
                  y_coord      = 'dec', 
                  chunksize    = 10000,
                  header       = 0, 
                )

    countObjects(path               = 'p3catalog.csv', 
                 patch_details_path = 'test_counts.json', 
                 output_path        = 'test_counts.json',
                 use_masks          = ['g_mask', 'r_mask', 'i_mask'], 
                 data_filters       = ['g_magnitude < 22.0'], 
                 expressions        = [], 
                 x_coord            = 'ra', 
                 y_coord            = 'dec', 
                 chunksize          = 10000,
                 header             = 0, 
                )
    
    return

def testResult():

    from cic.measure.utils import CountResult
    from cic.measure.stats import jackknife
    from scipy.special import gammaln, erf

    @jackknife
    def mean(x):
        return np.mean(x)
    
    def poissonPMF(k, lam):
        return np.exp( k * np.log(lam) - lam - gammaln(k+1) )

    def exponentialCDF(x, lam):
        return np.where( x < 0, 0., 1. - np.exp(-lam * x) )
    
    def normalCDF(x, loc, scale):
        return 0.5 * ( 1. + erf((x - loc) / scale) )
    
    
    res = CountResult.load('test_counts.json')

    pointDensity = 208979 / 400 # catalog size / area. original density = 500.
    cellArea     = res.pixsize**2

    bins       = np.arange(195.5, 350.5, 5.0)
    hist       = res.histogram('total_count', bins = bins, density = True).hist
    meanResult = mean(hist)
    # print(hist)

    x  = 0.5 * ( bins[1:] + bins[:-1] )
    y  = meanResult.estimate
    dy = meanResult.error

    lam = pointDensity * cellArea * normalCDF(22., loc = 22., scale = 5)
    yt  = poissonPMF(x, lam = lam)

    plt.figure()
    plt.errorbar(x, y, dy, fmt = 'o', capsize = 5)
    plt.plot(x, yt)
    plt.show()

    return

def testCosmology():

    from cic.models.cosmology import Cosmology
    # from cic.models import Cosmology
    # from cic.models.cosmology import plank18 as cm


    cm = Cosmology(h = 0.7, Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, name = 'test_cosmology')
    cm.set(power_spectrum = 'eisenstein98_zb', mass_function = 'tinker08', halo_bias = 'tinker10')
    cm.normalizePowerSpectrum() 
    
    
    fig, axs = plt.subplots(2, 3, figsize = [12, 6])
    plt.subplots_adjust(wspace = 0.3, hspace = 0.4)

    args = dict(marker = 's', ls = '-', ms = 4)

    zi, z_labels = [0, 2, 5], ['0', '2', '5'] 

    k = np.logspace(-3, 3, 21)  

    axs[0,0].loglog(k, cm.matterPowerSpectrum(k, z = zi, grid = True), **args)
    axs[0,0].legend(z_labels, title = 'redshift, z')
    axs[0,0].set(xlabel = 'k', ylabel = 'P(k, z)', title = 'power spectrum')

    r = np.logspace(-2, 2, 21)  

    axs[1,0].loglog(k, cm.matterVariance(r, z = zi, grid = True), **args)
    axs[1,0].legend(z_labels, title = 'redshift, z')
    axs[1,0].set(xlabel = 'r', ylabel = '$\\sigma^2$(k, z)', title = 'variance')

    m = np.logspace(6, 14, 21)  

    axs[0,1].loglog(m, cm.massFunction(m, z = zi, retval = 'dndlnm', grid = True), **args)
    axs[0,1].legend(z_labels, title = 'redshift, z')
    axs[0,1].set(xlabel = 'm', ylabel = 'dn/dlnm', title = 'mass function')

    axs[1,1].loglog(m, cm.biasFunction(m, z = zi, grid = True), **args)
    axs[1,1].legend(z_labels, title = 'redshift, z')
    axs[1,1].set(xlabel = 'm', ylabel = '$b_1$(m)', title = 'linear bias')

    z = np.linspace(0., 5., 21) 

    axs[0,2].plot(z, cm.dplus(z, 0), **args)
    axs[0,2].plot(z, cm.dplus(z, 1), **args)
    axs[0,2].set(xlabel = 'z', ylabel = '$D_+$(z)', title = 'linear growth factor')

    axs[1,2].plot(z, cm.angularDiameterDistance(z), **args)
    axs[1,2].set(xlabel = 'z', ylabel = '$d_A$(z)', title = 'angular dimeter distance')

    plt.show()
    

    return

def testProfile():

    from cic.models.cosmology import plank18
    from cic.models.halos.density.cmrelations import bullock01, zheng07
    from cic.models.halos.density.profiles import NFW


    plank18.createInterpolationTables()
    plank18.normalizePowerSpectrum()

    nfw = NFW(cm = bullock01)

    m = 1e+13
    z = [0., 2., 4.]
    r = np.logspace(-5, 5, 51)

    fig, axs = plt.subplots(3, 1, sharex=1)
    plt.subplots_adjust(hspace=0)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90], zorder=-1)
    ax.spines[:].set_visible(False)
    ax.set(xlabel='k', ylabel='u(k|m)', xticks = [], yticks = [])
    for i, m in enumerate([1e+08, 1e+10, 1e+12]):
        y = nfw(plank18, r, m, z, 200., ft = 1, truncate = 0, grid = 0)
        axs[i].semilogx()
        axs[i].plot(r, y)
        axs[i].legend(z, title="m=%g\nz:" % m, loc='lower left')
    plt.show()


    return

def testHaloModel():

    from cic.models.cosmology import plank18
    from cic.models.halo_model import HaloModel
    from scipy.optimize import brentq


    plank18.createInterpolationTables()
    plank18.normalizePowerSpectrum()

    hm = HaloModel(plank18, m_min = 1e+08, sigma = 0.2, m_sat = 1e+12, alpha = 1.)

    z = np.linspace(0, 5, 11)
    m = np.logspace(6, 20, 201)
    # y = hm.galaxyDensity(z)

    y1, y2 = 1e-08, 1e+05
    plt.figure()
    for i, z in enumerate([0.0, 5.0, 10.0]):
        m5 = plank18.lagrangianM(np.exp(brentq(lambda lnr, z: plank18.peakHeight(np.exp(lnr), z) - 5., 
                                               np.log(1e-03), 
                                               np.log(1e+03), 
                                               (z, ))))
        y = hm.totalCount( m ) * hm.massFunction(m, z, 'dndlnm', False) * hm.biasFunction( m, z, False )
        plt.loglog(m, y, '-', color = 'C%d'%i, label = '%g' % z)
        plt.vlines(m5, y1, y2, ['C%d'%i], ['--'], lw = 1)
    plt.ylim(y1, y2)
    plt.title("$m \\frac{dn(m)}{d\\ln(m)}N(m)b(m)$")
    plt.legend(title = 'z')
    plt.show()

    return


if __name__ == '__main__':
    print("testing...")

    # testCatelogGeneration()
    # testCounting()
    # testResult()
    # testCosmology()
    # testProfile()
    testHaloModel()