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

    from cic.models import Cosmology
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

def testHaloModel():

    from cic.models.cosmology import plank18
    from cic.models.halo_model import HaloModel
    from cic.models.halos.profiles import nfw


    plank18.createInterpolationTables()
    plank18.normalizePowerSpectrum()


    hm = HaloModel(plank18, m_min = 1e+08, sigma = 0.2, m_sat = 1e+12, alpha = 1.)

    x = np.linspace(0, 5, 11)
    z = [0.0, 1.0]
    
    # y = hm.galaxyDensity(x)
    # y = hm.averageHaloMass(x)
    y = hm.effectiveBias(x)



    # # r = np.logspace(-2, 2, 201)
    # # m = 1e+10
    # # z = 0.0

    # # y = nfw(plank18, r, m, z, 200).flatten()

    plt.figure()
    # plt.semilogy()
    plt.plot(x, y, 'o-')
    # plt.plot(r, np.ones_like(r))
    plt.show()

    return

if __name__ == '__main__':
    print("testing...")

    # testCatelogGeneration()
    # testCounting()
    # testResult()
    # testCosmology()
    testHaloModel()