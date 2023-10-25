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

    pointDensity = 500.
    cellArea     = res.pixsize**2

    bins       = np.arange(195.5, 350.5, 5.0)
    hist       = res.histogram('total_count', bins = bins, density = True).hist
    # print(hist)
    meanResult = mean(hist)
    
    x  = 0.5 * ( bins[1:] + bins[:-1] )
    y  = meanResult.estimate
    dy = meanResult.error

    pointDensity = 520.
    lam = pointDensity * cellArea * normalCDF(22., loc = 22., scale = 5)
    yt  = poissonPMF(x, lam = lam)

    plt.figure()
    plt.errorbar(x, y, dy, fmt = 'o', capsize = 5)
    plt.plot(x, yt)
    plt.show()

    return

if __name__ == '__main__':
    print("testing...")

    # testCatelogGeneration()
    # testCounting()
    testResult()