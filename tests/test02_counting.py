#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, erf
from cic.measure2.utils import Rectangle
from cic.measure2.counting import prepareRegion, estimateCellQuality, estimateObjectCount
from cic.measure2.utils import CountResult
from cic.measure2.stats import jackknife


# setup logging
logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [ logging.StreamHandler() ])
# logging.disable()


datafile = 'p3catalog.csv'
outfile  = 'test_counts.json'

def doCounting():

    # prepare for counting
    prepareRegion(output_path = outfile,
                  region      = Rectangle(0., 0., 40., 10.),
                  patchsize   = [10., 10.], 
                  pixsize     = 1.0,
                  bad_regions = [], )

    # estimating cell completeness value
    estimateCellQuality(path               = datafile, 
                        patch_details_path = outfile,
                        output_path        = None,
                        use_masks          = ['g_mask', 'r_mask', 'i_mask'],
                        data_filters       = [],
                        expressions        = [],
                        coord              = ['ra', 'dec'],
                        chunksize          = 10000,
                        header             = 0,   )

    # estimate object counts
    estimateObjectCount(path               = datafile, 
                        patch_details_path = outfile, 
                        output_path        = None,
                        use_masks          = ['g_mask', 'r_mask', 'i_mask'], 
                        data_filters       = ['g_magnitude < 22.0'], 
                        expressions        = [], 
                        coord              = ['ra', 'dec'], 
                        chunksize          = 10000,
                        header             = 0,   )
    
    return


def checkResult():

    @jackknife
    def mean(x): return np.mean(x) # mean with jacknife resampling

    # some distribution functions:
    def poissonPMF(k, lam): return np.exp( k * np.log(lam) - lam - gammaln(k+1) )   # poisson distr. pmf
    def exponentialCDF(x, lam): return np.where( x < 0, 0., 1. - np.exp(-lam * x) ) # exponential distr. cdf
    def normalCDF(x, loc, scale): return 0.5 * ( 1. + erf((x - loc) / scale) )      # normal distr. cdf
    
    # load saved result
    res = CountResult.load( outfile )

    # estimated distribution
    bins       = np.arange(195.5, 350.5, 5.0)
    hist       = res.histogram('total_count', bins = bins, density = True).hist
    meanResult = mean(hist)
    # print(hist)

    x    = 0.5 * ( bins[1:] + bins[:-1] )
    yEst = meanResult.estimate
    yErr = meanResult.error


    # expected distribution: poisson process with normal distributed feature 'g_magnitude'
    density = 208979 / 400   # density = catalog size / area. original density = 500.
    area    = res.pixsize**2 # cell area
    lam     = density * area * normalCDF(22., loc = 22., scale = 5)
    yExp    = poissonPMF(x, lam = lam)

    plt.figure()
    plt.errorbar(x, yEst, yErr, fmt = 'o', capsize = 5)
    plt.plot(x, yExp)
    plt.show()

    return


doCounting()
checkResult()
