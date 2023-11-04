#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import logging
import numpy as np
import matplotlib.pyplot as plt


# setup logging
logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [ logging.StreamHandler() ])
logging.disable()


datafile = 'p3catalog.csv'
outfile  = 'test_counts1.json'

def doCounting_old():

    from cic.measure.utils import Rectangle
    from cic.measure.counting import prepareRegion, countObjects
    from cic.measure.utils import CountResult
    from cic.measure.stats import jackknife

    prepareRegion(path = datafile,
                  output_path = outfile,
                  region = Rectangle(0., 0., 40., 10.),
                  patchsize_x = 10.,
                  patchsize_y = 10.,
                  pixsize = 1.,
                  bad_regions = [],
                  use_masks = ['g_mask', 'r_mask', 'i_mask'],
                  data_filters = [],
                  expressions = [],
                  x_coord = 'ra',
                  y_coord = 'dec',
                  chunksize = 10000,
                  header = 0,)
    
    countObjects(path = datafile, 
                 patch_details_path = outfile, 
                 output_path = outfile, 
                 include_patch_details = 1,
                 use_masks =  ['g_mask', 'r_mask', 'i_mask'], 
                 data_filters = ['g_magnitude < 22.0'], 
                 expressions = [], 
                 x_coord = 'ra', 
                 y_coord= 'dec', 
                 chunksize = 10000,
                 header = 0,)
    
    return

def doCounting():

    from cic.measure2.utils import Rectangle
    from cic.measure2.counting import prepareRegion, estimateCellQuality, estimateObjectCount


    csv_opts = dict(chunksize = 10000,
                    header    = 0,   )

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
                        csv_opts           = csv_opts,    )

    # estimate object counts
    estimateObjectCount(path               = datafile, 
                        patch_details_path = outfile, 
                        output_path        = None,
                        use_masks          = ['g_mask', 'r_mask', 'i_mask'], 
                        data_filters       = ['g_magnitude < 22.0'], 
                        expressions        = [], 
                        coord              = ['ra', 'dec'], 
                        csv_opts           = csv_opts,    )
    
    return

def checkResult():

    from scipy.special import gammaln, erf
    from cic.measure2.utils import CountResult
    from cic.measure2.stats import jackknife

    @jackknife
    def mean(x): return np.mean(x) # mean with jacknife resampling

    # some distribution functions:
    def poissonPMF(k, lam): return np.exp( k * np.log(lam) - lam - gammaln(k+1) )   
    def exponentialCDF(x, lam): return np.where( x < 0, 0., 1. - np.exp(-lam * x) ) 
    def normalCDF(x, loc, scale): return 0.5 * ( 1. + erf((x - loc) / scale) )      
    
    # load saved result
    res = CountResult.load( outfile )

    res.add( np.cumsum( res.get('total_count'   ), axis = -2 ), 'total_count'   , replace = True )
    res.add( np.cumsum( res.get('unmasked_count'), axis = -2 ), 'unmasked_count', replace = True )

    # estimated distribution
    bins = np.arange(195.5, 350.5, 5.0)
    hist = np.stack([np.histogram(res.get('total_count', 2, i).flatten(), 
                                  bins, 
                                  density = 1, )[0] for i in range(res.shape[-1])], -1)
    meanResult = mean(hist)

    x    = 0.5 * ( bins[1:] + bins[:-1] )
    yEst = meanResult.estimate
    yErr = meanResult.error

    # expected distribution: poisson process with normal distributed feature 'g_magnitude'
    density = 208979 / res.region.volume() # density = catalog size / area. original density = 500.
    area    = np.prod(res.pixsize)         # cell area
    lam     = density * area * normalCDF(22.0, loc = 22., scale = 5)
    yExp    = poissonPMF(x, lam = lam)

    plt.figure()
    plt.errorbar(x, yEst, yErr, fmt = 'o', capsize = 5, label = 'estimated')
    plt.plot(x, yExp, label = "expected")
    plt.legend(title = "count pmf")
    plt.xlabel("n"); plt.ylabel("p(n)") 
    plt.show()
    return


# doCounting()
checkResult()