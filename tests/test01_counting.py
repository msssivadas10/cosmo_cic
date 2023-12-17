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

def generateCatalog():
    from cic.misc.generate_p3 import p3Generator1
    df = p3Generator1(ra1 = 0., ra2 = 40., dec1 = 0., dec2 = 10., density = 500.)
    df.to_csv('p3catalog.csv', index = False)
    return

def doCounting():
    from cic.measure2.utils import Rectangle
    from cic.measure2.counting import prepareRegion, estimateCellQuality, estimateObjectCount
    # general csv options 
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
    # mean with jacknife resampling
    @jackknife
    def mean(x): return np.mean(x) 
    # some distribution functions:
    def poissonPMF(k, lam): return np.exp( k * np.log(lam) - lam - gammaln(k+1) )   
    def exponentialCDF(x, lam): return np.where( x < 0, 0., 1. - np.exp(-lam * x) ) 
    def normalCDF(x, loc, scale): return 0.5 * ( 1. + erf((x - loc) / scale) )      
    # load saved result
    res = CountResult.load( outfile, 1 )
    res.add( np.cumsum( res.get('total_count'   ), axis = -2 ), 'total_count'   , replace = True )
    res.add( np.cumsum( res.get('unmasked_count'), axis = -2 ), 'unmasked_count', replace = True )
    # estimated distribution
    bins = np.arange(195.5, 350.5, 5.0)
    hist = np.stack([np.histogram(res['total_count', 2, i].flatten(), 
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
    # graph
    fig = plt.figure(figsize = (8, 9))
    ax1 = fig.add_axes([0.12, 0.1, 0.8, 0.2])
    ax1.errorbar(x, yEst, yErr, fmt = 'o', capsize = 5, label = 'Estimated')
    ax1.plot(x, yExp, label = "Expected")
    ax1.set(xlabel="n", ylabel="p(n)") 
    ax1.legend(title = "Count PMF")
    ax2 = fig.add_axes([0.12, 0.35, 0.8, 0.65])
    x,y = np.loadtxt(datafile, usecols = (0, 1), skiprows = 1, unpack = 1, delimiter = ',')
    m   = (x < 10) & (y < 10)
    ax2.plot(x[m], y[m], 'o', ms = 0.2)
    ax2.set(xlim = (-1., 11.), ylim = (-1., 11.))
    __x = 0.
    while __x <= res.patchsize[0]:
        ax2.plot([__x, __x], [0., 10.], lw = 1, color = 'black') 
        ax2.plot([0., 10.], [__x, __x], lw = 1, color = 'black') 
        __x += res.pixsize[0]
    ax2.annotate('', 
                 xy         = (1., -0.5), 
                 xytext     = (2., -0.5), 
                 xycoords   = 'data', 
                 textcoords = 'data', 
                 arrowprops = dict(arrowstyle = '|-|'))
    ax2.annotate('1 ut.', xy = (1.5, -0.3), ha = 'center', va = 'center')
    ax2.text(0.2, 0.2, 
             'Point density: %.2f per sq. ut.' % density, 
             bbox = dict(facecolor = 'white', edgecolor = 'black', boxstyle = 'round'))
    ax2.axis('equal')
    ax2.axis('off')
    fig.savefig('cic_example.png')
    # plt.show()
    return

def main():
    # generateCatalog()
    # doCounting()
    checkResult()
    return

if __name__ == '__main__':
    main()
    