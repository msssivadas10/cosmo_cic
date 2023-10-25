#!/usr/bin/python3

import numpy as np
from random import choices
from dataclasses import dataclass
from typing import Any, Callable

###########################################################################################################
# Estimates with resampling
###########################################################################################################

@dataclass
class ResampleResult:

    num_samples: int  # number of sub samples used
    std_estimate: Any # values of the usual estimate of the statistics
    estimate: Any     # jackknife estimate of the statistics
    error: Any        # error in the jackknife estimate 
    bias: Any         # bias of the estimator

    @property
    def correct_estimate(self): 
        return self.std_estimate - self.bias # bias corrected estimate


def jackknife(statistic: Callable):
    r"""
    Extend a statistic estimator (e.g. mean) to do jackknife resampling. This allows to calculate the 
    estimate of the statistics as well as its error estimate. A jackknife estimator can be built by 
    aggregating the parameter estimates from each subsample of size (n-1) obtained by omitting one 
    observation (see https://en.wikipedia.org/wiki/Jackknife_resampling).

    E.g., to compute the jackknife average of a set of observations, create a function to compute the 
    average and decorate it with `jackknife`

    >>> @jackknife
    ... def mean(obs): 
    ...     return sum(obs) / len(obs)
    ...

    Then, for a set of 10 observations, 

    >>> obs = [0.13, 0.12, 0.10, 0.12, 0.09, 0.11, 0.11, 0.10] 
    >>> res = mean(obs)
    >>> res.estimate, res.error
    (0.10999999999999999, 0.004629100498862768)


    """

    def _jackknife(obs: Any, *args, **kwargs):
        
        obsShape = np.shape( obs ) # shape of the obsrvations array: last axis is the samples
        assert len( obsShape ) > 0, "atleast 2 observations are needed for jackknife resampling"

        obsCount   = obsShape[-1] # number of observations
        numSamples = obsCount # number of samples

        # usual estimator of the statistic
        estimate1 = np.apply_along_axis( statistic, -1, obs, *args, **kwargs )

        # statistics for each jackknife resamples (i-th sample contains all except the i-th one)
        sampleEstimate = [
                            np.apply_along_axis( 
                                                    statistic, 
                                                    -1, 
                                                    np.delete( obs, i, axis = -1 ), 
                                                    *args, 
                                                    **kwargs 
                                                ) 
                            for i in range( numSamples )
                        ]
        
        # jackknife estimator
        estimate2 = sum( sampleEstimate ) / numSamples

        # jackknife error
        error = np.sqrt( 
                            sum( np.square( np.subtract( estimate, estimate2 ) ) for estimate in sampleEstimate )
                                * ( numSamples - 1 ) / numSamples
                       ) 
        
        # bias
        bias = np.subtract( estimate2, estimate1 ) * ( numSamples - 1 )
        
        res = ResampleResult(num_samples  = numSamples, 
                             std_estimate = estimate1, 
                             estimate     = estimate2, 
                             error        = error, 
                             bias         = bias,     )
        return res
    
    return _jackknife


def bootstrap(statistic: Callable):
    r"""
    Extend a statistic estimator (e.g. mean) to do bootstrapping, i.e., random sampling with replacement
    (see https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). Each random sample set have size 100.

    E.g., to compute the bootstrap average of a set of observations, create a function to compute the 
    average and decorate it with `bootstrap`

    >>> @bootstrap
    ... def mean(obs): 
    ...     return sum(obs) / len(obs)
    ...

    Then, for a set of 10 observations, with average 0.11, 

    >>> obs = [0.13, 0.12, 0.10, 0.12, 0.09, 0.11, 0.11, 0.10] 
    >>> res = mean(obs)
    >>> res.estimate, res.error
    (0.11076249999999994, 0.037842851455063435)


    """

    def _bootstrap(obs: Any, *args, **kwargs):
        
        obsShape = np.shape( obs ) # shape of the obsrvations array: last axis is the samples
        assert len( obsShape ) > 0, "atleast 2 observations are needed for jackknife resampling"

        obsCount   = obsShape[-1] # number of observations
        numSamples = 100 # number of samples

        # usual estimator of the statistic
        estimate1 = np.apply_along_axis( statistic, -1, obs, *args, **kwargs )

        # statistics for each jackknife resamples (i-th sample contains all except the i-th one)
        sampleEstimate = [
                            np.apply_along_axis( 
                                                    statistic, 
                                                    -1, 
                                                    choices( obs, k = obsCount ), 
                                                    *args, 
                                                    **kwargs 
                                                ) 
                            for i in range( numSamples )
                        ]
        
        # jackknife estimator
        estimate2 = sum( sampleEstimate ) / numSamples

        # jackknife error
        error = np.sqrt( 
                            sum( np.square( np.subtract( estimate, estimate2 ) ) for estimate in sampleEstimate )
                                * ( numSamples - 1 ) / numSamples
                       )
        
        # bias
        bias = np.subtract( estimate2, estimate1 ) * ( numSamples - 1 )
        
        res = ResampleResult(num_samples  = numSamples, 
                             std_estimate = estimate1, 
                             estimate     = estimate2, 
                             error        = error, 
                             bias         = bias,      )
        return res
    
    return _bootstrap



