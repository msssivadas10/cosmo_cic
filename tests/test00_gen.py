#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from cic.models2 import Cosmology, PowerSpectrum
from cic.models2.stats.hod import HM5
from typing import Any 

class SampledPS(PowerSpectrum):
    __slots__ = 'lnk', 'lnt', 'interp'
    def __init__(self, lnk: Any, lnt: Any) -> None:
        super().__init__()
        self.lnk, self.lnt = lnk, lnt # lnp = 3*lnk + 2*lnt
        self.interp = CubicSpline(lnk, lnt)

    def call(self, model: Cosmology, k: Any, z: Any, deriv: int = 0, **kwargs: Any) -> Any:
        res = np.exp( self.interp( np.log(k), nu = 0 ) )
        # interpolation with linear growth factor (normalised using value at z = 0)
        if deriv:
            res = res * self.interp( np.log(k), nu = 1 ) / np.asfarray(k)
            res = res + np.zeros_like( z )
        else:
            res = res * model.dplus( z ) / model.dplus( 0. )
        return res
    
def get_cosmology():
    cm = Cosmology(h = 0.6774, 
                   Om0 = 0.229 + 0.049, 
                   Ob0 = 0.049, 
                   sigma8 = 0.8159, 
                   ns = 0.9667)
    lnh = np.log( cm.h )

    x, y = np.loadtxt('../ref/ref2/linear_power_z0.00.txt', delimiter = ',', comments = '#', unpack = True)

    # convert k from mpc^-1 to h/mpc
    x = np.log(x) - lnh

    # convert p(k) from mpc^-3 to (h/mpc)^3
    y = np.log(y) - 3*lnh

    # convert ln(p) to ln(t)
    y = 0.5*( y - cm.ns*x )
    y = y - np.max(y)

    cm.link(power_spectrum = SampledPS(x, y), 
            window         = 'tophat', 
            mass_function  = 'tinker08', 
            halo_bias      = 'tinker10', 
            cmreln         = 'bullock01_powerlaw',
            halo_profile   = 'nfw',   )
    return cm

def max_error(x, ref):
    err = 100 * np.abs(x - ref) / np.abs(ref)
    return np.max(err)

def test1():
    # import gzip
    import scipy.interpolate as interp

    cm = get_cosmology()
    
    # ref_data = {}
    # with gzip.open('../ref/ref1/halo_data/sigma_massfn_bias_z0.00_EH_tinker_tinker.dat.gz') as file:
    #     header, *_data = file.read().decode().splitlines()
    #     ref_data = dict( zip( header.split(), np.asfarray( list( map( lambda __line: __line.split(), _data ) ) ).T ) )
    # if not ref_data:
    #     return print('cannot load data :(')
    
    # print( ref_data.keys() )
    
    def interpolate(xp, yp, x, nu = 0):
        if nu:
            y = interp.CubicSpline(np.log( xp[::-1] ), np.log( yp[::-1] ) )( np.log(x), nu = nu ) 
            return y
        y = np.exp( interp.CubicSpline(np.log( xp[::-1] ), np.log( yp[::-1] ) )( np.log(x) ) )
        return y
    

    # growth factor
    # z, dz_ref = np.loadtxt('../ref/ref2/linear_growth.txt', delimiter = ',', comments = '#', unpack = True)
    # dz = cm.dplus( z ) / cm.dplus( 0 )
    # plt.figure()
    # plt.plot(z, dz_ref, '-', label = 'pyccl')
    # plt.plot(z, dz, '-', label = 'ente')
    # plt.title(f'growth: max. error: { max_error(dz, dz_ref) :.3g} %')
    # plt.legend()
    # plt.show()

    z = 4.

    # power spectrum
    # k, pk_ref = np.loadtxt('../ref/ref2/linear_power_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
    # pk = cm.matterPowerSpectrum( k/cm.h, z ) / cm.h**3
    # fig = plt.figure()
    # plt.loglog()
    # plt.plot(k, pk_ref, '-', label = 'pyccl')
    # plt.plot(k, pk, '-', label = 'ente')
    # plt.xlabel( 'k ($Mpc^{-1}$)' )
    # plt.ylabel( 'P(k) ($Mpc^3$)' )
    # plt.title(f'linear power: max. error: { max_error(pk, pk_ref) :.3g} %')
    # plt.legend()
    # ax = fig.add_axes([0.22, 0.2, 0.4, 0.3])
    # ax.loglog( k[(k > 2e-3) & (k < 5e-2)], pk_ref[(k > 2e-3) & (k < 5e-2)] )
    # ax.loglog( k[(k > 2e-3) & (k < 5e-2)],     pk[(k > 2e-3) & (k < 5e-2)] )
    # plt.show()

    # mass function
    m, s_ref, dlns_ref, f_ref, hmf_ref, b_ref = np.loadtxt('../ref/ref2/halo_props_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
    plt.figure()

    #==================================================
    # variance
    #==================================================
    # x     = m
    # y_ref = s_ref**2
    # y     = cm.matterVariance( cm.lagrangianR( m*cm.h ), z )
    # xlab  = 'm [Msun]'
    # ylab  = 'sigma^2(m)'  
    # plt.loglog()

    #==================================================
    # mass-function f(s)
    #==================================================
    # x     = s_ref
    # y_ref = f_ref
    # y     = cm.mass_function.call( cm, x, z, 200 )
    # xlab = 'sigma'   
    # ylab = 'f(sigma)' 

    #==================================================
    # mass-function dndlnm
    #==================================================
    x     = m
    y_ref = hmf_ref
    y     = cm.haloMassFunction( m*cm.h, z, 200 ) * cm.h**3
    xlab = 'm [Msun]'   
    ylab = 'dn/dm [Mpc^-3]' 
    plt.loglog()

    #==================================================
    # bias b(s) 
    #==================================================
    # x     = s_ref
    # y_ref = b_ref
    # y     = cm.halo_bias.call( cm, 1.686 / x, z, 200 )
    # xlab  = 'sigma'   
    # ylab  = 'b(sigma)' 

    #==================================================
    # bias b(m) 
    #==================================================
    # x     = m
    # y_ref =  b_ref
    # y     =  cm.haloBias(m*cm.h, z, 200)
    # xlab  = 'm [Msun]'   
    # ylab  = 'b(m)'
    # plt.semilogx() 

    plt.plot(x, np.abs(y - y_ref)/np.abs(y_ref), '-', label = 'pyccl')
    # plt.plot(x, y, '-', label = 'ente')
    plt.xlabel( xlab )
    plt.ylabel( ylab )
    plt.title(f'max. error: { max_error(y, y_ref) :.3g} %')
    plt.legend()
    plt.show()

    ######################################################################################################3

    # m = np.logspace( 10., 16., 21 )
    # r = cm.lagrangianR( m )
    #
    # plt.figure()
    # plt.semilogx()
    # plt.loglog()
    # y = interpolate( ref_data[ '#Mass(Mpc)' ], ref_data[ 'sigma' ], m/cm.h )
    # y = -interpolate( ref_data[ '#Mass(Mpc)' ], -ref_data[ 'dsigma_dm' ] * ref_data[ '#Mass(Mpc)' ] / ref_data[ 'sigma' ], m/cm.h )
    # y = interpolate( ref_data[ '#Mass(Mpc)' ], ref_data[ 'dn/dlogM' ], m/cm.h )
    # y = interpolate( ref_data[ '#Mass(Mpc)' ], ref_data[ 'bias' ], m/cm.h )
    # plt.plot( m/cm.h, y, 's-' )
    # y = cm.matterVariance( r, z = 0.0 )**0.5
    # y = cm.matterVariance( r, z = 0.0, deriv = 1 ) / 6.
    # y = cm.haloMassFunction( m, z = 0., retval = 'dndlnm' )
    # y = cm.haloBias( m, z = 0. )
    # plt.plot( m/cm.h, y, 'o', ms = 4 )
    # plt.xlabel('M (Msun)')
    # plt.show()
    return

def test2():
    cm = get_cosmology()

    fig = plt.figure()
    plt.semilogx()
    # plt.loglog()

    for z in [ 0., 1., 2., 4., ]:
        # power
        # k, pk_ref = np.loadtxt('../ref/ref2/linear_power_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
        # pk = cm.matterPowerSpectrum( k[::20]/cm.h, z ) / cm.h**3
        # plt.plot(k[::20], np.abs( pk / pk_ref[::20] - 1 )*100, '-', label = '%.2f' % z)

        m, s_ref, dlns_ref, f_ref, hmf_ref, b_ref = np.loadtxt('../ref/ref2/halo_props_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
        # y_ref, y = s_ref**2, cm.matterVariance( cm.lagrangianR( m*cm.h ), z ) # var
        y_ref, y = hmf_ref, cm.haloMassFunction( m*cm.h, z, 200 ) * cm.h**3 # massfunc
        # y_ref, y =  b_ref, cm.haloBias(m*cm.h, z, 200) # bias
        plt.plot(m, np.abs( y / y_ref - 1) * 100, '-', label = '%.2f' % z)


    plt.legend(title = 'z')
    plt.show()
    return

def test3():
    from scipy.integrate import simpson
    import scipy.special as sf

    cm = get_cosmology()

    h = cm.h
    lnh = np.log10(h)
    logMmin = 11.0
    sigma = 0.25
    logM0 = 8.0
    logM1 = 12.0
    alpha = 0.8
    m0 = 10**logM0
    m1 = 10**logM1

    hm = HM5(logm_min = logMmin + lnh, 
             sigma_logm = sigma, 
             logm0 = logM0 + lnh, 
             logm1 = logM1 + lnh, 
             alpha = alpha, 
             cosmology = cm, 
             overdensity = 200, )

    z, ng, mh, bg = np.loadtxt('../ref/ref3/hod_values.txt', delimiter = ',', comments = '#', unpack = True)
    mf_table = np.loadtxt('../ref/ref3/hod_values_mf.txt', delimiter = ',', comments = '#')
    bf_table = np.loadtxt('../ref/ref3/hod_values_bf.txt', delimiter = ',', comments = '#')

    m  = mf_table[:,0] 
    nt = hm.totalCount(m*h)

    plt.figure()

    # plt.loglog()
    # plt.plot( m, mf_table[:,1] + 1e-08 )
    # plt.plot( m, nt + 1e-08, '+' )

    # plt.loglog()
    # plt.plot( m, hm.satelliteFraction(m) + 1e-08 )
    
    plt.semilogy()
    # y_ref, y = ng, hm.galaxyDensity( z ) * cm.h**3
    # y_ref, y = mh, hm.averageHaloMass( z ) * cm.h**2
    y_ref, y = bg, hm.effectiveBias( z ) * cm.h**3
    
    plt.plot(z, y_ref) 
    plt.plot(z, y, '+') 
    plt.twinx().plot( z, np.abs( y / y_ref - 1.)*100, color = 'green', lw = 0.7 )

    plt.show()
    return


def main():
    # test1()
    # test2()
    test3()
    return

if __name__ =='__main__':
    main()