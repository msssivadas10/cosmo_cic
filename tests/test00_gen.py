#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from cic.models2 import cosmology, Cosmology
from cic.models2.stats.hod import Zheng07

cm = cosmology('plank18')
cm.link(power_spectrum = 'eisenstein98_zb', 
        window         = 'tophat', 
        mass_function  = 'tinker08', 
        halo_bias      = 'tinker10', 
        cmreln         = 'bullock01_powerlaw',
        halo_profile   = 'nfw',   )
# print( cm )

def test2():
    MUV_TH = -22.5 # uv magnitude threshold
    ZBAR   =  3.8  # redshift
    def z_distribution(z, cm = None): # for z = 3.8 (~ 4)
        res = np.interp(z, 
                        [3.00, 3.20, 3.40, 3.60, 4.00, 4.20, 4.40, 4.50],
                        [0.00, 0.02, 0.55, 0.75, 0.60, 0.55, 0.05, 0.00],
                        left  = 0.,
                        right = 0., )
        return res
    hm = Zheng07.harikane22(mag = MUV_TH, z = ZBAR, cosmology = cm, overdensity = 200)
    hm.setRedshiftDistribution(z_distribution, z_min = 3.0, z_max = 4.5)
    #################################################################
    logm_min = hm.logm_min - np.log10(cm.h) # in Msun       
    logm1    = hm.logm1    - np.log10(cm.h)        
    ng       = hm.averageDensity * cm.h**3  # Mpc**-3       
    beff     = hm.effectiveBias()                  
    mh       = hm.averageHaloMass()/cm.h 
    #################################################################
    print(f"log Mmin    : { logm_min     :10.3f}")
    print(f"log Msat    : { logm1        :10.3f}")
    print(f"avg. Density: { ng           :10.3e}")
    print(f"Eff. bg     : { beff         :10.3f}")
    print(f"log <Mh>    : { np.log10(mh) :10.3f}")
    #################################################################
    print(f"bias        : { hm.biasFunction(mh*cm.h, ZBAR) :10.3f}")
    print(f"density     : { hm.galaxyDensity(ZBAR)*cm.h**3 :10.3e}")
    return

def test1():
    # import gzip
    import scipy.interpolate as interp
    
    # ref_data = {}
    # with gzip.open('../ref/ref1/halo_data/sigma_massfn_bias_z0.00_EH_tinker_tinker.dat.gz') as file:
    #     header, *_data = file.read().decode().splitlines()
    #     ref_data = dict( zip( header.split(), np.asfarray( list( map( lambda __line: __line.split(), _data ) ) ).T ) )
    # if not ref_data:
    #     return print('cannot load data :(')
    
    # print( ref_data.keys() )

    cm = Cosmology(h = 0.6774, Om0 = 0.229 + 0.049, Ob0 = 0.049, sigma8 = 0.8159, ns = 0.9667)
    cm.link(power_spectrum = 'eisenstein98_zb', 
            window         = 'tophat', 
            mass_function  = 'tinker08', 
            halo_bias      = 'tinker10', 
            cmreln         = 'bullock01_powerlaw',
            halo_profile   = 'nfw',   )
    
    def interpolate(xp, yp, x, nu = 0):
        if nu:
            y = interp.CubicSpline(np.log( xp[::-1] ), np.log( yp[::-1] ) )( np.log(x), nu = nu ) 
            return y
        y = np.exp( interp.CubicSpline(np.log( xp[::-1] ), np.log( yp[::-1] ) )( np.log(x) ) )
        return y
    
    def max_error(x, ref):
        err = 100 * np.abs(x - ref) / np.abs(ref)
        return np.max(err)

    # growth factor
    # z, dz_ref = np.loadtxt('../ref/ref2/linear_growth.txt', delimiter = ',', comments = '#', unpack = True)
    # dz = cm.dplus( z ) / cm.dplus( 0 )
    # plt.figure()
    # plt.plot(z, dz_ref, '-', label = 'pyccl')
    # plt.plot(z, dz, '-', label = 'ente')
    # plt.title(f'growth: max. error: { max_error(dz, dz_ref) :.3g} %')
    # plt.legend()
    # plt.show()

    z = 0.

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

    plt.plot(x, y_ref, '-', label = 'pyccl')
    plt.plot(x, y, '-', label = 'ente')
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

if __name__ =='__main__':
    test1()
    # test2()
