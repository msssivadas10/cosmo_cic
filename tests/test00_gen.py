#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from cic.models2 import cosmology, Cosmology, PowerSpectrum
from cic.models2.stats.hod import Zheng07
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
    

cm = cosmology('plank18')
cm.link(power_spectrum = 'eisenstein98_zb', 
        window         = 'tophat', 
        mass_function  = 'tinker08', 
        halo_bias      = 'tinker10', 
        cmreln         = 'bullock01_powerlaw',
        halo_profile   = 'nfw',   )
# print( cm )

############################################################################################

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

def max_error(x, ref):
    err = 100 * np.abs(x - ref) / np.abs(ref)
    return np.max(err)

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

    cm = Cosmology(h = 0.6774, 
                   Om0 = 0.229 + 0.049, 
                   Ob0 = 0.049, 
                   sigma8 = 0.8159, 
                   ns = 0.9667)
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

def test3():
    z, ng, mh, bg = np.loadtxt('../ref/hod_values.txt', delimiter = ',', comments = '#', unpack = True)

    h = 1. # cm.h
    hm = Zheng07(logm_min = np.log10(1e11 / h), 
                 sigma_logm = 0.25, 
                 logm0 = np.log10(1e8 / h), 
                 logm1 = np.log10(1e12 / h), 
                 alpha = 0.8, 
                 cosmology = cm, 
                 overdensity = 200, )
    # hm.settings.m_quad

    #==========================================
    # galaxy density
    #==========================================
    y_ref = ng
    y     = hm.galaxyDensity( z ) * cm.h**3
    ylab  = '$n_g$'

    plt.figure()
    # plt.loglog()
    # plt.plot(z, 100*np.abs(y - y_ref)/np.abs(y_ref), '-', label = 'pyccl')
    plt.plot(z, y / y_ref, '-', label = 'pyccl')
    # plt.plot(z, y_ref, '-', label = 'pyccl')
    # plt.plot(z, y, '-', label = 'ente')
    plt.xlabel( 'z' )
    plt.ylabel( ylab )
    plt.title(f'max. error: { max_error(y, y_ref) :.3g} %')
    plt.legend()
    plt.show()
    return

def test4():
    from scipy.integrate import simpson

    h = 1. # cm.h
    hm = Zheng07(logm_min = np.log10(1e11 / h), 
                 sigma_logm = 0.25, 
                 logm0 = np.log10(1e8 / h), 
                 logm1 = np.log10(1e12 / h), 
                 alpha = 0.8, 
                 cosmology = cm, 
                 overdensity = 200, )

    z, ng, mh, bg = np.loadtxt('../ref/ref3/hod_values.txt', delimiter = ',', comments = '#', unpack = True)
    mf_table = np.loadtxt('../ref/ref3/hod_values_mf.txt', delimiter = ',', comments = '#')
    bf_table = np.loadtxt('../ref/ref3/hod_values_bf.txt', delimiter = ',', comments = '#')

    plt.figure()
    # plt.loglog()
    # plt.semilogy()
    plt.semilogx()


    # for zz in range(0, len(z), 10):
    #     y = hm.massFunction( mf_table[:,0] * cm.h, z[ zz ] ) #* cm.h
    #     plt.plot( mf_table[:,0], 100 * np.abs( y / mf_table[:,zz+2] - 1. ), label = '%.2f' % z[ zz ] )
    #     # plt.plot( mf_table[:,0], mf_table[:,zz+2] )
    #     # plt.plot( mf_table[:,0], y, '+')
    # plt.legend(title = 'z')

    for zz in range(0, len(z), 10):
        y = hm.biasFunction( bf_table[:,0] * cm.h, z[ zz ] ) #* cm.h
        plt.plot( bf_table[:,0], 100 * np.abs( y / bf_table[:,zz+2] - 1. ), label = '%.2f' % z[ zz ] )
        # plt.plot( mf_table[:,0], mf_table[:,zz+2] )
        # plt.plot( mf_table[:,0], y, '+')
    plt.legend(title = 'z')


    x = np.log10( mf_table[:,0] )
    y = []
    for zz in range(len(z)):
        # y = hm.massFunction( mf_table[:,0] * cm.h, z[ zz ] ) #* cm.h**3
        # plt.plot( mf_table[:,0], mf_table[:,zz+2] )
        # plt.plot( mf_table[:,0], y)
        # plt.plot( mf_table[:,0], mf_table[:,zz+2] / y )

        ...
        # y.append( simpson( mf_table[:,1] * mf_table[:,zz+2], x = x ) )
        # y.append( simpson( mf_table[:,0] * mf_table[:,1] * mf_table[:,zz+2], x = x ) )
        # y.append( simpson( bf_table[:,zz+2] * mf_table[:,1] * mf_table[:,zz+2], x = x ) )
    # y = np.asfarray(y)

    ## total
    # plt.plot( mf_table[:,0], mf_table[:,1] )
    # plt.plot( mf_table[:,0], hm.totalCount( mf_table[:,0] ) )

    # plt.plot(z, ng) 
    # plt.plot(z, y, '+') 

    plt.show()

    return

def test5():

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
    
    ##
    z = 0.

    fig = plt.figure()
    plt.semilogx()
    # plt.loglog()

    for z in [ 0., 1., 2., 4., ]:
        # power
        k, pk_ref = np.loadtxt('../ref/ref2/linear_power_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
        pk = cm.matterPowerSpectrum( k[::20]/cm.h, z ) / cm.h**3
        plt.plot(k[::20], np.abs( pk / pk_ref[::20] - 1 )*100, '-', label = '%.2f' % z)

        # m, s_ref, dlns_ref, f_ref, hmf_ref, b_ref = np.loadtxt('../ref/ref2/halo_props_z%.2f.txt' % z, delimiter = ',', comments = '#', unpack = True)
        # # y_ref, y = s_ref**2, cm.matterVariance( cm.lagrangianR( m*cm.h ), z ) # var
        # # y_ref, y = hmf_ref, cm.haloMassFunction( m*cm.h, z, 200 ) * cm.h**3 # massfunc
        # y_ref, y =  b_ref, cm.haloBias(m*cm.h, z, 200) # bias
        # plt.plot(m, np.abs( y / y_ref - 1) * 100, '-', label = '%.2f' % z)


    plt.legend(title = 'z')
    plt.show()

    return

def main():
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
    return

if __name__ =='__main__':
    main()