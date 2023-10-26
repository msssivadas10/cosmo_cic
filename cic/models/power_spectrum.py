#!/usr/bin/python3

import numpy as np
from cic.models.constants import *
from scipy.integrate import simpson
from abc import ABC, abstractmethod
from typing import Any

#########################################################################################
# Smoothing window functions
#########################################################################################

def tophat(x: Any, nu: int = 0) -> Any:
    r"""
    Spherical tophat function.
    """

    x   = np.asfarray( x )
    res = np.zeros_like( x )

    # NOTE: applied some masks to make sure the function looks good!...
    
    # mask for x closer to 0
    closeToZero = ( np.abs(x) < 1e-03 )

    # mask for other smallr x; at both ends, the function values is practically 0
    otherValues = np.logical_not( closeToZero | ( np.abs(x) > 1e+04 ) )

    if nu == 0:
        res[ closeToZero ] = 1.0
        res[ otherValues ] = 3.0 * ( 
                                        np.sin( x[ otherValues ] ) - x[ otherValues ] * np.cos( x[ otherValues ] ) 
                                   ) / x[ otherValues ]**3
        return res
    elif nu == 1:
        # res[ closeToZero ] = 0.0
        res[ otherValues ] = 3.0 * ( 
                                        ( x[ otherValues ]**2 - 3.0 ) * np.sin( x[ otherValues ] ) 
                                            + 3.0 * x[ otherValues ] * np.cos( x[otherValues] ) 
                                   ) / x[ otherValues ]**4
        return res
    elif nu == 2:
        res[ closeToZero ] = -0.2
        res[ otherValues ] = 3.0 * ( 
                                        ( x[ otherValues ]**2 - 12.0 ) * x[ otherValues ] * np.cos( x[ otherValues ] ) 
                                            - ( 5 * x[ otherValues ]**2 - 12.0 ) * np.sin( x[ otherValues ] ) 
                                   ) / x[ otherValues ]**5
        return res
    
    raise ValueError("nu can only be 0, 1 or 2")


def gaussian(x: Any, nu: int = 0) -> Any:
    r"""
    Gaussian function.
    """

    x = np.asfarray( x )

    if nu == 0:
        return np.exp( -0.5*x**2 )
    elif nu == 1:
        return -x * np.exp( -0.5*x**2 )
    elif nu == 2:
        return ( x**2 - 1 ) * np.exp( -0.5*x**2 )

    raise ValueError("nu can only be 0, 1 or 2")


#########################################################################################
# POwer spectrum models
#########################################################################################

class PowerSpectrum( ABC ):
    r"""
    Base class representing a matter power spectrum object.
    """

    @abstractmethod
    def transfer(self, model: object, lnk: Any, lnzp1: Any, exact: bool = True) -> Any:
        r"""
        Calculate the log of linear transfer function.
        """
        ...

    def __call__(self, 
                 model: object, 
                 lnk: Any, 
                 lnzp1: Any, 
                 der: bool = False, 
                 exact: bool = True, ) -> Any:
        r"""
        Calculate the log of linear matter power spectrum.
        """
        
        # flatten any multi-dimensional arrays, if given
        lnk = np.ravel( lnk ) if np.ndim( lnk ) > 1 else lnk
        lnzp1 = np.ravel( lnzp1 ) if np.ndim( lnzp1 ) > 1 else lnzp1

        if not der:
            res = 2 * self.transfer( model, lnk, lnzp1, exact = exact ) # transfer function

            init = np.multiply( lnk, model.ns ) # initial power spectrum
            init = init[..., None] if np.ndim( res ) > 1 else init

            res = res + init 
            return res
        
        # derivative calculation with finite difference
        k  = np.exp( lnk ) # k0
        h  = 0.01 * k
        
        k     = k - 2*h # k0 - 2h
        lnk   = np.log( k )
        dlnpk = self.__call__( model, lnk, lnzp1, False, exact )

        k     = k + h   # k0 - h
        lnk   = np.log( k )
        dlnk  = lnk
        dlnpk = dlnpk - 8 * self.__call__( model, lnk, lnzp1, False, exact )

        k     = k + 2*h   # k0 + h
        lnk   = np.log( k )
        dlnk  = 6 * ( lnk - dlnk )
        dlnpk = dlnpk + 8 * self.__call__( model, lnk, lnzp1, False, exact )

        k     = k + h   # k0 + 2h
        lnk   = np.log( k )
        dlnpk = dlnpk - self.__call__( model, lnk, lnzp1, False, exact )

        res = np.transpose( np.transpose( dlnpk ) / dlnk )
        return res

    def matterVariance(self, 
                       model: object, 
                       lnr: Any, 
                       lnzp1: Any, 
                       nu: int = 0, 
                       exact: bool = True, 
                       window: str = 'tophat',
                       integral_pts: int = 10001, ) -> Any:
        r"""
        Calculate the linear matter variance or its first two derivatives.
        """

        assert nu in [0, 1, 2], "nu must be 0, 1 or 2"

        # flatten any multi-dimensional arrays, if given
        lnr   = np.ravel( lnr ) if np.ndim( lnr ) > 1 else lnr
        lnzp1 = np.ravel( lnzp1 ) if np.ndim( lnzp1 ) > 1 else lnzp1
        
        lnk, dlnk = np.linspace( np.log(EPS), np.log(INF), integral_pts, retstep = True )

        # dimensionless power spectrum k^3 * P(k)
        f1 = np.transpose( self.__call__( model, lnk, lnzp1, False, exact ) )
        f1 = np.exp( f1 + 3 * lnk )
        f1 = f1[:, None, :] if np.ndim( f1 ) > 1 else f1

        # window = 'gauss'
        if isinstance(window, str):
            window = tophat if window == 'tophat' else ( gaussian if window == 'gauss' else None )
        assert callable(window), "filter must be a callable or any of 'tophat' or 'gauss'" 

        # NOTE: from now, lnk and lnr are k and r...
        lnr = np.exp( lnr )
        lnk = np.exp( lnk )
        kr  = np.outer( lnr, lnk ) 
        f3  = window( kr, nu = 0 )

        # variance, s2
        f2   = 2 * f1 * f3
        res1 = simpson( f2 * f3, dx = dlnk, axis = -1 ) 
        if nu == 0:
            res1 = np.transpose( np.log( res1 * ONE_OVER_FOUR_PI2 ) ) 
            return res1
        
        # first log derivative, dln(s2)/dln(r)
        res1 = 2 * lnr / res1
        f3   = window( kr, nu = 1 ) * lnk
        res2 = simpson( f2 * f3, dx = dlnk, axis = -1 ) * res1
        if nu == 1:
            res2 = np.transpose( res2 )
            return res2

        # second log derivative, d2ln(s2)/d2ln(r)
        f2   = f2 * lnk * lnk * window( kr, nu = 2 ) + 2 * f1 * f3 * f3
        res2 = simpson( f2, dx = dlnk, axis = -1 ) * res1 * lnr + res2 * (1 - res2)
        res2 = np.transpose( res2 )
        return res2

    def matterCorrelation(self, 
                          model: object, 
                          r: Any, 
                          z: Any, 
                          exact: bool = True, 
                          integral_pts: int = 10001, ) -> Any:
        r"""
        Calculate the linear matter correlation function.
        """
        raise NotImplementedError()
    

# Pre-defined models    

class Eisenstein98_zeroBaryon( PowerSpectrum ):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), not including baryon oscillations.
    """

    def transfer(self, model: object, lnk: Any, lnzp1: Any, exact: bool = True) -> Any:

        k = np.exp( lnk )
        
        # cosmology parameters
        theta = model.Tcmb0 / 2.7
        Om0   = model.Om0
        Ob0   = model.Ob0
        h     = model.h

        Omh2, Obh2, fb = Om0 * h**2, Ob0 * h**2, Ob0 / Om0

        s = 44.5*np.log( 9.83/Omh2 ) / np.sqrt( 1 + 10*Obh2**0.75 )  # eqn. 26
        a_gamma   = 1 - 0.328*np.log( 431*Omh2 ) * fb + 0.38*np.log( 22.3*Omh2 ) * fb**2  # eqn. 31
        gamma_eff = Om0*h * ( a_gamma + ( 1 - a_gamma ) / ( 1 + ( 0.43*k*s )**4 ) ) # eqn. 30

        q = k * ( theta**2 / gamma_eff ) # eqn. 28
        l = np.log( 2*np.e + 1.8*q )
        c = 14.2 + 731.0 / ( 1 + 62.5*q )
        
        lnTk = np.log(l) - np.log( l + c*q**2 )

        # linear growth factor
        z    = np.exp( lnzp1 ) - 1 
        lnDz = np.log( model.dplus( z, exact = exact ) / model.dplus( 0, exact = exact ) )

        if np.ndim( lnDz ):
            lnTk = lnTk[:, None]

        lnTk = lnTk + lnDz
        return lnTk
    
    
class Eisenstein98_withNeutrino( PowerSpectrum ):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), including massive neutrinos.
    """

    def transfer(self, model: object, lnk: Any, lnzp1: Any, exact: bool = True) -> Any:

        theta  = model.Tcmb0 / 2.7
        Om0    = model.Om0
        Ob0    = model.Ob0
        h      = model.h
        Onu0   = model.Onu0 
        Nnu    = model.Nnu

        k = np.exp( lnk ) * h # Mpc^-1

        Omh2, Obh2 = Om0 * h**2, Ob0 * h**2
        fb, fnu    = Ob0 / Om0, Onu0 / Om0
        fc         = 1.0 - fb - fnu
        fcb, fnb   = fc + fb, fnu + fc

        assert fnu != 0

        # redshift at matter-radiation equality: eqn. 1
        zp1_eq = 2.5e+4 * Omh2 / theta**4

        # redshift at drag epoch : eqn 2
        c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
        c2  = 0.238*Omh2**0.223
        z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

        yd  = zp1_eq / (1 + z_d) # eqn 3

        # sound horizon : eqn. 4
        s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))

        q = k * theta**2 / Omh2 # eqn 5

        pc  = 0.25*( 5 - np.sqrt( 1 + 24.0*fc  ) ) # eqn. 14 
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) ) 

        # small-scale suppression : eqn. 15
        alpha  = (fc / fcb) * (5 - 2 *(pc + pcb)) / (5 - 4 * pcb)
        alpha *= (1 - 0.533 * fnb + 0.126 * fnb**3) / (1 - 0.193 * np.sqrt(fnu * Nnu) + 0.169 * fnu * Nnu**0.2)
        alpha *= (1 + yd)**(pcb - pc)
        alpha *= (1 + 0.5 * (pc - pcb) * (1 + 1 / (3 - 4 * pc) / (7 - 4 * pcb)) / (1 + yd))

        Gamma_eff = Omh2 * (np.sqrt(alpha) + (1 - np.sqrt(alpha)) / (1 + (0.43 * k * s)**4)) # eqn. 16
        qeff      = k * theta**2 / Gamma_eff

        # transfer function T_sup :
        beta_c = (1 - 0.949 * fnb)**(-1) # eqn. 21
        L      = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha) * qeff) # eqn. 19
        C      = 14.4 + 325 / (1 + 60.5 * qeff**1.08) # eqn. 20
        Tk_sup = L / (L + C * qeff**2) # eqn. 18

        # master function :
        qnu  = 3.92 * q * np.sqrt(Nnu / fnu) # eqn. 23
        Bk   = 1 + (1.24 * fnu**0.64 * Nnu**(0.3 + 0.6 * fnu)) / (qnu**(-1.6) + qnu**0.8) # eqn. 22
        lnTk = np.log( Tk_sup * Bk ) # eqn. 24

        # linear growth factor
        z    = np.exp( lnzp1 ) - 1
        lnDz = np.log( model.dplus( z, exact = exact ) / model.dplus( 0, exact = exact ) )

        # growth suppression by free-streaming neutrino
        pcb   = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) ) # eqn. 14
        q     = k * theta**2 / Omh2
        yfs   = 17.2 * fnu * ( 1 + 0.488*fnu**(-7./6.) ) * ( self.Nnu*q / fnu )**2
        x     = np.exp( ( lnDz - np.log( 1 + yfs[:, None] ) ) * 0.7 )
        y     = 1.0 # fcb**( 0.7 / pcb ) if including neutrino
        lnDcb = np.log( y + x ) * ( pcb / 0.7 ) +  lnDz * ( 1 - pcb ) # with free-streaming

        if np.ndim( lnDcb ):
            lnTk = lnTk[:, None]
        
        lnTk = lnTk + lnDcb
        return lnTk
    


# a dict of predefined models
available = {
                'eisenstein98_zb': Eisenstein98_zeroBaryon(), 
                'eisenstein98_nu': Eisenstein98_withNeutrino(),
            }

