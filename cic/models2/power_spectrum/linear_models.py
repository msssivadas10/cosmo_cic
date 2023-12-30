#!/usr/bin/python3

import numpy as np
from typing import Any

from cic.models2._base import Cosmology 
from .._base import PowerSpectrum, Cosmology, CosmologyError

class PowerLaw(PowerSpectrum):
    r"""
    A power law model power spectrum (transfer function is 1).
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      
             deriv: int = 0, ) -> Any:
        if deriv:
            return np.zeros( np.broadcast_shapes( np.shape(k), np.shape(z) ) )
        res = np.ones_like(k)
        # interpolation with linear growth factor (normalised using value at z = 0)
        res = res * ( model.dplus( z ) / model.dplus( 0 ) )
        return res
    
class BBKS(PowerSpectrum):
    r"""
    Power spectrum model given by Bardeen et al (1986).
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any, 
             deriv: int = 0, ) -> Any:
        # wavenumber is in h/Mpc
        k = np.asfarray(k) 
        z = np.asfarray(z)

        # shape parameter
        gamma = model.Om0 * model.h

        q   = k / gamma
        res = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
        if deriv:
            res = 0.25 * ( 3.89*q + 2*(16.1*q)**2 + 3*(5.46*q)**3 + 4*(6.71*q)**4 ) / res + 1.
            q   = 2.34*q
            res = q / ( (1 + q) * np.log(1 + q) ) - res
        else:
            q   = 2.34*q
            res = res**-0.25 * np.log(1 + q) / q

        # interpolation with linear growth factor (normalised using value at z = 0)
        if deriv:
            res = res + np.zeros_like( z )
        else:
            res = res * model.dplus( z ) / model.dplus( 0. )

        return res
    
class Sugiyama95(BBKS):
    r"""
    Power spectrum model given by Bardeen et al (1986), including correction by Sugiyama (1995).
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any, 
             deriv: int = 0, ) -> Any:
        # model parameters
        Om0, Ob0, h = model.Om0, model.Ob0, model.h

        # baryonic mass causes a reduction of the shape parameter:
        shape_reduction = np.exp( -Ob0 - np.sqrt(2*h)*Ob0 / Om0 )
        k_eff = np.asfarray(k) / shape_reduction
        return super().call(model, k_eff, z, deriv)

class Eisenstein98_zeroBaryon(PowerSpectrum):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), not including baryon oscillations.
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,
             deriv: int = 0, ) -> Any:
        # wavenumber is in h/Mpc
        k = np.asfarray(k) 
        z = np.asfarray(z)
        
        # model parameters
        theta = model.Tcmb0 / 2.7
        h     = model.h
        Omh2, Obh2 = model.Om0 * h**2, model.Ob0 * h**2
        fb    = Obh2 / Omh2

        # Eqn. 26
        s = 44.5*np.log( 9.83/Omh2 ) / np.sqrt( 1 + 10*Obh2**0.75 )  
        # Eqn. 31
        a_gamma   = 1 - 0.328*np.log( 431*Omh2 ) * fb + 0.38*np.log( 22.3*Omh2 ) * fb**2
        # Eqn. 30  
        gamma_eff = Omh2 / h * ( a_gamma + ( 1 - a_gamma ) / ( 1 + ( 0.43*k*s )**4 ) ) 
        # Eqn. 28
        q  = k * ( theta**2 / gamma_eff ) 
        L = 2*np.e + 1.8*q
        C = 14.2 + 731.0 / ( 1 + 62.5*q )
        if deriv: # first log-derivative 
            res = 1.8 / L
            L   = np.log( L )
            res = res / L - ( res + 2*q * C - 62.5*q**2 * ( C - 14.2 )**2 / 731. ) / ( L + C*q**2 )
            res = q * res
        else:
            L   = np.log( L )
            res = L / ( L + C*q**2 )

        # interpolation with linear growth factor (normalised using value at z = 0)
        if deriv:
            res = res + np.zeros_like( z )
        else:
            res = res * model.dplus( z ) / model.dplus( 0. )

        return res

class Eisenstein98_withNeutrino(PowerSpectrum):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), including massive neutrinos.
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      
             deriv: int = 0, ) -> Any:
        # model parameters
        theta = model.Tcmb0 / 2.7
        h     = model.h
        Omh2, Obh2 = model.Om0 * h**2, model.Ob0 * h**2
        Nnu        = model.Nnu
        fb, fnu    = Obh2 / Omh2, model.Onu0 / model.Om0
        fc         = 1.0 - fb - fnu
        fcb, fnb   = fc + fb, fnu + fc

        # wavenumber is in 1/Mpc
        k = np.asfarray(k) * h
        z = np.asfarray(z)

        # redshift at matter-radiation equality: eqn. 1
        zp1_eq = 2.5e+4 * Omh2 / theta**4
        # redshift at drag epoch : eqn 2
        c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
        c2  = 0.238*Omh2**0.223
        z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)
        # eqn 3
        yd  = zp1_eq / (1 + z_d) 
        # sound horizon : eqn. 4
        s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))
        # eqn. 14 
        pc  = 0.25*( 5 - np.sqrt( 1 + 24.0*fc  ) ) 
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) ) 
        # small-scale suppression : eqn. 15
        alpha  = (fc / fcb) * (5 - 2 *(pc + pcb)) / (5 - 4 * pcb)
        alpha *= (1 - 0.533 * fnb + 0.126 * fnb**3) / (1 - 0.193 * np.sqrt(fnu * Nnu) + 0.169 * fnu * Nnu**0.2)
        alpha *= (1 + yd)**(pcb - pc)
        alpha *= (1 + 0.5 * (pc - pcb) * (1 + 1 / (3 - 4 * pc) / (7 - 4 * pcb)) / (1 + yd))
        # eqn. 16
        Gamma_eff = Omh2 * (np.sqrt(alpha) + (1 - np.sqrt(alpha)) / (1 + (0.43 * k * s)**4)) 

        # transfer function T_sup :
        q      = k * theta**2 / Gamma_eff # q_eff
        beta_c = (1 - 0.949 * fnb)**(-1)  # eqn. 21
        L      = (np.e + 1.84 * beta_c * np.sqrt(alpha) * q) # eqn. 19
        C      = 14.4 + 325 / (1 + 60.5 * q**1.08) # eqn. 20
        if deriv:
            res = 1.84 * beta_c * np.sqrt(alpha) / L
            L   = np.log( L )
            res = res / L - ( res + 2*q*C - 60.5 * 1.08*q**2.08 * (C - 14.4)**2 / 325. ) / ( L + C*q**2 )
            res = q * res
        else:
            L   = np.log( L )
            res = L / (L + C * q**2) # eqn. 18

        if fnu == 0:
            if deriv:
                res = res + np.zeros_like( z )
            else:
                # interpolation with linear growth factor
                res = res * model.dplus( z ) / model.dplus( 0 )
            return res 
        
        # master function:
        q   = k * theta**2 /  Omh2 # eqn 5 
        qnu = 3.92 * q * np.sqrt(Nnu / fnu) # eqn. 23
        B0  = (1.24 * fnu**0.64 * Nnu**(0.3 + 0.6 * fnu))
        Bk  = 1 + B0 / (qnu**(-1.6) + qnu**0.8) # eqn. 22
        if deriv:
            res = res - qnu / Bk * (Bk - 1.)**2 / B0 * ( -1.6*qnu**(-2.6) + 0.8*qnu**(-0.2) )
        else:
            res = res * Bk # eqn. 24

        # interpolation with linear growth factor (suppressed by free-streaming neutrino)
        dplus = model.dplus( z ) / model.dplus( 0 )
        yfs   = 17.2 * fnu * ( 1 + 0.488*fnu**(-7./6.) ) * ( Nnu / fnu )**2 * q**2
        x     = ( dplus / ( 1 + yfs ) )**0.7 
        y     = 1.0 # fcb**( 0.7 / pcb ) if including neutrino
        if deriv:
            res = res + ( pcb / 0.7 ) / ( y + x ) * 1.4*yfs * (1 + yfs)**0.3 / dplus**0.3
        else:
            dplus = ( y + x )**( pcb / 0.7 ) * dplus**(1 - pcb) # with free-streaming
            res = res * dplus

        return res


# initialising models to be readily used
powerlaw = PowerLaw()
bbks     = BBKS()
sugiyama95 = Sugiyama95()
eisenstein98_zb = Eisenstein98_zeroBaryon()
eisenstein98_nu = Eisenstein98_withNeutrino()

_available_models__ = {'powerlaw'       : powerlaw, 
                       'bbks'           : bbks, 
                       'sugiyama95'     : sugiyama95, 
                       'eisenstein98_zb': eisenstein98_zb, 
                       'eisenstein98_nu': eisenstein98_nu, }
