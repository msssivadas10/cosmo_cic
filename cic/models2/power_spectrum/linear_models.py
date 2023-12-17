#!/usr/bin/python3

import numpy as np
from typing import Any 
from .._base import PowerSpectrum, Cosmology, CosmologyError

class PowerLaw(PowerSpectrum):
    r"""
    A power law model power spectrum (transfer function is 1).
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      ) -> Any:
        res = np.ones_like(k)
        # interpolation with linear growth factor (normalised using value at z = 0)
        res = res * ( model.dplus( z ) / model.dplus( 0 ) )
        return res

class Eisenstein98_zeroBaryon(PowerSpectrum):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), not including baryon oscillations.
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      ) -> Any:
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
        cq2 = k * ( theta**2 / gamma_eff ) # now it is just `q`... 
        res = np.log( 2*np.e + 1.8*cq2 )
        cq2 = ( 14.2 + 731.0 / ( 1 + 62.5*cq2 ) ) * cq2**2 
        res = res / ( res + cq2 )
        # interpolation with linear growth factor (normalised using value at z = 0)
        res = res * model.dplus(z) / model.dplus(0)
        return res

class Eisenstein98_withNeutrino(PowerSpectrum):
    r"""
    Power spectrum model given by Eisentein & Hu (1998), including massive neutrinos.
    """
    def call(self, 
             model: Cosmology, 
             k: Any, 
             z: Any,      ) -> Any:
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

        if fnu == 0:
            raise CosmologyError("cosmology does not include any massive neutrinos")

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
        # eqn 5
        q = k * theta**2 / Omh2 
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
        qeff      = k * theta**2 / Gamma_eff
        # transfer function T_sup :
        beta_c = (1 - 0.949 * fnb)**(-1) # eqn. 21
        L      = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha) * qeff) # eqn. 19
        C      = 14.4 + 325 / (1 + 60.5 * qeff**1.08) # eqn. 20
        res    = L / (L + C * qeff**2) # eqn. 18
        # master function :
        qnu = 3.92 * q * np.sqrt(Nnu / fnu) # eqn. 23
        Bk  = 1 + (1.24 * fnu**0.64 * Nnu**(0.3 + 0.6 * fnu)) / (qnu**(-1.6) + qnu**0.8) # eqn. 22
        res = res * Bk # eqn. 24
        # linear growth factor
        dplus = model.dplus( z ) / model.dplus( 0 )
        # linear growth factor: incl. suppression by free-streaming neutrino
        pcb   = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) ) # eqn. 14
        q     = k * theta**2 / Omh2
        yfs   = 17.2 * fnu * ( 1 + 0.488*fnu**(-7./6.) ) * ( Nnu*q / fnu )**2
        x     = ( dplus / ( 1 + yfs ) )**0.7 
        y     = 1.0 # fcb**( 0.7 / pcb ) if including neutrino
        dplus = ( y + x )**( pcb / 0.7 ) * dplus**( 1 - pcb ) # with free-streaming
        # interpolation with linear growth factor
        res = res * dplus
        return res


# initialising models to be readily used
powerlaw = PowerLaw()
eisenstein98_zb = Eisenstein98_zeroBaryon()
eisenstein98_nu = Eisenstein98_withNeutrino()

def _init_module() -> None:
    PowerSpectrum.available.add('powerlaw', powerlaw)
    PowerSpectrum.available.add('eisenstein98_zb', eisenstein98_zb)
    PowerSpectrum.available.add('eisenstein98_nu', eisenstein98_nu)
    return