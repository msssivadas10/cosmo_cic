#!/usr/bin/python3

import numpy as np
from typing import Any

from .._base import Cosmology, HaloConcentrationMassRelation

class NFW97(HaloConcentrationMassRelation):
    r"""
    Concentration mass relation in NFW (1997).
    """
    def __init__(self, f: float, ) -> None:
        super().__init__()
        # model parameters
        self.f = f

    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        raise NotImplementedError()

class Bullock01(HaloConcentrationMassRelation):
    r"""
    Concentration mass relation given by Bullock et. al. (2001).

    Parameters
    ----------
    f, k: float, optional
        Default values are `f = 0.01` and `k = 4.0`.

    """
    def __init__(self, 
                 f: float = 0.01, 
                 k: float = 4.0, ) -> None:
        super().__init__()
        # model parameters
        self.f, self.k = f, k

    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        # collapse redshift
        rc = model.lagrangianR( self.f * m, overdensity ) # radius at collapse
        zc = model.collapseRedshift(rc)
        # concentration
        res = self.k * np.add(zc, 1.) / np.add(z, 1.)
        return res

class Bullock01_PowerLaw(HaloConcentrationMassRelation):
    r"""
    Concentration mass relation (power law) given by by Bullock et. al. (2001).

    Parameters
    ----------
    params: str, default = `zheng07`
        Specify which pre-defined set of parameters are used.
    c0, beta: float, optional
        Custom parameter values (`params` must be set to None for this).
    
    """
    def __init__(self, 
                 params: str = 'zheng07', 
                 c0: float = None, 
                 beta: float = None, ) -> None:
        super().__init__()
        self._correction_mode = None
        # a pre-defined set of parameters: c0and beta
        self._parameters = {'zheng07'    : ( 11.0, -0.13 ), 
                            'bullock01'  : (  9.0, -0.13 ),
                            'shimizu03'  : (  8.0, -0.13 ), 
                            'abazajian05': ( 11.0, -0.05 ),}
        # set parameters
        self.use_set = params
        self.c0, self.beta = None, None
        if params not in self._parameters:
            assert c0 is not None and beta is not None
            self.c0, self.beta = c0, beta

    def _getParameters(self, 
                       model: Cosmology, 
                       m: Any, 
                       z: Any, 
                       overdensity: float | None, ) -> tuple:
        # Zheng et al (2007)
        if self.use_set == 'zheng07': return 11.0, -0.13 
        # original paper
        if self.use_set == 'bullock01': return 9.0, -0.13 
        # Shimizu et al (2003) 
        # NOTE: c0 valid for NFW (a = 1) profile only. multiply with 2-a for others
        if self.use_set == 'shimizu03': return 8.0, -0.13 
        # Abazajian, et al (2005)
        if self.use_set == 'abazajian05': 
            # effective slope at non-linear scale
            neff = model.matterPowerSpectrum( model.nonlinearWavenumber(z), z, deriv = 1 )
            c0   = 11.0 * ( model.Om0 / 0.3 )**-0.35 * ( neff / -1.7 )**-1.6 
            beta = -0.05
            return c0, beta
        # custom parameters: 
        return self.c0, self.beta

    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        m   = np.asfarray(m)
        zp1 = np.add(z, 1)
        # parameters
        c0, beta = self._getParameters(model, m, z, overdensity)
        # non-linear mass at z = 0
        mstar =  model.lagrangianM(model.collapseRadius(0.), overdensity) 
        # concentration
        res = c0 * (m / mstar)**beta / zp1 
        return res
    
class Diemer15(HaloConcentrationMassRelation):
    r"""
    Concentration mass relation given by Diemer & Kravtsov (2015).

    Parameters
    ----------
    params: str, default = `median_u`
        Specify which set of parameters are used (`median_u`, `mean_u`, `median` or `mean`). 
        Subscript `u` indicate updated values from Diemer & Joyce (2019). 

    """
    def __init__(self, params: str = 'median_u', ) -> None:
        super().__init__()
        # model parameters:
        # --------- Table 3 ----------: ( phi0, phi1, eta0, eta1, a,    b,    k     )
        self._parameters = {'median'  : ( 6.58, 1.37, 6.82, 1.42, 1.12, 1.69, 0.69, ), 
                            'mean'    : ( 7.14, 1.60, 4.10, 0.75, 1.40, 0.67, 0.69, ), 
                            'median_u': ( 6.58, 1.27, 7.28, 1.56, 1.08, 1.77, 1.00, ), 
                            'mean_u'  : ( 6.66, 1.37, 5.41, 1.06, 1.22, 1.22, 1.00, ),}
        self.use_set = params
        
    def call(self, 
             model: Cosmology, 
             m: Any, 
             z: Any, 
             overdensity: float | None = None, ) -> Any:
        # model parameters
        phi0, phi1, eta0, eta1, a, b, k = self._parameters[ self.use_set ]
        # halo size
        r = model.lagrangianR(m, overdensity)
        # local slope of power spectrum
        nloc = model.matterPowerSpectrum(2*np.pi*k/r, z, deriv = True)
        # concentration
        cmin = phi0 + phi1 * nloc 
        vmin = eta0 + eta1 * nloc
        res = model.nu(r, z) / vmin
        res = 0.5*cmin*( res**-a + res**b ) 
        return res

# initialising models to be readily used
bullock01 = Bullock01()
bullock01_powerlaw = Bullock01_PowerLaw()
diemer15 = Diemer15()

_available_models__ = {'bullock01_powerlaw': bullock01_powerlaw, 
                       'bullock01': bullock01, 
                       'diemer15' : diemer15, }

