!!
!! Some constants...
!!
module constants
    use iso_fortran_env, only: dp => real64
    implicit none

    real(dp), parameter :: EPS = 1.0e-08 !! tolerance 

    real(dp), parameter :: PI     = 3.141592653589793_dp !! pi
    real(dp), parameter :: SQRT_2 = 1.4142135623730951_dp    

    ! Values related to cosmology
    real(dp), parameter :: DELTA_SC        = 1.6864701998411453_dp !! Overdensity threshold for collapse
    real(dp), parameter :: RHO_CRIT0_ASTRO = 2.77536627E+11_dp     !! Critical density in h^2 Msun / Mpc^3        
    real(dp), parameter :: SPEED_OF_LIGHT_KMPS = 299792.458_dp     !! Speed of light in km/sec  

    ! Units of mass
    real(dp), parameter :: GMSUN = 1.32712440018e+20_dp !! GM for sun in m^3/s^2     
    real(dp), parameter :: MSUN  = 1.98842e+30_dp       !! Mass of sun in kg  
    
    ! Distance units
    real(dp), parameter :: AU  = 1.49597870700e+11_dp     !! 1 astronomical unit (au) in m
    real(dp), parameter :: MPC = 3.085677581491367e+22_dp !! 1 mega parsec (Mpc) 

    ! Time units
    real(dp), parameter :: YEAR = 31558149.8_dp !! Seconds in a sidereal year            

end module constants