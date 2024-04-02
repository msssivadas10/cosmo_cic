!!
!! Linear matter power spectrum model by BBKS.
!!
module transfer_bbks
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmology_model
    use growth_calculator, only: calculate_linear_growth
    use variance_calculator, only: calculate_variance
    implicit none

    private

    real(dp) :: theta     !! CMB temperaure in 2.7K unit
    real(dp) :: dplus0    !! Growth factor scaling
    real(dp) :: Gamma_eff !! Shape parameter

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_K  = 40 !! invalid value for wavenumber

    public :: tf_sugiyama95_calculate_params, tf_sugiyama95
    
contains

    !>
    !! Calculate the quantities related to BBKS linear transfer function.
    !!
    !! Parameters:
    !!  cm      : cosmology_model - Cosmology parameters.
    !!  use_bbks: logical         - Tells if to use original BBKS or corrected version.
    !!  stat    : integer         - Status flag. Non-zero for failure.
    !! 
    subroutine tf_sugiyama95_calculate_params(cm, use_bbks, stat) 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        logical, intent(in) , optional :: use_bbks 
        integer, intent(out), optional :: stat
        
        real(dp) :: Om0, Ob0, h      
        integer  :: stat2 = 0
        Om0   = cm%Omega_m
        Ob0   = cm%Omega_b
        h     = 0.01*cm%H0 !! hubble parameter in 100 km/sec/Mpc unit
        theta = cm%Tcmb0 / 2.7
        
        !! growth factor at z=0
        call calculate_linear_growth(0.0_dp, cm, dplus0, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return
        
        Gamma_eff = Om0*h**2 !! shape parameter
        
        if ( present(use_bbks) ) then
            if ( use_bbks ) return !! use original BBKS function
        end if
        
        !! apply baryon correction factor
        Gamma_eff = Gamma_eff * exp(-Ob0 - sqrt(2*h)*Ob0/Om0)
        
    end subroutine tf_sugiyama95_calculate_params

    !>
    !! Transfer function model given by Bardeen et al (1986), including correction 
    !! by Sugiyama (1995).  BBKS.
    !!
    !! Parameters:
    !!  k    : real            - Wavenumebr in 1/Mpc.
    !!  z    : real            - Redshift.
    !!  cm   : cosmology_model - Cosmology model parameters.
    !!  tk   : real            - Transfer function.
    !!  dlntk: real            - 1-st log derivative of transfer function (optional).
    !!  stat : integer         - Status flag. Non-zero for failure.
    !!
    subroutine tf_sugiyama95(k, z, cm, tk, dlntk, stat)
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        real(dp) :: dplus, q, t0, t1, t2, t3, t4
        integer  :: stat2 = 0

        if ( z <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( k <=  0. ) stat2 = ERR_INVALID_VALUE_K
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        q = k / Gamma_eff !! dimensionless scale

        t0 = 2.34*q
        t1 = 3.89*q
        t2 = (16.1*q)**2
        t3 = (5.46*q)**3
        t4 = (6.71*q)**4
        
        !! BBKS transfer function
        tk = log(1. + t0) / t0 * ( 1. + t1 + t2 + t3 + t4 )**(-0.25)
        
        !! linear growth
        call calculate_linear_growth(z, cm, dplus, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return
        dplus = dplus / dplus0 !! normalization
        tk = dplus * tk

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            t0    = t0 / ( (1. + t0) * log(1. + t0) )
            t1    = 0.25 * ( t1 + 2*t2 + 3*t3 + 4*t4 ) / (1 + t1 + t2 + t3 + t4) + 1.
            dlntk = t0 - t1
        end if
        
    end subroutine tf_sugiyama95

end module transfer_bbks
