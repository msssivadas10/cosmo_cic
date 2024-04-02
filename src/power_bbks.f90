!!
!! Linear matter power spectrum model by BBKS.
!!
module power_bbks
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmology_model
    use growth_calculator, only: calculate_linear_growth
    use variance_calculator, only: calculate_variance
    implicit none

    private

    real(dp) :: theta         !! CMB temperaure in 2.7K unit
    real(dp) :: dplus0        !! Growth factor scaling
    real(dp) :: Gamma_eff     !! Shape parameter
    real(dp) :: NORM = 1.0_dp !! Power spectrum normalization factor so that sigma^2(8 Mpc/h) = 1

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_K  = 40 !! invalid value for wavenumber

    public :: tf_sugiyama95_calculate_params, tf_sugiyama95
    public :: get_power_spectrum, get_power_unnorm
    public :: get_variance, set_normalization, get_normalization
    
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
        type(cosmology_model), intent(inout) :: cm !! cosmology parameters
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

    !====================================================================================================!

    !>
    !! Calculate the linear matter power spectrum. Scale the calculated power spectrum value 
    !! by `sigma8^2` to get the actual normalised power spectrum.
    !!
    !! Parameters:
    !!  k    : real            - Wavenumber in 1/Mpc.
    !!  z    : real            - Redshift
    !!  cm   : cosmology_model - Cosmology parameters.
    !!  pk   : real            - Value of calculated power spectrum (unit: Mpc^-3).
    !!  tk   : real            - Value of calculated transfer function (optional).
    !!  dlnpk: real            - Value of calculated log-derivative / effective index (optional).
    !!  stat : integer         - Status flag. Non-zero for failure.
    !! 
    subroutine get_power_spectrum(k, z, cm, pk, tk, dlnpk, stat) 
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: pk
        real(dp), intent(out), optional :: tk, dlnpk
        integer , intent(out), optional :: stat

        real(dp) :: f, ns
        integer  :: stat2 = 0
        ns = cm%ns

        !! transfer function
        call tf_sugiyama95(k, z, cm, f, dlntk = dlnpk, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return
        if ( present(tk) ) tk = f

        !! effective index: 1-st log-derivative of p(k) w.r.to k
        if ( present(dlnpk) ) dlnpk = ns*log(k) + 2*dlnpk

        !! power spectrum, normalised so that sigma^2(8 Mpc/h) = 1  
        pk = NORM * k**ns * f**2
        
    end subroutine get_power_spectrum

    !>
    !! Calculate the linear matter power spectrum, without normalization.
    !!
    !! Parameters:
    !!  k    : real            - Wavenumber in 1/Mpc.
    !!  z    : real            - Redshift
    !!  cm   : cosmology_model - Cosmology parameters.
    !!  pk   : real            - Value of calculated power spectrum (unit: Mpc^-3).
    !!  stat : integer         - Status flag. Non-zero for failure.
    !! 
    subroutine get_power_unnorm(k, z, cm, pk, stat) 
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: pk
        integer , intent(out), optional :: stat

        integer  :: stat2 = 0
        real(dp) :: ns
        ns = cm%ns

        !! transfer function
        call tf_sugiyama95(k, z, cm, pk, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! power spectrum
        pk = k**ns * pk**2
        
    end subroutine get_power_unnorm

    !>
    !! Calculate the smoothed linear variance of matter density. Calculated sigma^2 value is in
    !! units of `sigma8^2`.
    !!
    !! Parameters:
    !!  r    : real            - Smoothing scale in Mpc.
    !!  z    : real            - Redshift
    !!  cm   : cosmology_model - Cosmology parameters.
    !!  sigma: real            - Value of calculated variance (unit: Mpc^-3).
    !!  dlns : real            - Value of calculated 1-st log-derivative (optional).
    !!  d2lns: real            - Value of calculated 2-nd log-derivative (optional).
    !!  stat : integer         - Status flag. Non-zero for failure.
    !! 
    subroutine get_variance(r, z, cm, sigma, dlns, d2lns, stat) 
        real(dp), intent(in) :: r !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: sigma
        real(dp), intent(out), optional :: dlns, d2lns
        integer , intent(out), optional :: stat 
        integer :: stat2 = 0

        !! calculate variance
        call calculate_variance(get_power_unnorm, r, z, cm, sigma, dlns = dlns, d2lns = d2lns, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! normalization
        sigma = NORM * sigma 
        
    end subroutine get_variance

    !>
    !! Calculate sigma-8 normalization.
    !!
    !! Parameters:
    !!  cm  : cosmology_model - Cosmology parameters.
    !!  stat: integer         - Status flag. Non-zero for failure.
    !! 
    subroutine set_normalization(cm, stat)
        type(cosmology_model), intent(inout) :: cm !! cosmology parameters
        integer , intent(out), optional :: stat 

        real(dp) :: calculated, r, z
        integer  :: stat2 = 0
        r = 8.0 / (0.01 * cm%H0) !! = 8 Mpc/h
        z = 0._dp

        !! calculating variance at 8 Mpc/h
        call calculate_variance(get_power_unnorm, r, z, cm, calculated, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! normalization
        NORM = 1. / calculated
        
    end subroutine set_normalization

    !>
    !! Get the current normalization factor.
    !!
    function get_normalization() result(retval)
        real(dp) :: retval
        retval = NORM
    end function get_normalization
    
end module power_bbks
