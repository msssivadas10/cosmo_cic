module calculate_mfbf
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: PI, RHO_CRIT0
    use cosmology, only: cosmo_t
    implicit none

    private

    interface
        !! Interface to power spectrum calculator
        subroutine ps_calculate(cm, k, pk, args, stat)
            use iso_fortran_env, only: dp => real64
            use cosmology, only: cosmo_t
            !! inputs:
            class(cosmo_t), intent(in) :: cm
            real(dp), intent(in)       :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in), optional :: args(:) !! additional arguments
            !! outputs:
            real(dp), intent(out) :: pk
            integer , intent(out), optional :: stat
        end subroutine ps_calculate
    end interface

    interface
        !! Interface to mass function or bias calculator
        subroutine fs_calculate(cm, args, retval, stat)
            use iso_fortran_env, only: dp => real64
            use cosmology, only: cosmo_t
            !! Inputs
            class(cosmo_t), intent(in) :: cm !! cosmology parameters
            real(dp), intent(in) :: args(3)
            !! Outputs
            real(dp), intent(out) :: retval
            integer , intent(out), optional :: stat
        end subroutine fs_calculate
    end interface
    
contains

    !>
    !! Calculate the halo mass-function for a given mass and redshift.
    !!
    !! Parameters:
    !!  m     : real    - Mass in Msun
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity w.r.to mean background density.
    !!  cm    : cosmo_t - Cosmology parameters
    !!  dndlnm: real    - Calculated mass function in 1/Mpc^3
    !!  fs    : real    - Calculated mass function, f(sigma)
    !!  s     : real    - Calculated variance, sigma
    !!  dlns  : real    - Calculated log derivative of sigma w.r.to mass
    !!  stat  : integer - Status flag
    !! 
    subroutine calculate_massfunc(cm, mf, ps, m, z, Delta, dndlnm, fs, s, dlns, psnorm, psargs, stat)
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        procedure(fs_calculate)    :: mf !! mass function
        procedure(ps_calculate)    :: ps !! power spectrum model 
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean

        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out), optional :: fs, s, dlns  
        integer , intent(out), optional :: stat 

        real(dp) :: r_lag, rho_m, sigma, dlnsdlnm, fsigma
        integer  :: stat2 = 0

        if ( m < 0. )   stat2 = ERR_INVALID_VALUE_M
        if ( z <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            return
        end if

        if ( .not. ready ) then
            if ( present(stat) ) stat = ERR_CALC_NOT_SETUP 
            return
        end if

        !! universe density in Msun/Mpc^3 
        rho_m = cm%Omega_m * RHO_CRIT0_ASTRO * (0.01*cm%H0)**2 !! at z = 0 
        
        !! lagrangian radius corresponding to m (Mpc)
        r_lag = (0.75*m / PI / rho_m)**(1./3.) !! rho_m = rho_m * Delta ?

        !! calculate matter variance inside halo
        call vc(r_lag, z, cm, sigma, dlns = dlnsdlnm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        sigma    = sqrt(sigma) * cm%sigma8 !! actual normalization
        dlnsdlnm = dlnsdlnm / 6. 
        if ( present(s) ) s = sigma
        if ( present(dlns) ) dlns = dlnsdlnm

        !! calculate mass-function f(sigma)
        call mf(sigma, z, Delta, cm, fsigma, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        if ( present(fs) ) fs = fsigma

        !! calculate mass-function dn/dlnm in 1/Mpc^3 unit
        dndlnm = fsigma * abs(dlnsdlnm) * rho_m / m 
        
    end subroutine calculate_massfunc

    !>
    !! Calculate the halo bias function for a given mass and redshift.
    !!
    !! Parameters:
    !!  m     : real    - Mass in Msun
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity w.r.to mean background density.
    !!  cm    : cosmo_t - Cosmology parameters
    !!  bm    : real    - Calculated bias function
    !!  s     : real    - Calculated variance, sigma
    !!  stat  : integer - Status flag
    !! 
    subroutine calculate_bias(m, z, Delta, cm, bm, s, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        class(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: bm   !! halo bias
        real(dp), intent(out), optional :: s
        integer , intent(out), optional :: stat 

        real(dp) :: r_lag, rho_m, sigma, nu
        integer  :: stat2 = 0

        if ( m < 0. )   stat2 = ERR_INVALID_VALUE_M
        if ( z <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            return
        end if

        !! universe density in Msun/Mpc^3 
        rho_m = cm%Omega_m * RHO_CRIT0_ASTRO * (0.01*cm%H0)**2 !! at z = 0 
        
        !! lagrangian radius corresponding to m (Mpc)
        r_lag = (0.75*m / PI / rho_m)**(1./3.) !! rho_m = rho_m * Delta ?

        !! calculate matter variance inside halo
        !! calculate matter variance inside halo
        call vc(r_lag, z, cm, sigma, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        sigma = sqrt(sigma) * cm%sigma8 !! actual normalization
        if ( present(s) ) s = sigma

        !! calculate bias function b(sigma)
        nu = cm%get_collapse_density(z) / sigma
        call bf(nu, z, Delta, cm, bm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        
    end subroutine calculate_bias

    !>
    !! Calculate the both halo mass-function and bias for a given mass and redshift.
    !!
    !! Parameters:
    !!  m     : real    - Mass in Msun
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity w.r.to mean background density.
    !!  cm    : cosmo_t - Cosmology parameters
    !!  dndlnm: real    - Calculated mass function in 1/Mpc^3
    !!  fs    : real    - Calculated mass function, f(sigma)
    !!  bm    : real    - Calculated bias function
    !!  s     : real    - Calculated variance, sigma
    !!  dlns  : real    - Calculated log derivative of sigma w.r.to mass
    !!  stat  : integer - Status flag
    !! 
    subroutine calculate_massfunc_bias(m, z, Delta, cm, dndlnm, bm, fs, s, dlns, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out) :: bm     !! halo bias
        real(dp), intent(out), optional :: fs, s, dlns  
        integer , intent(out), optional :: stat 

        real(dp) :: sigma, nu
        integer  :: stat2 = 0

        !! calculating mass function
        call calculate_massfunc(m, z, Delta, cm, dndlnm, fs = fs, s = sigma, dlns = dlns, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        if ( present(s) ) s = sigma

        !! calculate bias function b(sigma)
        nu = cm%get_collapse_density(z) / sigma
        call bf(nu, z, Delta, cm, bm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        
    end subroutine calculate_massfunc_bias
    
end module calculate_mfbf