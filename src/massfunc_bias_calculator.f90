module massfunc_bias_calculator
    use iso_fortran_env, only: dp => real64
    use constants, only: PI, RHO_CRIT0_ASTRO
    use objects, only: cosmology_model
    implicit none

    private

    interface
        !! Interface to variance calculator
        subroutine var_calculate(r, z, cm, sigma, dlns, d2lns, stat)
            use iso_fortran_env, only: dp => real64
            ! use constants, only: dp
            use objects, only: cosmology_model
            real(dp), intent(in) :: r !! scale in Mpc
            real(dp), intent(in) :: z !! redshift
            type(cosmology_model), intent(in) :: cm !! cosmology parameters

            real(dp), intent(out) :: sigma !! variance 
            real(dp), intent(out), optional :: dlns, d2lns 
            integer , intent(out), optional :: stat
        end subroutine var_calculate
    end interface

    interface
        !! Interface to mass function or bias calculator
        subroutine fs_calculate(s, z, Delta, cm, retval, stat)
            use iso_fortran_env, only: dp => real64
            ! use constants, only: dp
            use objects, only: cosmology_model
            real(dp), intent(in) :: s 
            real(dp), intent(in) :: z !! redshift
            real(dp), intent(in) :: Delta !! overdensity w.r.to mean
            type(cosmology_model), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: retval
            integer , intent(out), optional :: stat
        end subroutine fs_calculate
    end interface

    procedure(var_calculate), pointer :: vc => null() !! variance calculation
    procedure(fs_calculate) , pointer :: mf => null() !! halo mass-function model
    procedure(fs_calculate) , pointer :: bf => null() !! halo bias function model

    real(dp) :: ps_norm = 1.0_dp   !! power spectrum normalization
    logical  :: has_mf     = .false. 
    logical  :: has_bf     = .false. 
    logical  :: has_setup  = .false. 

    public :: calculate_massfunc
    public :: calculate_bias
    public :: calculate_massfunc_bias
    
contains

    subroutine setup_massfunc_bias_calculator(vc1, ps_norm1, stat)
        procedure(var_calculate) :: vc1 !! variance calculation
        real(dp), intent(in)     :: ps_norm1 !! power spectrum normalization
        integer , intent(out):: stat

        vc => vc1
        ps_norm = ps_norm1
        has_setup = .true.

        stat = 0
    end subroutine setup_massfunc_bias_calculator

    subroutine set_massfunc(stat, mf1, bf1)
        procedure(fs_calculate), optional :: mf1 !! halo mass-function model
        procedure(fs_calculate), optional :: bf1 !! halo bias function model
        integer , intent(out) :: stat

        stat = 1

        if ( present(mf1) ) then !! set mass function model
            mf => mf1
            has_mf = .true.
            stat   = 0
        end if 

        if ( present(bf1) ) then !! set bias model
            bf => bf1
            has_bf = .true.
            stat   = 0
        end if 
        
    end subroutine set_massfunc

    subroutine calculate_massfunc(m, z, Delta, cm, dndlnm, fs, s, dlns, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out), optional :: fs, s, dlns  
        integer , intent(out), optional :: stat 

        real(dp) :: r_lag, rho_m, sigma, dlnsdlnm, fsigma
        integer  :: stat2 = 0

        if ( m < 0. ) stat2 = 1
        if ( z <= -1. ) stat2 = 1
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            return
        end if

        if ( .not. has_setup ) then
            if ( present(stat) ) stat = 1
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

        sigma    = sqrt(ps_norm*sigma) * cm%sigma8 !! actual normalization
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

    subroutine calculate_bias(m, z, Delta, cm, bm, s, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: bm   !! halo bias
        real(dp), intent(out), optional :: s
        integer , intent(out), optional :: stat 

        real(dp) :: r_lag, rho_m, sigma, nu
        integer  :: stat2 = 0

        if ( m < 0. ) stat2 = 1
        if ( z <= -1. ) stat2 = 1
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

        sigma = sqrt(ps_norm*sigma) * cm%sigma8 !! actual normalization
        if ( present(s) ) s = sigma

        !! calculate bias function b(sigma)
        nu = cm%get_collapse_density(z) / sigma
        call bf(nu, z, Delta, cm, bm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        
    end subroutine calculate_bias
    
    subroutine calculate_massfunc_bias(m, z, Delta, cm, dndlnm, bm, fs, s, dlns, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out) :: bm     !! halo bias
        real(dp), intent(out), optional :: fs, s, dlns  
        integer , intent(out), optional :: stat 

        real(dp) :: sigma, nu
        integer  :: stat2 = 0

        if ( m < 0. ) stat2 = 1   
        if ( z <= -1. ) stat2 = 1
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            return
        end if

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
    
end module massfunc_bias_calculator
