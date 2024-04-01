module massfunc_bias_calculator
    use constants, only: dp, PI, RHO_CRIT0_ASTRO
    use objects, only: cosmology_model
    use interfaces
    implicit none

    private

    public :: calculate_massfunc
    public :: calculate_bias
    public :: calculate_massfunc_bias
    
contains

    subroutine calculate_massfunc(mf, m, z, Delta, cm, ps_norm, ps, vc, &
                &                 dndlnm, fs, s, dlns, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        real(dp), intent(in) :: ps_norm !! power spectrum normalization
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        procedure(fs_calculate)  :: mf !! halo mass-function model
        procedure(ps_calculate)  :: ps !! power spectrum model
        procedure(var_calculate) :: vc !! variance calculation

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

        !! universe density in Msun/Mpc^3 
        rho_m = cm%Omega_m * RHO_CRIT0_ASTRO * (0.01*cm%H0)**2 !! at z = 0 
        
        !! lagrangian radius corresponding to m (Mpc)
        r_lag = (0.75*m / PI / rho_m)**(1./3.) !! rho_m = rho_m * Delta ?

        !! calculate matter variance inside halo
        call vc(ps, r_lag, z, cm, sigma, dlns = dlnsdlnm, stat = stat2)
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

    subroutine calculate_bias(bf, m, z, Delta, cm, ps_norm, ps, vc, bm, s, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        real(dp), intent(in) :: ps_norm !! power spectrum normalization
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        procedure(fs_calculate)  :: bf !! halo bias function model
        procedure(ps_calculate)  :: ps !! power spectrum model
        procedure(var_calculate) :: vc !! variance calculation

        real(dp), intent(out) :: bm   !! halo bias
        real(dp), intent(out), optional :: s
        integer , intent(out), optional :: stat 

        real(dp) :: r_lag, rho_m, sigma
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
        call vc(ps, r_lag, z, cm, sigma, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        sigma = sqrt(ps_norm*sigma) * cm%sigma8 !! actual normalization
        if ( present(s) ) s = sigma

        !! calculate bias function b(sigma)
        call bf(sigma, z, Delta, cm, bm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        
    end subroutine calculate_bias
    
    subroutine calculate_massfunc_bias(mf, bf, m, z, Delta, cm, ps_norm, ps, vc, &
                &                      dndlnm, bm, fs, s, dlns, stat)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        real(dp), intent(in) :: ps_norm !! power spectrum normalization
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        procedure(fs_calculate)  :: mf !! halo mass-function model
        procedure(fs_calculate)  :: bf !! halo bias function model
        procedure(ps_calculate)  :: ps !! power spectrum model
        procedure(var_calculate) :: vc !! variance calculation
        
        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out) :: bm     !! halo bias
        real(dp), intent(out), optional :: fs, s, dlns  
        integer , intent(out), optional :: stat 

        real(dp) :: sigma
        integer  :: stat2 = 0

        if ( m < 0. ) stat2 = 1   
        if ( z <= -1. ) stat2 = 1
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            return
        end if

        !! calculating mass function
        call calculate_massfunc(mf, m, z, Delta, cm, ps_norm, ps, vc, dndlnm, &
                &               fs = fs, s = sigma, dlns = dlns, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        if ( present(s) ) s = sigma

        !! calculate bias function b(sigma)
        call bf(sigma, z, Delta, cm, bm, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! return with error
        
    end subroutine calculate_massfunc_bias
    
end module massfunc_bias_calculator
