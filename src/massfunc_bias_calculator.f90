module massfunc_bias_calculator
    use constants, only: dp, PI, RHO_CRIT0_ASTRO
    use numerical, only: quadrule
    use objects, only: cosmology_model
    use variance_calculator, only: calculate_variance, tf_calculate
    implicit none
    
    interface
        !! Interface to mass function calculator
        subroutine fs_calculate(s, z, Delta, cm, retval, stat)
            use constants, only: dp
            use objects, only: cosmology_model
            real(dp), intent(in) :: s 
            real(dp), intent(in) :: z !! redshift
            real(dp), intent(in) :: Delta !! overdensity w.r.to mean
            type(cosmology_model), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: retval
            integer , intent(out) :: stat
        end subroutine fs_calculate
    end interface
contains

    subroutine calculate_massfunc(mf, m, z, Delta, cm, qrule_z, qrule_k, tf, stat, dndlnm, fs, s, dlns)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_z   !! integration rule for redshift
        type(quadrule), intent(in) :: qrule_k   !! integration rule for k 
        procedure(fs_calculate) :: mf !! halo mass-function model
        procedure(tf_calculate) :: tf !! power spectrum model

        integer , intent(out) :: stat 
        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out), optional :: fs, s, dlns  

        real(dp) :: r_lag, rho_m, sigma, dlnsdlnm, fsigma
        stat = 0

        !! universe density in Msun/Mpc^3 
        rho_m = cm%Omega_m * RHO_CRIT0_ASTRO * (0.01*cm%H0)**2 !! at z = 0 
        
        !! lagrangian radius corresponding to m (Mpc)
        r_lag = (0.75*m / PI / rho_m)**(1./3.) !! rho_m = rho_m * Delta ?

        !! calculate matter variance inside halo
        call calculate_variance(tf, r_lag, z, cm, qrule_k, qrule_z, sigma, dlns = dlnsdlnm)
        sigma    = sqrt(sigma) * cm%sigma8 !! actual normalization
        dlnsdlnm = dlnsdlnm / 6. 
        if ( present(s) ) then
            s = sigma
        end if
        if ( present(dlns) ) then
            dlns = dlnsdlnm
        end if 

        !! calculate mass-function f(sigma)
        call mf(sigma, z, Delta, cm, fsigma, stat)
        if ( stat .ne. 0 ) return !! return with error
        if ( present(fs) ) then
            fs = fsigma
        end if

        !! calculate mass-function dn/dlnm in 1/Mpc^3 unit
        dndlnm = fsigma * abs(dlnsdlnm) * rho_m / m 
        
    end subroutine calculate_massfunc

    subroutine calculate_bias(bf, m, z, Delta, cm, qrule_z, qrule_k, tf, stat, bm, s, dlns)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_z   !! integration rule for redshift
        type(quadrule), intent(in) :: qrule_k   !! integration rule for k 
        procedure(fs_calculate) :: bf !! halo bias function model
        procedure(tf_calculate) :: tf !! power spectrum model

        integer , intent(out) :: stat
        real(dp), intent(out) :: bm   !! halo bias
        real(dp), intent(out), optional :: s, dlns  

        real(dp) :: r_lag, rho_m, sigma
        stat = 0

        !! universe density in Msun/Mpc^3 
        rho_m = cm%Omega_m * RHO_CRIT0_ASTRO * (0.01*cm%H0)**2 !! at z = 0 
        
        !! lagrangian radius corresponding to m (Mpc)
        r_lag = (0.75*m / PI / rho_m)**(1./3.) !! rho_m = rho_m * Delta ?

        !! calculate matter variance inside halo
        call calculate_variance(tf, r_lag, z, cm, qrule_k, qrule_z, sigma, dlns = dlns)
        sigma    = sqrt(sigma) * cm%sigma8 !! actual normalization
        if ( present(s) ) then
            s = sigma
        end if
        if ( present(dlns) ) then
            dlns = dlns / 6.
        end if 

        !! calculate bias function b(sigma)
        call bf(sigma, z, Delta, cm, bm, stat)
        
    end subroutine calculate_bias

    subroutine calculate_massfunc_bias(mf, bf, m, z, Delta, cm, qrule_z, qrule_k, tf, stat, &
                    &                  dndlnm, fs, bm, s, dlns)
        real(dp), intent(in) :: m !! mass in Msun
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity w.r.to mean
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_z   !! integration rule for redshift
        type(quadrule), intent(in) :: qrule_k   !! integration rule for k 
        procedure(fs_calculate) :: mf !! halo mass-function model
        procedure(fs_calculate) :: bf !! halo bias function model
        procedure(tf_calculate) :: tf !! power spectrum model

        integer , intent(out) :: stat
        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out) :: bm     !! halo bias
        real(dp), intent(out), optional :: fs, s, dlns  
        real(dp) :: sigma  !! variance
        stat = 0

        !! calculating mass function
        call calculate_massfunc(mf, m, z, Delta, cm, qrule_z, qrule_k, tf, stat, dndlnm, &
                &               fs = fs, s = sigma, dlns = dlns)
        if ( stat .ne. 0) return !! return with error
        if ( present(s) ) then
            s = sigma
        end if

        !! calculate bias function b(sigma)
        call bf(sigma, z, Delta, cm, bm, stat)
        
    end subroutine calculate_massfunc_bias
    
end module massfunc_bias_calculator
