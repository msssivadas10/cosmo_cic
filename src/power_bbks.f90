module power_bbks
    use constants, only: dp
    use numerical, only: quadrule
    use objects, only: cosmology_model
    use growth_calculator, only: calculate_linear_growth
    use variance_calculator
    implicit none

    private

    real(dp) :: theta     !! CMB temperaure in 2.7K unit
    real(dp) :: dplus0    !! Growth factor scaling
    real(dp) :: Gamma_eff !! Shape parameter
    real(dp) :: NORM     = 1.0_dp !! Power spectrum normalization factor so that sigma^2(8 Mpc/h) = 1

    public :: tf_sugiyama95_calculate_params, tf_sugiyama95
    public :: set_filter
    public :: set_normalization, get_power_spectrum, get_variance
    
contains

    !>
    !! Calculate the quantities related to BBKS linear transfer function.
    !!
    !! Parameters:
    !!  cm: cosmology_model - Cosmology parameters.
    !!  qrule: quadrule - Integration rule for growth calculations
    !!  bbks : logical  - Tells if to use original BBKS or corrected version.
    !! 
    subroutine tf_sugiyama95_calculate_params(cm, qrule, bbks) 
        type(cosmology_model), intent(inout) :: cm !! cosmology parameters
        type(quadrule), intent(in)    :: qrule !! integration rule
        logical, intent(in), optional :: bbks 
        
        real(dp) :: Om0, Ob0, h      
        Om0   = cm%Omega_m
        Ob0   = cm%Omega_b
        h     = 0.01*cm%H0 !! hubble parameter in 100 km/sec/Mpc unit
        theta = cm%Tcmb0 / 2.7
        
        !! growth factor at z=0
        call calculate_linear_growth(0.0_dp, cm, qrule, dplus0)
        
        Gamma_eff = Om0*h**2 !! shape parameter
        
        if ( present(bbks) ) then
            if ( bbks ) return !! use original BBKS function
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
    !!  qrule: quadrule        - Integration rule (for growth factor calculations).
    !!  tk   : real            - Transfer function.
    !!  dlntk: real            - 1-st log derivative of transfer function (optional).
    !!
    !!
    subroutine tf_sugiyama95(k, z, cm, qrule, tk, dlntk)
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk

        real(dp) :: dplus, q, t0, t1, t2, t3, t4

        q = k / Gamma_eff !! dimensionless scale

        t0 = 2.34*q
        t1 = 3.89*q
        t2 = (16.1*q)**2
        t3 = (5.46*q)**3
        t4 = (6.71*q)**4
        
        !! BBKS transfer function
        tk = log(1. + t0) / t0 * ( 1. + t1 + t2 + t3 + t4 )**(-0.25)
        
        !! linear growth
        call calculate_linear_growth(z, cm, qrule, dplus)
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
    !!  qrule: quadrule        - Integration rule for growth calculations
    !!  pk   : real            - Value of calculated power spectrum (unit: Mpc^-3).
    !!  tk   : real            - Value of calculated transfer function (optional).
    !!  dlnpk: real            - Value of calculated log-derivative / effective index (optional).
    !! 
    subroutine get_power_spectrum(k, z, cm, qrule, pk, tk, dlnpk) 
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp), intent(out) :: pk
        real(dp), intent(out), optional :: tk, dlnpk

        real(dp) :: f, dlnf, ns
        ns = cm%ns

        !! transfer function
        if ( present(dlnpk) ) then
            call tf_sugiyama95(k, z, cm, qrule, f, dlntk = dlnf)

            !! effective index: 1-st log-derivative of p(k) w.r.to k
            dlnpk = ns*log(k) + 2*dlnf
        else
            call tf_sugiyama95(k, z, cm, qrule, f)
        end if
        if ( present(tk) ) then 
            tk = f
        end if

        !! power spectrum
        pk = NORM * k**ns * f**2
        
    end subroutine get_power_spectrum

    !>
    !! Calculate the smoothed linear variance of matter density. Calculated sigma^2 value is in
    !! units of `sigma8^2`.
    !!
    !! Parameters:
    !!  r      : real            - Smoothing scale in Mpc.
    !!  z      : real            - Redshift
    !!  cm     : cosmology_model - Cosmology parameters.
    !!  qrule_k: quadrule        - Integration rule for variance calculations (limits must be set).
    !!  qrule_z: quadrule        - Integration rule for growth calculations
    !!  sigma  : real            - Value of calculated variance (unit: Mpc^-3).
    !!  dlns   : real            - Value of calculated 1-st log-derivative (optional).
    !!  d2lns  : real            - Value of calculated 2-nd log-derivative (optional).
    !! 
    subroutine get_variance(r, z, cm, qrule_k, qrule_z, sigma, dlns, d2lns) 
        real(dp), intent(in) :: r !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_k, qrule_z !! integration rule
        
        real(dp), intent(out) :: sigma
        real(dp), intent(out), optional :: dlns, d2lns

        call calculate_variance(tf_sugiyama95, r, z, cm, qrule_k, qrule_z, sigma, dlns, d2lns)    
        sigma = NORM * sigma !! normalization
        
    end subroutine get_variance

    !>
    !! Calculate sigma-8 normalization.
    !!
    !! Parameters:
    !!  cm     : cosmology_model - Cosmology parameters.
    !!  qrule_k: quadrule        - Integration rule for variance calculations (limits must be set).
    !!  qrule_z: quadrule        - Integration rule for growth calculations
    !! 
    subroutine set_normalization(cm, qrule_k, qrule_z)
        type(cosmology_model), intent(inout) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_k, qrule_z !! integration rule

        call calculate_sigma8_normalization(tf_sugiyama95, cm, qrule_k, qrule_z, NORM)
        
    end subroutine set_normalization
    
end module power_bbks
