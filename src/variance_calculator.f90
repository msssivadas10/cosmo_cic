module variance_calculator
    use constants, only: dp, PI
    use numerical, only: quadrule
    use objects, only: cosmology_model
    implicit none

    !!
    !! Window function models for varience calculations
    !!
    integer, parameter, private :: WIN_TOPHAT = 4001 !! Spherical top-hat window
    integer, parameter, private :: WIN_GAUSS  = 4002 !! Gaussian window
    integer, parameter, private :: WIN_SHARPK = 4003 !! Sharp-k window

    integer, private :: use_filter = WIN_TOPHAT !! Filter to use

    public :: calculate_variance, calculate_sigma8_normalization, set_filter

    interface
        !! Interface to transfer function calculator
        subroutine tf_calculate(k, z, cm, qrule, tk, dlntk)
            use constants, only: dp
            use numerical, only: quadrule
            use objects, only: cosmology_model
            real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in) :: z !! redshift
            type(cosmology_model), intent(in) :: cm !! cosmology parameters
            type(quadrule), intent(in) :: qrule     !! integration rule
            real(dp), intent(out) :: tk
            real(dp), intent(out), optional :: dlntk
        end subroutine tf_calculate
    end interface
    
contains
    
    !>
    !! Set the filter function for smoothing: gaussian (`gauss`) or spherical tophat (`tophat`).
    !!
    subroutine set_filter(id)
        character(len = 6), intent(in) :: id

        select case ( id )
        case ( 'gauss' ) !! use gaussian filter
            use_filter = WIN_GAUSS
        ! case ( 'sharpk' ) !! use sharp-k filter
        !     use_filter = WIN_SHARPK
        case ( 'tophat' ) !! use tophat filter
            use_filter = WIN_TOPHAT
        case default 
            use_filter = WIN_TOPHAT
        end select

    end subroutine set_filter

    !>
    !! Calculate the matter variance by smoothing over a scale r Mpc.
    !!
    !! Parameters:
    !!  tf     : procedure       - Transfer function
    !!  r      : real            - Smoothing scale in Mpc
    !!  z      : real            - Redshift
    !!  cm     : cosmology_model - Cosmology prameters
    !!  qrule_k: quadrule        - Integration rule for k-space. Limits must be specified here.
    !!  qrule_r: quadrule        - Integration rule for redshift (in growth calculation)
    !!  sigma  : real            - Calculated variance
    !!  dlns   : real            - Calculatetd 1-st log-derivative (optional)
    !!  d2lns  : real            - Calculatetd 2-nd log-derivative (optional)
    !!
    subroutine calculate_variance(tf, r, z, cm, qrule_k, qrule_z, sigma, dlns, d2lns)
        procedure(tf_calculate) :: tf !! transfer function 
        real(dp), intent(in) :: r !! scale in Mpc
        real(dp), intent(in) :: z !! redshift
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_z   !! integration rule for redshift
        type(quadrule), intent(in) :: qrule_k   !! integration rule for k

        real(dp), intent(out) :: sigma !! variance 
        real(dp), intent(out), optional :: dlns, d2lns 

        real(dp) :: k_shift, k_scale, lnka, lnkb
        real(dp) :: k, kr, f2, f3, res1, res2, res3, const, ns, wk, pk
        integer  :: i, max_deriv
        
        !! calculating integration nodes transformation
        lnka    = log(1.0e-04_dp)     !! lower limit is fixed at k = 1e-4
        lnkb    = log(1.0e+04_dp / r) !! upper limit is variable, but set kr = 1e+4, where w(kr) ~ 0
        k_scale = 0.5*( lnkb - lnka )
        k_shift = lnka + k_scale
        
        max_deriv = 0
        if ( present(dlns) ) then 
            max_deriv = 1
        end if
        if ( present(d2lns) ) then
            max_deriv = 2
        end if
        
        const  = k_scale / (2*PI**2)
        ns     = cm%ns
        
        res1 = 0.0_dp
        res2 = 0.0_dp
        res3 = 0.0_dp
        do i = 1, qrule_k%n

            k  = exp( k_scale * qrule_k%x(i) + k_shift ) !! wavenumber in 1/Mpc
            wk = qrule_k%w(i) 
            kr = k*r 

            !! calculating power spectrum
            call tf(k, z, cm, qrule_z, pk)
            pk = k**(ns + 3.) * pk**2

            !! calculating window function, w(kr)
            if ( use_filter == WIN_GAUSS ) then !! gaussian window function
                f3 = exp(-0.5*kr**2)
            else !! spherical tophat window function
                f3 = 3*( sin(kr) - kr*cos(kr) ) / kr**3
            end if
            
            f2   = pk * f3               !! p(k)*w(kr) 
            res1 = res1 + wk * ( f2*f3 ) !! sigma^2 integration

            if ( max_deriv == 0 ) cycle !! no need to calculate derivatives!...

            !! calculating window function 1-st derivative, dwdx
            if ( use_filter == WIN_GAUSS ) then !! gaussian window function
                f3 = -kr*exp(-0.5*kr**2)
            else !! spherical tophat window function
                f3 = 3*( ( kr**2 - 3. )*sin(kr) + 3*kr*cos(kr) ) / kr**4
            end if
            
            f3   = 2*f3 * k              !! 2*dwdx*k := 2*dwdr
            res2 = res2 + wk * ( f2*f3 ) !! ds2dr integration

            if ( max_deriv == 1 ) cycle !! no need to calculate 2-nd derivative!...

            !! NOTE: check this implementation!... :)

            res3 = res3 + wk * ( 0.5*pk*f3**2 ) !! d2s2dr2 integration part:1

            !! calculating window function 2-nd derivative
            if ( use_filter == WIN_GAUSS ) then !! gaussian window function
                f3 = (kr**2 - 1.)*exp(-0.5*kr**2)
            else !! spherical tophat window function
                f3 = 3*( ( kr**2 - 12. )*kr*cos(kr) + ( 12 - 5*kr**2 )*sin(kr) ) / kr**5
            end if

            res3 = res3 + wk * ( 2*f2*f3*k**2 ) !! d2s2dr2 integration part: 2
            
        end do

        sigma = const * res1
        if ( max_deriv == 0 ) return
        
        !! 1-st log-derivative, dlnsdlnr
        res1 = r / res1
        res2 = res1 * res2 
        if ( present(dlns) ) then 
            dlns = res2
        end if
        
        !! 2-nd log-derivative, d2lnsdlnr2
        if ( present(d2lns) ) then 
            d2lns = res1 * res3 * r + res2 * ( 1. - res2 )
        end if

    end subroutine calculate_variance

    !>
    !! Normalize the matter power spectrum using sigma8 value.
    !!
    !! Parameters:
    !!  tf     : procedure       - Transfer function
    !!  cm     : cosmology_model - Cosmology prameters
    !!  qrule_k: quadrule        - Integration rule for k-space. Limits must be specified here.
    !!  qrule_r: quadrule        - Integration rule for redshift (in growth calculation)
    !!  norm   : real            - Calculated value of the normalization factor.
    !!
    subroutine calculate_sigma8_normalization(tf, cm, qrule_k, qrule_z, norm)
        procedure(tf_calculate) :: tf !! transfer function 
        type(cosmology_model), intent(inout) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule_z   !! integration rule for redshift
        type(quadrule), intent(in) :: qrule_k   !! integration rule for k
        
        real(dp), intent(out) :: norm !! normalization factor
        
        real(dp) :: calculated, r
        r = 8.0 / (0.01 * cm%H0) !! = 8 Mpc/h
        cm%ps_norm = 1.0_dp
    
        !! calculating variance at 8 Mpc/h
        call calculate_variance(tf, r, 0.0_dp, cm, qrule_k, qrule_z, calculated)
        
        !! normalization
        norm = 1. / calculated
        
    end subroutine calculate_sigma8_normalization
    
end module variance_calculator
