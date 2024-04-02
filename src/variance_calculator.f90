!!
!! Generalized variance calculations for a power spectrum model.
!!
module variance_calculator
    use iso_fortran_env, only: dp => real64
    use constants, only: PI
    use numerical, only: generate_gaussleg
    use objects, only: cosmo_t
    implicit none

    private

    interface
        !! Interface to power spectrum calculator
        subroutine ps_calculate(k, cm, pk, args, stat)
            use iso_fortran_env, only: dp => real64
            ! use constants, only: dp
            use objects, only: cosmo_t
            real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
            type(cosmo_t), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: pk
            real(dp), intent(in), optional :: args(:) !! additional arguments
            integer , intent(out), optional :: stat
        end subroutine ps_calculate
    end interface

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_R  = 30 !! invalid value for radius
    integer, parameter :: ERR_CALC_NOT_SETUP   = 31 !! calculator not set up

    !! Window function models for varience calculations
    integer, parameter :: WIN_TOPHAT = 4001 !! Spherical top-hat window
    integer, parameter :: WIN_GAUSS  = 4002 !! Gaussian window
    integer, parameter :: WIN_SHARPK = 4003 !! Sharp-k window

    integer :: use_filter = WIN_TOPHAT !! Filter to use
    integer  :: nq = 0             !! number of points for integration
    real(dp), allocatable :: xq(:) !! nodes for integration
    real(dp), allocatable :: wq(:) !! weights for integration
    logical :: ready = .false.

    public :: set_filter, setup_variance_calculator
    public :: calculate_variance
    
contains
    
    !>
    !! Set the filter function for smoothing: gaussian (`gauss`) or spherical tophat (`tophat`).
    !!
    !! Parameters:
    !!  filt: character - String id of the filter.
    !!
    !! Filter ids:
    !!  `gauss`  - Gaussian
    !!  `tophat` - Spherical top-hat
    !!  `sharpk` - Sharp-k filter (not using)
    !!
    subroutine set_filter(filt)
        character(len = 6), intent(in) :: filt

        select case ( filt )
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
    !! Setup the variance calculator.
    !!
    !! Parameters:
    !!  n   : integer   - Size of the integration rule.
    !!  stat: integer   - Status variable. 0 for success.
    !!  filt: character - String id of the filter.
    !!
    subroutine setup_variance_calculator(n, stat, filt)
        integer, intent(in)  :: n
        integer, intent(out) ::  stat
        character(len = 6), intent(in), optional :: filt

        !! allocate node array
        if ( .not. allocated(xq) ) allocate( xq(n) )
        
        !! allocate weights array
        if ( .not. allocated(wq) ) allocate( wq(n) )
        
        !! generating integration rule...
        nq = n
        call generate_gaussleg(n, xq, wq, stat = stat)
        if ( stat .ne. 0 ) return !! failed to generate integration rule 

        if ( present(filt) ) call set_filter( filt ) !! setting filter 

        ready = .true. !! ready for calculations
        
    end subroutine setup_variance_calculator
    
    !>
    !! Reset variance calculator to initial state. 
    !!
    subroutine reset_variance_calculator()
        deallocate( xq )
        deallocate( wq )
        nq = 0
        use_filter = WIN_TOPHAT
        ready      = .false. 
    end subroutine reset_variance_calculator

    !>
    !! Calculate the matter variance by smoothing over a scale r Mpc.
    !!
    !! Parameters:
    !!  tf     : procedure - Transfer function
    !!  r      : real      - Smoothing scale in Mpc
    !!  z      : real      - Redshift
    !!  cm     : cosmo_t   - Cosmology prameters
    !!  sigma  : real      - Calculated variance
    !!  dlns   : real      - Calculatetd 1-st log-derivative (optional)
    !!  d2lns  : real      - Calculatetd 2-nd log-derivative (optional)
    !!
    subroutine calculate_variance(ps, r, z, cm, sigma, dlns, d2lns, args, stat)
        procedure(ps_calculate) :: ps !! power spectrum
        real(dp), intent(in) :: r !! scale in Mpc
        real(dp), intent(in) :: z !! redshift
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: sigma !! variance 
        real(dp), intent(out), optional :: dlns, d2lns 
        real(dp), intent(in), optional :: args(:) !! additional arguments
        integer , intent(out), optional :: stat

        real(dp) :: k_shift, k_scale, lnka, lnkb
        real(dp) :: k, kr, f2, f3, res1, res2, res3, const, ns, wk, pk
        integer  :: i, max_deriv, stat2
        stat2 = 0

        if ( z <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( r <=  0. ) stat2 = ERR_INVALID_VALUE_R
        if ( .not. ready ) stat2 = ERR_CALC_NOT_SETUP
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return
        
        !! calculating integration nodes transformation
        lnka    = log(1.0e-04_dp)     !! lower limit is fixed at k = 1e-4
        lnkb    = log(1.0e+04_dp / r) !! upper limit is variable, but set kr = 1e+4, where w(kr) ~ 0
        k_scale = 0.5*( lnkb - lnka )
        k_shift = lnka + k_scale
        
        !! get how many derivatives to calculate
        max_deriv = 0
        if ( present(dlns)  ) max_deriv = 1
        if ( present(d2lns) ) max_deriv = 2

        !! status variable
        if ( present(stat) ) stat = 0 !! success
        
        const = k_scale / (2*PI**2)
        ns    = cm%ns
        stat2 = 0
        res1  = 0.0_dp
        res2  = 0.0_dp
        res3  = 0.0_dp
        do i = 1, nq

            k  = exp( k_scale * xq(i) + k_shift ) !! wavenumber in 1/Mpc
            wk = wq(i) 
            kr = k*r 

            !! calculating power spectrum
            call ps(k, cm, pk, args = args, stat = stat2)
            if ( stat2 .ne. 0 ) exit !! error in power spectrum calculation
            pk = k**3 * pk

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

        !! status variable
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return !! failure

        sigma = const * res1
        if ( max_deriv == 0 ) return
        
        !! 1-st log-derivative, dlnsdlnr
        res1 = r / res1
        res2 = res1 * res2 
        if ( present(dlns) ) dlns = res2
        
        !! 2-nd log-derivative, d2lnsdlnr2
        if ( present(d2lns) ) d2lns = res1 * res3 * r + res2 * ( 1. - res2 )

    end subroutine calculate_variance
    
end module variance_calculator
