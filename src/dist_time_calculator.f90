!!
!! Distance and time calculations
!!
module dist_time_calculator
    use iso_fortran_env, only: dp => real64
    use constants, only: PI, SPEED_OF_LIGHT_KMPS, EPS
    use numerical, only: generate_gaussleg
    use objects, only: cosmology_model
    implicit none

    private 

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_CALC_NOT_SETUP   = 21 !! calculator not setup
    logical :: ready = .false.

    integer  :: nq = 0 !! number of points for integration
    real(dp), allocatable :: xq(:) !! nodes for integration
    real(dp), allocatable :: wq(:) !! weights for integration

    public :: setup_distance_calculator, reset_distance_calculator
    public :: calculate_comoving_distance
    public :: get_comoving_coordinate
    public :: get_luminocity_distance
    public :: get_angular_diameter_distance
    public :: get_physical_size, get_angular_size

contains

    !>
    !! Setup the growth calculator.
    !!
    !! Parameters:
    !!  n   : integer - Size of the integration rule.
    !!  stat: integer - Status variable. 0 for success.
    !!
    subroutine setup_distance_calculator(n, stat)
        integer, intent(in) :: n
        integer, intent(out) ::  stat

        !! allocate node array
        if ( .not. allocated(xq) ) then
            allocate( xq(n) )
        end if
        
        !! allocate weights array
        if ( .not. allocated(wq) ) then
            allocate( wq(n) )
        end if
        
        !! generating integration rule...
        call generate_gaussleg(n, xq, wq, stat = stat)
        nq    = n
        ready = .true.
        
    end subroutine setup_distance_calculator

    !>
    !! Reset growth calculator to initial state. 
    !!
    subroutine reset_distance_calculator()
        deallocate( xq )
        deallocate( wq )
        nq    = 0
        ready = .false.
    end subroutine reset_distance_calculator

    !>
    !! Calculate comoving distance at redshift z.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  r    : real            - Comoving distance in Mpc.
    !!  dvdz : real            - Comoving volume element in Mpc^3.
    !!  stat : integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine calculate_comoving_distance(z, cm, r, dvdz, stat) 
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: r !! distance in Mpc
        real(dp), intent(out), optional :: dvdz !! volume element in Mpc^3
        integer , intent(out), optional :: stat !! integration rule

        real(dp) :: retval, drdz
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: a, fa, x_scale
        integer  :: i
        H0 = cm%H0

        if ( .not. ready ) then !! not properly setup
            if ( present(stat) ) stat = ERR_CALC_NOT_SETUP
            return
        end if

        if ( z <= -1. ) then !! invalid value for redshift
            if ( present(stat) ) stat = ERR_INVALID_VALUE_Z
            return
        end if

        r = 0.0_dp; dvdz = 0.0_dp
        if ( abs( z ) < EPS ) return !! z = 0

        !! ditance by evaluating the integral of [ a^2 * E(a) ]^-1 from a to 1...
        x_scale = 0.5_dp * z / ( z + 1._dp ) 
        do i = 1, nq
            !! scale factor
            a = x_scale * xq(i) + (1. - x_scale) 
            
            !! calculating integrand
            call cm%calculate_hubble_func(1/a-1., fa) !! E**2(z)
            fa = 1./ a**2 / sqrt( fa )
            
            retval = retval + wq(i) * fa
        end do
        retval = retval * x_scale
        
        !! comoving distance in Mpc
        r = retval * SPEED_OF_LIGHT_KMPS / H0

        if ( present(dvdz) ) then !! calculate comoving volume element, dvdz
            
            !! distance derivative w.r.to z: 1/E(a)
            call cm%calculate_hubble_func(z, drdz)
            drdz = 1.0 / sqrt( drdz ) * SPEED_OF_LIGHT_KMPS / H0

            !! volume element
            dvdz = 4*PI * r**2 * drdz

        end if

        if ( present(stat) ) then !! succesful!
            stat = 0
        end if
        
    end subroutine calculate_comoving_distance

    !!====================================================================================

    !>
    !! Calculate comoving coordinate at redshift z.
    !!
    !! Parameters:
    !!  z   : real            - Redshift (must be greater than -1).
    !!  cm  : cosmology_model - Cosmology parameters
    !!  r   : real            - Comoving coordinate in Mpc.
    !!  stat: integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine get_comoving_coordinate(z, cm, r, stat)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: r !! value of coordinate in Mpc
        integer , intent(out), optional :: stat

        real(dp) :: Ok0 !! density parameters at z=0
        real(dp) :: k !! curvature
        Ok0  = cm%Omega_k
        
        call calculate_comoving_distance(z, cm, r, stat = stat) !! comoving distance
        
        if ( abs(Ok0) < EPS ) return !! for flat cosmology, comoving distance = comoving coordinate
        
        k = sqrt( abs( Ok0 ) ) / SPEED_OF_LIGHT_KMPS * cm%H0
        if ( Ok0 < 0. ) then !! spherical or closed geometry
            r = sin( k*r ) / k
        else !! hyperbolic or open geometry
            r = sinh( k*r ) / k
        end if
        
    end subroutine get_comoving_coordinate

    !>
    !! Calculate luminocity distance at redshift z.
    !!
    !! Parameters:
    !!  z   : real            - Redshift (must be greater than -1).
    !!  cm  : cosmology_model - Cosmology parameters
    !!  r   : real            -  Luminocity distance in Mpc.
    !!  stat: integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine get_luminocity_distance(z, cm, r, stat)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: r !! distance in Mpc
        integer , intent(out), optional :: stat

        call get_comoving_coordinate(z, cm, r, stat = stat) 
        r = r * (z + 1.)
        
    end subroutine get_luminocity_distance

    !>
    !! Calculate angular diameter distance at redshift z.
    !!
    !! Parameters:
    !!  z   : real            - Redshift (must be greater than -1).
    !!  cm  : cosmology_model - Cosmology parameters
    !!  r   : real            -  Angular diameter distance in Mpc.
    !!  stat: integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine get_angular_diameter_distance(z, cm, r, stat)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: r !! distance in Mpc
        integer , intent(out), optional :: stat

        call get_comoving_coordinate(z, cm, r, stat = stat) 
        r = r / (z + 1.)
        
    end subroutine get_angular_diameter_distance

    !>
    !! Calculate angular size corresponding to a physical size x, redshift z.
    !!
    !! Parameters:
    !!  x    : real            - Physical size in Mpc.
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  theta: real            - Calculated angular size in arcsec.
    !!  stat : integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine get_angular_size(x, z, cm, theta, stat)
        real(dp), intent(in)  :: x !! physical size in Mpc
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: theta !! angular size in arcsec
        integer , intent(out), optional :: stat

        real(dp) :: r 
        call get_angular_diameter_distance(z, cm, r, stat = stat) !! angular diameter distance in Mpc
        theta = (x / r) * ( 3600.0*180.0 / PI ) !! angular size in arcsec

    end subroutine get_angular_size

    !>
    !! Calculate physical size corresponding to a angular size x, redshift z.
    !!
    !! Parameters:
    !!  x   : real            - Angular size in arcsec.
    !!  z   : real            - Redshift (must be greater than -1).
    !!  cm  : cosmology_model - Cosmology parameters
    !!  r   : real            - Physical size in Mpc.
    !!  stat: integer         - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine get_physical_size(x, z, cm, r, stat)
        real(dp), intent(in)  :: x !! angular size in arcsec 
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: r !! physical size in Mpc
        integer , intent(out), optional :: stat
    
        call get_angular_diameter_distance(z, cm, r, stat = stat) !! angular diameter distance in Mpc
        r = (x*r) * ( PI / 3600.0 / 180.0 ) !! physical size 

    end subroutine get_physical_size
    
end module dist_time_calculator
