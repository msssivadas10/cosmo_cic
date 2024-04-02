!!
!! Linear growth calculations
!!
module growth_calculator
    use iso_fortran_env, only: dp => real64
    use constants, only: PI, SPEED_OF_LIGHT_KMPS, EPS
    use numerical, only: generate_gaussleg
    use objects, only: cosmo_t
    implicit none

    private 

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_CALC_NOT_SETUP   = 21 !! calculator not setup

    integer  :: nq = 0 !! number of points for integration
    real(dp), allocatable :: xq(:) !! nodes for integration
    real(dp), allocatable :: wq(:) !! weights for integration
    logical :: ready = .false.

    public :: calculate_linear_growth
    public :: setup_growth_calculator, reset_growth_calculator

contains

    !>
    !! Setup the growth calculator.
    !!
    !! Parameters:
    !!  n   : integer - Size of the integration rule.
    !!  stat: integer - Status variable. 0 for success.
    !!
    subroutine setup_growth_calculator(n, stat)
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
        
    end subroutine setup_growth_calculator
    
    !>
    !! Reset growth calculator to initial state. 
    !!
    subroutine reset_growth_calculator()
        deallocate( xq )
        deallocate( wq )
        nq    = 0
        ready = .false.
    end subroutine reset_growth_calculator

    !>
    !! Evaluate the integral expression related to linear growth_calculator calculation.
    !!
    !! Parameters:
    !!  z    : real    - Redshift (must be greater than -1).
    !!  cm   : cosmo_t - Cosmology parameters
    !!  dplus: real    - Calculated value of growth factor.
    !!  fplus: real    - Calculated value of growth rate (optional).
    !!  stat : integer - Status. 1: not setup propery, 2: invalid redshift.
    !!
    subroutine calculate_linear_growth(z, cm, dplus, fplus, stat)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: dplus           !! growth factor
        real(dp), intent(out), optional :: fplus !! growth rate
        integer , intent(out), optional :: stat  !! integration rule

        integer  :: i
        real(dp) :: integral !! value of the integral
        real(dp) :: a, fa, dlnfa, x_scale, zp1
        
        if ( .not. ready ) then !! not properly setup
            if ( present(stat) ) stat = ERR_CALC_NOT_SETUP
            return
        end if

        if ( z <= -1. ) then !! invalid value for redshift
            if ( present(stat) ) stat = ERR_INVALID_VALUE_Z
            return
        end if
        
        integral = 0.0_dp

        !! ditance by evaluating the integral of [ a * E(a) ]^-3 from 0 to a...
        x_scale = 0.5_dp / ( z + 1._dp ) 
        do i = 1, nq
            !! scale factor
            a = x_scale * (xq(i) + 1.) 

            !! calculating integrand
            call cm%calculate_hubble_func(1/a-1., fa) !! E**2(z)
            fa = 1./ ( a * sqrt( fa ) )**3
            
            integral = integral + wq(i) * fa
        end do
        integral = integral * x_scale

        !! calculation of hubble parameter function E(z)
        call cm%calculate_hubble_func(z, fa, dlnfa)
        zp1 = z + 1

        !! calculating linear growth factor
        dplus = sqrt(fa) * integral

        if ( present(fplus) ) then !! calculating linear growth rate
            fplus = zp1**2 / fa / dplus - 0.5*dlnfa
        end if

        if ( present(stat) ) then !! succesful!
            stat = 0
        end if

    end subroutine calculate_linear_growth
    
end module growth_calculator