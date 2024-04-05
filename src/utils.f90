module utils
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    implicit none

    real(dp), parameter :: EPS    = 1.0e-08 !! tolerance 
    real(dp), parameter :: PI     = 3.141592653589793_dp !! pi
    real(dp), parameter :: SQRT_2 = 1.4142135623730951_dp    

    ! Constants related to cosmology
    real(dp), parameter :: DELTA_SC  = 1.6864701998411453_dp    !! Overdensity threshold for collapse
    real(dp), parameter :: RHO_CRIT0 = 2.77536627E+11_dp        !! Critical density in h^2 Msun / Mpc^3        
    real(dp), parameter :: C_KMPS    = 299792.458_dp            !! Speed of light in km/sec  
    real(dp), parameter :: GMSUN     = 1.32712440018e+20_dp     !! GM for sun in m^3/s^2     
    real(dp), parameter :: MSUN      = 1.98842e+30_dp           !! Mass of sun in kg  
    real(dp), parameter :: AU        = 1.49597870700e+11_dp     !! 1 astronomical unit (au) in m
    real(dp), parameter :: MPC       = 3.085677581491367e+22_dp !! 1 mega parsec (Mpc) 
    real(dp), parameter :: YEAR      = 31558149.8_dp            !! Seconds in a sidereal year      
    
    ! Flags
    integer, parameter :: FLAG_SUCCESS = 0
    integer, parameter :: FLAG_FAIL    = 1
    
contains

    !>
    !! Generate gauss-legendre integration points and weights
    !! 
    !! Parameters:
    !!  n   : integer - Number of points to use. Must be a positive non-zero integer.
    !!  x, w: real    - Calculated nodes and weights arrays.
    !!  stat: integer - Status. 0 for success and 1 for failure.
    !!
    subroutine generate_gaussleg(n, x, w, stat)
        integer , intent(in)  :: n    !! order of the rule: number of points
        real(dp), intent(out) :: x(n) !! integration nodes (points)
        real(dp), intent(out) :: w(n) !! weights
        integer , intent(out), optional :: stat

        real(dp) :: xj, xjo, pm, pn, ptmp
        integer  :: j, k

        if ( n < 1 ) then !! why calculate for n < 2 ?
            if ( present(stat) ) stat = FLAG_FAIL
            write (stderr, '(a)') 'error: cannot calculate nodes for n < 1'
            return
        end if 

        !! for odd order, x = 0 is the {floor(n/2)+1}-th node
        if ( modulo(n, 2) == 1 ) then
            
            !! calculating lenegedre polynomial P_n(0) using its reccurence relation
            xj = 0.d0
            pm = 0.d0
            pn = 1.d0
            do k = 0, n-1
                ptmp = -k*pm / (k + 1.d0)
                pm   = pn
                pn   = ptmp
            end do
            x(n/2 + 1) = 0.d0
            w(n/2 + 1) = 2.d0 / (n*pm)**2 !! weight 
        end if

        !! other nodes
        do j = 1, n/2

            !! initial guess for j-th node (j-th root of n-th legendre polynomial)
            xj  = cos( (2.d0*j - 0.5d0) * PI / (2.d0*n + 1.d0) ) 
            xjo = 1000._dp
            do while ( abs(xj - xjo) > EPS )
                !! calculating lenegedre polynomial P_n(xj) using its reccurence relation
                pm = 0.d0
                pn = 1.d0
                do k = 0, n-1
                    ptmp = ( (2.d0*k + 1.d0)*xj*pn - k*pm ) / (k + 1.d0)
                    pm   = pn
                    pn   = ptmp
                end do
                !! next estimate of the root
                xjo = xj
                xj  = xj - pn * (xj**2 - 1.d0) / (n*xj*pn - n*pm)

            end do
            x(j)     = -xj
            w(j)     =  2.d0 * (1.d0 - xj**2) / (n*xj*pn - n*pm)**2 !! weight for j-th node
            x(n-j+1) =  xj
            w(n-j+1) =  w(j)
            
        end do

        if ( present(stat) ) stat = 0 !! success

    end subroutine generate_gaussleg
    
end module utils