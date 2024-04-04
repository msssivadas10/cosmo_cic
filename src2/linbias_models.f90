!!
!! Some linear bias models
!!
!! All the models defined here has the general call signature `f(args(3), retval, stat=stat)`
!! where the 3-element `args` is the list of inputs arguments: `args(1)=nu`, mass variable, 
!! `args(2)=z`, redshift and `args(3)=Delta`, overdensity relative to mean background density.
!!
module linbias_models
    use iso_fortran_env, only: dp => real64
    use utils, only: PI, DELTA_SC
    implicit none

    private

    public :: bf_cole89, bf_tinker10
    
contains

    !>
    !! Calculate linear halo bias model by Cole & Kaiser (1989) and Mo & White (1996).
    !!
    !! References:
    !!  Shaun Cole and Nick Kaiser. Mon. Not.R. astr. Soc. 237, 1127-1146 (1989).
    !!  H. J. Mo, Y. P. Jing and S. D. M. White. Mon. Not. R. Astron. Soc. 284, 189-201 (1997).
    !!
    !! Parameters:
    !!  args   : real    - Arguments.
    !!  retval : real    - Calculated function value 
    !!  stat   : integer - Status flag (non-zero for errors)
    !!
    subroutine bf_cole89(args, retval, stat)
        real(dp), intent(in) :: args(3)
        
        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        
        real(dp) :: nu

        nu     = args(1)
        retval = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        if ( present(stat) ) stat   = 0
        
    end subroutine bf_cole89

    !>
    !! Calculate linear halo bias model by Tinker et al. (2010).
    !!
    !! References: <http://arxiv.org/abs/1001.3162v2> (2010)
    !!
    !! Parameters:
    !!  args   : real     - Arguments.
    !!  retval : real     - Calculated function value 
    !!  stat   : integer  - Status flag (non-zero for errors)
    !!
    subroutine bf_tinker10(args, retval, stat)
        real(dp), intent(in) :: args(3)
        
        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: y, A1, a, B1, b, C1, c
        real(dp) :: nu, Delta
        nu    = args(1)
        Delta = args(3)

        !! parameters
        y  = log10( Delta )
        A1 = 1.0 + 0.24 * y * exp( -( 4. / y )**4 )
        a  = 0.44 * y - 0.88
        B1 = 0.183_dp
        b  = 1.5_dp
        C1 = 0.019 + 0.107 * y + 0.19 * exp( -( 4. / y )**4 )
        c  = 2.4_dp

        retval = 1.0 - A1 * nu**a / ( nu**a + DELTA_SC**a ) + B1 * nu**b + C1 * nu**c
        if ( present(stat) ) stat   = 0

    end subroutine bf_tinker10
    
end module linbias_models