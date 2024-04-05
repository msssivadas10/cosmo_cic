module mod_hmf_reed03
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    !! Parameters
    real(dp) :: AA = 0.3222_dp, a = 0.707_dp, p = 0.3_dp

    public :: mf_reed03
    
contains

    !>
    !! Halo mass function by Halo mass function by Reed et al (2003).
    !! Reference: <http://arXiv.org/abs/astro-ph/0702360v2>
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_reed03(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        real(dp) :: nu
        
        !! Sheth (2001) mass-function:
        nu = DELTA_SC / s
        fs = AA * sqrt( 2.*a / PI ) * nu * exp(-0.5*a*nu**2) * ( 1. + (nu**2 / a)**(-p) )

        !! Reed (2003) modification:
        fs = fs * exp( -0.7 / s / cosh(2*s)**5. )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_reed03
    
end module mod_hmf_reed03