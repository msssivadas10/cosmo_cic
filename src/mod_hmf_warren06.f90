module mod_hmf_warren06
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    !! parameters
    real(dp) :: AA = 0.7234_dp
    real(dp) :: a  = 1.625_dp
    real(dp) :: b  = 0.2538_dp
    real(dp) :: c  = 1.1982_dp

    public :: mf_warren06
    
contains

    !>
    !! Halo mass function by Halo mass function by Warren et al (2006).
    !! Reference: <http://arXiv.org/abs/astro-ph/0702360v2>
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_warren06(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        
        fs = AA * ( s**(-a) + b ) * exp( -c/s**2 )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_warren06
    
end module mod_hmf_warren06