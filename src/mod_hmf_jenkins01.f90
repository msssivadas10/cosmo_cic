module mod_hmf_jenkins01
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    public :: mf_jenkins01
    
contains

    !>
    !! Halo mass function by Jenkins et al (2001).
    !! Valid range: -1.2 <= -log(sigma) <= 1.05.
    !! Reference: <http://arxiv.org/abs/astro-ph/0005260v2>
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_jenkins01(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        
        fs = 0.315 * ( -abs( 0.61 - log(s) )**(3.8) )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_jenkins01
    
end module mod_hmf_jenkins01