module mod_hmf_crocce01
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    public :: mf_crocce01
    
contains

    !>
    !! Halo mass function by Halo mass function by Crocce et al (2010).
    !! Reference: <http://arxiv.org/abs/0907.0019v2>
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_crocce01(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        
        !! parameters:
        real(dp) :: AA, a, b, c, zp1
        zp1 = args(1) + 1.
        AA  = 0.580*zp1**( -0.130 )
        a   = 1.370*zp1**( -0.150 )
        b   = 0.300*zp1**( -0.084 )
        c   = 1.036*zp1**( -0.024 )

        fs = AA * ( s**(-a) + b ) * exp( -c/s**2 )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_crocce01
    
end module mod_hmf_crocce01