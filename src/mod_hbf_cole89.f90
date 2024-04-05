module mod_hbf_cole89
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    public :: bf_cole89
    
contains

    !>
    !! Calculate linear halo bias model by Cole & Kaiser (1989) and Mo & White (1996).
    !!
    !! References:
    !!  [1] Shaun Cole and Nick Kaiser. Mon. Not. R. astr. Soc. 237, 1127-1146 (1989).
    !!  [2] H. J. Mo, Y. P. Jing and S. D. M. White. Mon. Not. R. Astron. Soc. 284, 189-201 (1997).
    !!
    !! Parameters:
    !!  nu     : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine bf_cole89(nu, bnu, args, stat)
        real(dp), intent(in) :: nu
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: bnu
        integer , intent(out), optional :: stat
                
        bnu = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        if ( present(stat) ) stat   = 0
        
    end subroutine bf_cole89
    
end module mod_hbf_cole89