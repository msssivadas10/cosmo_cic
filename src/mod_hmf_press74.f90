module mod_hmf_press74
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    public :: mf_press74
    
contains

    !>
    !! Calculate halo mass function model by Press & Schechter (1974). it is 
    !! based on spherical collapse.
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_press74(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        real(dp) :: nu
                
        nu = DELTA_SC / s
        fs = sqrt( 2./PI ) * nu * exp(-0.5*nu**2)
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_press74
    
end module mod_hmf_press74