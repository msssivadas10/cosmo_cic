module mod_hmf_sheth01
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    !! Parameters
    real(dp) :: AA = 0.3222_dp, a = 0.707_dp, p = 0.3_dp

    public :: mf_sheth01
    
contains

    !>
    !! Calculate halo mass function model by Sheth et al (2001). It is based 
    !! on ellipsoidal collapse.
    !!
    !! Parameters:
    !!  s      : real    - Mass variable
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments (eg., redshift)
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_sheth01(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat
        real(dp) :: nu
                
        nu = DELTA_SC / s
        fs = AA * sqrt( 2.*a / PI ) * nu * exp(-0.5*a*nu**2) * ( 1. + (nu**2 / a)**(-p) )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_sheth01
    
end module mod_hmf_sheth01