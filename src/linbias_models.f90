!!
!! Some linear bias models
!!
module linbias_models
    use iso_fortran_env, only: dp => real64
    use constants, only: PI, DELTA_SC
    use objects, only: cosmology_model
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
    !!  s     : real            
    !!  z     : real            - Redshift
    !!  Delta : real            - Overdensity relative to mean density.
    !!  cm    : cosmology_model - Cosmology parameters.
    !!  retval: real            - Calculated function value 
    !!  stat  : integer         - Status flag (non-zero for errors)
    !!
    subroutine bf_cole89(nu, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: nu
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat

        retval = 1.0 + ( nu**2 - 1.0 ) / DELTA_SC
        if ( present(stat) ) stat   = 0
        
    end subroutine bf_cole89

    !>
    !! Calculate linear halo bias model by Tinker et al. (2010).
    !!
    !! References: <http://arxiv.org/abs/1001.3162v2> (2010)
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real            - Redshift
    !!  Delta : real            - Overdensity relative to mean density.
    !!  cm    : cosmology_model - Cosmology parameters.
    !!  retval: real            - Calculated function value 
    !!  stat  : integer         - Status flag (non-zero for errors)
    !!
    subroutine bf_tinker10(nu, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: nu
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmology_model), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: y, A1, a, B1, b, C1, c

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