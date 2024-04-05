module mod_hbf_tinker10
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: DELTA_SC, PI
    implicit none

    private

    !! Tinker (2010) parameters
    real(dp) :: y, AA, BB, CC, a, b, c
    logical  :: ready = .false.

    public :: bf_tinker10_init
    public :: bf_tinker10
    
contains

    !>
    !! Setup Tinker (2010) model linear bias function based on the overdensity, 
    !! Delta = args(1) relative to mean background density.
    !!
    subroutine bf_tinker10_init(args)
        real(dp), intent(in) :: args(:)
        real(dp) :: Delta 

        Delta = args(1)
        y     = log10( Delta )
        AA    = 1.0 + 0.24 * y * exp( -( 4. / y )**4 )
        a     = 0.44 * y - 0.88
        BB    = 0.183_dp
        b     = 1.5_dp
        CC    = 0.019 + 0.107 * y + 0.19 * exp( -( 4. / y )**4 )
        c     = 2.4_dp
        ready = .true.
        
    end subroutine bf_tinker10_init

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
    subroutine bf_tinker10(nu, bnu, args, stat)
        real(dp), intent(in) :: nu
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: bnu
        integer , intent(out), optional :: stat
                
        !! check if the model is ready
        if ( .not. ready ) then
            write (stderr, '(a)') 'error: bf_tinker10 - model not initialised'
            if ( present(stat) ) stat = 1
            return 
        end if
        bnu = 1.0 - AA * nu**a / ( nu**a + DELTA_SC**a ) + BB * nu**b + CC * nu**c
        if ( present(stat) ) stat   = 0
        
    end subroutine bf_tinker10
    
end module mod_hbf_tinker10