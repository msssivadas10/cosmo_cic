module mod_hmf_tinker08
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    implicit none
    
    private

    !! Tinker (2008) parameter table:
    integer, parameter :: ts = 9
    real(dp) :: ptab(5*ts) = [ 200. ,  300. ,  400. ,  600. ,  800. ,  1200.,  1600.,  2400.,  3200., & ! Delta
                               0.186,  0.200,  0.212,  0.218,  0.248,  0.255,  0.260,  0.260,  0.260, & ! A
                               1.47 ,  1.52 ,  1.56 ,  1.61 ,  1.87 ,  2.13 ,  2.30 ,  2.53 ,  2.66 , & ! a
                               2.57 ,  2.25 ,  2.05 ,  1.87 ,  1.59 ,  1.51 ,  1.46 ,  1.44 ,  1.41 , & ! b
                               1.19 ,  1.27 ,  1.34 ,  1.45 ,  1.58 ,  1.80 ,  1.97 ,  2.24 ,  2.44   ] ! c
    
    real(dp) :: a0, b0, c0, AA0, alpha
    logical  :: ready = .false.

    public :: mf_tinker08_init
    public :: mf_tinker08

contains

    !>
    !! Setup Tinker (2008) model mass-function based on the overdensity, 
    !! Delta = args(1) relative to mean background density.
    !!
    subroutine mf_tinker08_init(args)
        real(dp), intent(in) :: args(:)
        real(dp) :: t, Delta 
        integer  :: i = 1
        
        Delta = args(1)  !! overdensity value

        !! getting parameters by linearly interpolating 
        do while (i < ts)
            i = i + 1
            if ( .not. ( Delta > ptab(i) ) ) exit
        end do
        t     = (Delta - ptab(i-1)) / (ptab(i) - ptab(i-1))
        AA0   = ptab(1*ts+i)*t + ptab(1*ts+i-1)*(1. - t)
        a0    = ptab(2*ts+i)*t + ptab(2*ts+i-1)*(1. - t)
        b0    = ptab(3*ts+i)*t + ptab(3*ts+i-1)*(1. - t)
        c0    = ptab(4*ts+i)*t + ptab(4*ts+i-1)*(1. - t)
        alpha = 10.**( -( 0.75 / log10(Delta/75.) )**1.2 ) !! eqn. 8
        
        ready = .true.

    end subroutine mf_tinker08_init

    !>
    !! Halo mass function by Halo mass function by Tinker et al (2008).
    !!
    !! Reference: <http://arXiv.org/abs/0803.2706v1>
    !!
    !! Parameters:
    !!  s      : real    - Mass variable.
    !!  retval : real    - Calculated value.
    !!  args   : real    - Other arguments: redshift, overdensity etc.
    !!  stat   : integer - Status flag.
    !!
    subroutine mf_tinker08(s, fs, args, stat)
        real(dp), intent(in) :: s
        real(dp), intent(in), optional :: args(:)
        real(dp), intent(out) :: fs
        integer , intent(out), optional :: stat

        real(dp) :: zp1 = 1._dp, AA, a, b, c
        !! check if the model is ready
        if ( .not. ready ) then
            write (stderr, '(a)') 'error: mf_tinker08 - model not initialised'
            if ( present(stat) ) stat = 1
            return 
        end if
        if ( present(args) ) zp1 = args(1) + 1. ! 1-st argument is redshift
        AA  = AA0 * zp1**( -0.14 )   !! eqn. 5
        a   =  a0 * zp1**( -0.06 )   !! eqn. 6
        b   =  b0 * zp1**( -alpha )  !! eqn. 7
        c   =  c0

        !! mass function (eqn. 3)
        fs = AA * ( 1 + ( b / s )**a ) * exp( -c / s**2 ) 
        if ( present(stat) ) stat   = 0 
        
    end subroutine mf_tinker08


    
end module mod_hmf_tinker08