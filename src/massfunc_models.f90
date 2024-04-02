!!
!! Some halo mass-function models
!!
module massfunc_models
    use iso_fortran_env, only: dp => real64
    use constants, only: PI
    use objects, only: cosmo_t
    implicit none

    private

    !! Tinker (2008) parameter table:
    integer, parameter :: ts = 9
    real(dp) :: ptab(5*ts) = [ 200. ,  300. ,  400. ,  600. ,  800. ,  1200.,  1600.,  2400.,  3200., & ! Delta
                               0.186,  0.200,  0.212,  0.218,  0.248,  0.255,  0.260,  0.260,  0.260, & ! A
                               1.47 ,  1.52 ,  1.56 ,  1.61 ,  1.87 ,  2.13 ,  2.30 ,  2.53 ,  2.66 , & ! a
                               2.57 ,  2.25 ,  2.05 ,  1.87 ,  1.59 ,  1.51 ,  1.46 ,  1.44 ,  1.41 , & ! b
                               1.19 ,  1.27 ,  1.34 ,  1.45 ,  1.58 ,  1.80 ,  1.97 ,  2.24 ,  2.44   ] ! c
    
    public :: mf_pres74, mf_sheth01, mf_jenkins01, mf_reed03
    public :: mf_warren06, mf_crocce01, mf_courtin10, mf_tinker08
    
contains

    !>
    !! Calculate halo mass function model by Press & Schechter (1974). it is 
    !! based on spherical collapse.
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_pres74(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: nu

        nu     = cm%get_collapse_density(z) / s
        retval = sqrt( 2./PI ) * nu * exp(-0.5*nu**2)
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_pres74

    !>
    !! Calculate halo mass function model by Sheth et al (2001). It is based 
    !! on ellipsoidal collapse.
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real            - Redshift
    !!  Delta : real            - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real            - Calculated value.
    !!  stat  : integer         - Status flag.
    !!
    subroutine mf_sheth01(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: nu

        !! parameters
        real(dp), parameter :: A_= 0.3222_dp
        real(dp), parameter :: a = 0.707_dp
        real(dp), parameter :: p = 0.3_dp

        nu     = cm%get_collapse_density(z) / s
        retval = A_ * sqrt( 2.*a / PI ) * nu * exp(-0.5*a*nu**2) * ( 1. + (nu**2 / a)**(-p) )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_sheth01

    !>
    !! Halo mass function by Jenkins et al (2001). Valid range: -1.2 <= -log(sigma) <= 1.05.
    !!
    !! Reference: <http://arxiv.org/abs/astro-ph/0005260v2>
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_jenkins01(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat

        retval = 0.315 * ( -abs( 0.61 - log(s) )**(3.8) )
        if ( present(stat) ) stat   = 0

    end subroutine mf_jenkins01

    !>
    !! Halo mass function by Halo mass function by Reed et al (2003).
    !!
    !! Reference: <http://arXiv.org/abs/astro-ph/0702360v2>
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_reed03(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat

        call mf_sheth01(s, z, Delta, cm, retval, stat)
        if ( stat .ne. 0 ) return
         
        retval = retval * exp( -0.7 / s / cosh(2*s)**5. )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_reed03

    !>
    !! Halo mass function by Halo mass function by Warren et al (2006).
    !!
    !! Reference: <http://arXiv.org/abs/astro-ph/0702360v2>
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_warren06(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat

        !! parameters
        real(dp), parameter :: A_ = 0.7234_dp
        real(dp), parameter :: a  = 1.625_dp
        real(dp), parameter :: b  = 0.2538_dp
        real(dp), parameter :: c  = 1.1982_dp
        
        retval = A_ * ( s**(-a) + b ) * exp( -c/s**2 )
        if ( present(stat) ) stat   = 0

    end subroutine mf_warren06

    !>
    !! Halo mass function by Halo mass function by Crocce et al (2010).
    !!
    !! Reference: <http://arxiv.org/abs/0907.0019v2>
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_crocce01(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: zp1

        !! parameters
        real(dp) :: Aaz, az, bz, cz
        zp1 = z + 1.
        Aaz = 0.580 * zp1**(-0.130)
        az  = 1.370 * zp1**(-0.150)
        bz  = 0.300 * zp1**(-0.084)
        cz  = 1.036 * zp1**(-0.024)

        retval = Aaz * ( s**(-az) + bz ) * exp( -cz / s**2 )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_crocce01

    !>
    !! Halo mass function by Halo mass function by Courtin et al (2010).
    !!
    !! Reference: J. Courtin et al. Mon. Not. R. Astron. Soc. 410, 1911-1931 (2011)
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_courtin10(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: nu

        !! parameters
        real(dp) :: A_ = 0.348_dp
        real(dp) :: a  = 0.695_dp
        real(dp) :: p  = 0.1_dp

        nu     = cm%get_collapse_density(z) / s
        retval = A_ * sqrt( 2.*a / PI ) * nu * exp(-0.5*a*nu**2) * ( 1. + (nu**2 / a)**(-p) )
        if ( present(stat) ) stat   = 0
        
    end subroutine mf_courtin10

    !>
    !! Halo mass function by Halo mass function by Tinker et al (2008).
    !!
    !! Reference: <http://arXiv.org/abs/0803.2706v1>
    !!
    !! Parameters:
    !!  s     : real            
    !!  z     : real    - Redshift
    !!  Delta : real    - Overdensity relative to mean density.
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  retval: real    - Calculated value.
    !!  stat  : integer - Status flag.
    !!
    subroutine mf_tinker08(s, z, Delta, cm, retval, stat)
        real(dp), intent(in) :: s 
        real(dp), intent(in) :: z !! redshift
        real(dp), intent(in) :: Delta !! overdensity (not used)
        type(cosmo_t), intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: retval
        integer , intent(out), optional :: stat
        real(dp) :: zp1, t
        real(dp) :: alpha, A1, a, b, c
        integer  :: i

        !! getting parameters by linearly interpolating 
        i = 1
        do while (i < ts)
            i = i + 1
            if ( .not. ( Delta > ptab(i) ) ) exit
        end do
        t  = (Delta - ptab(i-1)) / (ptab(i) - ptab(i-1))
        A1 = ptab(1*ts+i)*t + ptab(1*ts+i-1)*(1. - t)
        a  = ptab(2*ts+i)*t + ptab(2*ts+i-1)*(1. - t)
        b  = ptab(3*ts+i)*t + ptab(3*ts+i-1)*(1. - t)
        c  = ptab(4*ts+i)*t + ptab(4*ts+i-1)*(1. - t)

        !! redshift dependence
        zp1   = z + 1.
        alpha = 10.**( -( 0.75 / log10(Delta/75.) )**1.2 ) !! eqn. 8
        A1    = A1 * zp1**( -0.14 )   !! eqn. 5
        a     =  a * zp1**( -0.06 )   !! eqn. 6
        b     =  b * zp1**( -alpha )  !! eqn. 7

        !! mass function (eqn. 3)
        retval = A1 * ( 1 + ( b / s )**a ) * exp( -c / s**2 ) 
        if ( present(stat) ) stat   = 0 
        
    end subroutine mf_tinker08
    
end module massfunc_models
