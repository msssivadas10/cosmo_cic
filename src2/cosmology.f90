module cosmology
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils
    implicit none

    private

    !! An object to store cosmology parameters
    type, public :: cosmo_t
        real(dp) :: H0     !! Hubble parameter in km/sec/Mpc
        real(dp) :: Om0    !! Total matter density
        real(dp) :: Ob0    !! Baryon density 
        real(dp) :: Onu0   =  0._dp !! Massive neutrino density
        real(dp) :: Ode0   = -1._dp !! Dark-energy density
        real(dp) :: Ok0    =  0._dp !! Curvature density
        real(dp) :: Nnu    =  3._dp !! Number of massive neutrinos
        real(dp) :: ns     =  1._dp !! Index of power spectrum
        real(dp) :: sigma8 =  1._dp !! Power sectrum normalization factor 
        real(dp) :: Tcmb0  =  2.725_dp !! CMB tenmperature in K
        real(dp) :: w0     = -1._dp    !! Dark-energy evolution parameter (constant part)
        real(dp) :: wa     =  0._dp    !! Dark-energy evolution parameter (variable part)
        logical  :: flat   = .true.    !! Is the universe flat or not

        real(dp) :: ps_norm    = 1._dp   !! Power spectrum normalization
        logical  :: include_nu = .false. !! Include neutrinos for transfer function?
        
        logical, private :: ready = .false. !! Tell if the object ready for use

        contains
            procedure :: init
            procedure :: get_E2
            procedure :: is_ready
    end type cosmo_t
    
contains

    !>
    !! Initialise an cosmology object to make it useful.
    !!
    !! Returns:
    !!  stat : integer - Tells if the initialisation is success (0) or not (1).
    !!
    function init(self) result(stat)
        class(cosmo_t), intent(inout) :: self
        integer :: stat

        stat = 1
        if ( self%H0 <= 0. ) then
            write (stderr, '(a)') 'error: H0 must be positive'
            return
        else if ( self%Om0 < 0. ) then
            write (stderr, '(a)') 'error: Om0 must be zero or positive'
            return
        else if ( self%Ob0 < 0. ) then
            write (stderr, '(a)') 'error: Ob0 must be zero or positive'
            return
        else if ( self%Onu0 < 0. ) then
            write (stderr, '(a)') 'error: Onu0 must be positive'
            return
        else if ( self%Ob0 + self%Onu0 > self%Om0 ) then
            write (stderr, '(a)') 'error: Ob0 + Onu0 mast be less than Om0'
            return
        else if ( self%Nnu <= 0. ) then
            write (stderr, '(a)') 'error: Nnu must be non-zero positive'
            return
        else if ( self%Tcmb0 <= 0. ) then
            write (stderr, '(a)') 'error: Tcmb0 must be non-zero positive'
            return
        end if

        !! calculating dark-energy density and curvature
        if ( self%Ode0 < 0. ) then
            if ( self%flat ) then
                self%Ode0 = 1. - self%Om0
                self%Ok0  = 0._dp
            else
                write (stderr, '(a)') 'error: Ode0 must be positive'
                return
            end if
        else
            self%Ok0 = 1. - ( self%Om0 + self%Ode0 )
        end if 

        self%ready = .true.
        stat       = 0
        
    end function init

    !>
    !! Calculate hubble parameter function E(z)
    !!
    !! Parameters:
    !!  z    : real    - Redshift
    !!  f    : real    - Calculated value of function
    !!  dlnf : real    - Calculated value of 1-st log-derivative
    !!  stat : integer - Status
    !!
    subroutine get_E2(self, z, f, dlnf, stat)
        class(cosmo_t), intent(in) :: self
        real(dp), intent(in)  ::  z
        real(dp), intent(out) ::  f
        real(dp), intent(out), optional ::  dlnf
        integer , intent(out), optional ::  stat

        real(dp) :: zp1, pde, tmp

        if ( .not. self%ready ) then
            write (stderr, '(a)') 'error: object is not properly initialized'
            if ( present(stat) ) stat = 1
            return
        end if
        if ( present(stat) ) stat = 0
        
        !! matter part
        zp1 = z + 1. 
        f   = self%Om0 * zp1**3
        if ( present(dlnf) ) dlnf = 3*f

        !! curvature part
        if ( abs( self%Ok0 ) > EPS ) then !! model with non-zero curvature
            tmp = self%Ok0 * zp1**2
            f   = f + tmp
            if ( present(dlnf) ) dlnf = dlnf + 2*tmp
        end if

        !! dark-energy part
        if ( abs( self%wa ) < EPS ) then !! wa = 0: constant w
            if ( abs( self%w0 + 1. ) < EPS ) then !! w0 = -1: cosmological constant
                pde = 0._dp
                tmp = self%Ode0 
                f   = f + tmp
            else !! constant w 
                pde = 3*( self%w0 + 1. )
                tmp = self%Ode0 * zp1**pde
                f   = f + tmp
                if ( present(dlnf) ) dlnf = dlnf + pde * tmp / zp1 
            end if
        else !! general w0-wa model
            pde = 3.*( self%w0 + self%wa * z / zp1 ) + 1.
            tmp = self%Ode0 * zp1**pde
            f   = f + tmp
            if ( present(dlnf) ) dlnf = dlnf + tmp * ( pde + 3*self%wa * log(zp1) / zp1 ) / zp1 
        end if 

        if ( present(dlnf) ) dlnf = dlnf / f !! log derivative

    end subroutine get_E2

    function is_ready(self) result(retval)
        class(cosmo_t), intent(in) :: self
        logical :: retval

        retval = self%ready
    end function is_ready
    
end module cosmology