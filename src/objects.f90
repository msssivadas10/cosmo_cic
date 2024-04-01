module objects
    use iso_fortran_env, only: dp => real64
    use constants, only: EPS, DELTA_SC
    implicit none

    private

    !>
    !! Cosmology model parameters
    !!
    type, public :: cosmology_model
        real(dp) :: H0 !! Hubble parameter

        real(dp) :: Omega_m  !! Total matter density parameter
        real(dp) :: Omega_b  !! Baryon density parameter
        real(dp) :: Omega_nu =  0.0_dp !! Massive neutrino density parameter
        real(dp) :: Omega_de = -1.0_dp !! Dark-energy density parameter
        real(dp) :: Omega_k  =  0.0_dp !! Curvature density parameter

        real(dp) :: Nnu = 3.0_dp !! Number of massive neutrino species

        real(dp) :: ns = 1.0_dp !! Index of power spectrum

        real(dp) :: sigma8 = 1.0_dp !! RMS matter variance, smoothed at 8 Mpc/h scale

        real(dp) :: Tcmb0 = 2.725_dp !! Temperature of CMB radiation

        real(dp) :: w0 = -1.0_dp !! Dark-energy evolution parameter: constant part
        real(dp) :: wa =  0.0_dp !! Dark-energy evolution parameter: variable part

        !! Matter density fractions
        real(dp) :: fb  = 0.0_dp !! baryons
        real(dp) :: fc  = 0.0_dp !! cdm
        real(dp) :: fnu = 0.0_dp !! neutrino
        real(dp) :: fnb = 0.0_dp !! neutrino + baryon
        real(dp) :: fcb = 0.0_dp !! cdm + baryon
        real(dp) :: pc  = 0.0_dp !! pc
        real(dp) :: pcb = 0.0_dp !! pcb

        real(dp) :: ps_norm = 1.0_dp !! Matter power spectrum normalization     

        !! Extra arguments for power spectrum calculations
        logical :: psarg_include_nu = .false. !! Use power spectrum including neutrino (for EH + neutrino models)

        contains
            procedure :: initialize_cosmology !! Initialise the model 
            procedure :: calculate_hubble_func
            procedure :: get_collapse_density
    end type

contains

    subroutine initialize_cosmology(self, stat)
        class(cosmology_model), intent(inout) :: self
        integer, intent(out) :: stat

        stat = 1 !! 0: success and 1: failure

        !! checking hubble parameter is positive
        if ( self%H0 <= 0. ) return

        !! checking density parameters (must be positive)
        if ( self%Omega_m < 0. ) return
        if ( self%Omega_b < 0. ) return
        if ( self%Omega_nu < 0. ) return
        if ( self%Omega_b + self%Omega_nu > self%Omega_m ) return

        !! neutrino number must be positive
        if ( self%Nnu < 0. ) return

        !! Temperature must be non-zero positive
        if ( self%Tcmb0 <= 0. ) return

        !! checking / setting dark-energy and curvature density
        if ( self%Omega_de < 0. ) then !! calculate from matter density assuming flat model
            self%Omega_de = 1. - self%Omega_m
        else !! calculate curvature density for curved model
            self%Omega_k  = 1. - ( self%Omega_m + self%Omega_de )
        end if

        !! fractions
        self%fb  = self%Omega_b / self%Omega_m
        self%fnu = self%Omega_nu / self%Omega_m
        self%fnb = self%fnu + self%fb
        self%fcb = 1. - self%fnu
        self%fc  = 1. - self%fnb

        !! eqn. 14 in EH98 paper
        self%pc  = 0.25*( 5 - sqrt( 1 + 24.0*self%fc  ) ) 
        self%pcb = 0.25*( 5 - sqrt( 1 + 24.0*self%fcb ) )

        stat = 0 !! success
        
    end subroutine initialize_cosmology

    !>
    !! Calculate hubble parameter function E(z)
    !!
    !! Parameters:
    !!  z   : real - Redshift
    !!  f   : real - Calculated value of function
    !!  dlnf: real - Calculated value of 1-st log-derivative
    !!
    subroutine calculate_hubble_func(self, z, f, dlnf)
        class(cosmology_model), intent(in) :: self
        real(dp), intent(in) :: z
        real(dp), intent(out) :: f
        real(dp), intent(out), optional :: dlnf
        real(dp) :: zp1, Om0, Ode0, Ok0, tmp, wa, w0, p
        Om0  = self%Omega_m
        Ode0 = self%Omega_de
        Ok0  = self%Omega_k
        w0   = self%w0
        wa   = self%wa
        
        zp1 = z + 1. 

        !! matter part
        f   =  Om0 * zp1**3
        if ( present(dlnf) ) then
            dlnf = 3*f
        end if

        !! curvature part
        if ( abs(Ok0) > EPS ) then !! model with non-zero curvature
            tmp = Ok0 * zp1**2
            f   = f + tmp
            if ( present(dlnf) ) then 
                dlnf = dlnf + 2*tmp
            end if
        end if

        !! dark-energy part
        if ( abs(wa) < EPS ) then !! wa = 0: constant w
            if ( abs(w0 + 1.) < EPS ) then !! w0 = -1: cosmological constant
                p   = 0._dp
                tmp = Ode0 
                f   = f + tmp
            else !! constant w 
                p   = 3*(w0 + 1.)
                tmp = Ode0 * zp1**p 
                f   = f + tmp
                if ( present(dlnf) ) then
                    dlnf = dlnf + p * tmp / zp1 
                end if    
            end if
        else !! general w0-wa model
            p   = 3.*( w0 + wa * z / zp1 ) + 1.
            tmp = Ode0 * zp1**p
            f   = f + tmp
            if ( present(dlnf) ) then
                dlnf = dlnf + tmp * ( p + 3*wa * log(zp1) / zp1 ) / zp1 
            end if 
        end if 

        if ( present(dlnf) ) then
            dlnf = dlnf / f !! log derivative
        end if  

    end subroutine calculate_hubble_func

    function get_collapse_density(self, z) result(retval)
        class(cosmology_model), intent(in) :: self
        real(dp), intent(in) :: z

        real(dp) :: retval
        
        retval = DELTA_SC
        
    end function get_collapse_density
    
end module objects