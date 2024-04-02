!!
!! Calculation of matter power spectrum and related quatities.
!!
module matter_power_calculator
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmo_t
    use variance_calculator, only: vc => calculate_variance
    implicit none
    
    private

    interface
        !! Interface to transfer function model
        subroutine tf_calculate(k, z, cm, tk, dlntk, stat) 
            use iso_fortran_env, only: dp => real64
            use objects, only: cosmo_t
            real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in) :: z !! redshift
            class(cosmo_t), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: tk
            real(dp), intent(out), optional :: dlntk
            integer , intent(out), optional :: stat
        end subroutine tf_calculate
    end interface
        
    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_K  = 40 !! invalid value for wavenumber
    integer, parameter :: ERR_MODEL_NOT_SET_TF = 41 !! transfer function model not set

    procedure(tf_calculate), pointer :: tf => null() !! linear transfer function model
    logical  :: has_tf = .false.
    real(dp) :: NORM   = 1.0_dp  !! Power spectrum normalization factor so that sigma^2(8 Mpc/h) = 1

    public :: set_power_model
    public :: get_power_spectrum
    public :: get_variance, set_normalization, get_normalization
    
contains

    !>
    !! Set mass-function model to use.
    !!
    !! Parameters:
    !!  mf1 : procedure - Mass function model
    !!  stat: integer   - Status flag
    !!
    subroutine set_power_model(tf1, stat)
        procedure(tf_calculate) :: tf1  !! linear transfer function model
        integer , intent(out)   :: stat

        !! set transfer function model
        tf => tf1
        has_tf = .true.
        stat   = 0

    end subroutine set_power_model

    !>
    !! Calculate the linear matter power spectrum. Scale the calculated power spectrum value 
    !! by `sigma8^2` to get the actual normalised power spectrum. 
    !!
    !! Parameters:
    !!  k    : real    - Wavenumber in 1/Mpc.
    !!  z    : real    - Redshift
    !!  cm   : cosmo_t - Cosmology parameters.
    !!  pk   : real    - Value of calculated power spectrum (unit: Mpc^-3).
    !!  tk   : real    - Value of calculated transfer function (optional).
    !!  dlnpk: real    - Value of calculated log-derivative / effective index (optional).
    !!  stat : integer - Status flag. Non-zero for failure.
    !! 
    subroutine get_power_spectrum(k, z, cm, pk, tk, dlnpk, stat) 
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: pk
        real(dp), intent(out), optional :: tk, dlnpk
        integer , intent(out), optional :: stat

        real(dp) :: f, ns
        integer  :: stat2
        ns = cm%ns

        if ( .not. has_tf ) then
            if ( present(stat) ) stat =  ERR_MODEL_NOT_SET_TF
            return
        end if

        !! transfer function
        call tf(k, z, cm, f, dlntk = dlnpk, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return
        if ( present(tk) ) tk = f

        !! effective index: 1-st log-derivative of p(k) w.r.to k
        if ( present(dlnpk) ) dlnpk = ns + 2*dlnpk

        !! power spectrum, normalised so that sigma^2(8 Mpc/h) = 1  
        pk = NORM * k**ns * f**2
        
    end subroutine get_power_spectrum

    !>
    !! Calculate the linear matter power spectrum, without normalization.
    !!
    !! Parameters:
    !!  k    : real    - Wavenumber in 1/Mpc.
    !!  z    : real    - Redshift
    !!  cm   : cosmo_t - Cosmology parameters.
    !!  pk   : real    - Value of calculated power spectrum (unit: Mpc^-3).
    !!  stat : integer - Status flag. Non-zero for failure.
    !! 
    subroutine ps_calculate(k, cm, pk, args, stat) 
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: pk
        real(dp), intent(in), optional :: args(:) !! additional arguments
        integer , intent(out), optional :: stat
        
        real(dp) :: z = 0.0_dp !! redshift (default is 0)
        integer  :: stat2 = 0
        real(dp) :: ns
        ns = cm%ns
        if ( present(args) ) z = args(1) !! 1-st argument is the redshift

        if ( .not. has_tf ) then
            if ( present(stat) ) stat =  ERR_MODEL_NOT_SET_TF
            return
        end if

        !! transfer function
        call tf(k, z, cm, pk, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! power spectrum
        pk = k**ns * pk**2
        
    end subroutine ps_calculate

    !>
    !! Calculate the smoothed linear variance of matter density. Calculated sigma^2 value is in
    !! units of `sigma8^2`.
    !!
    !! Parameters:
    !!  r    : real    - Smoothing scale in Mpc.
    !!  z    : real    - Redshift
    !!  cm   : cosmo_t - Cosmology parameters.
    !!  sigma: real    - Value of calculated variance (unit: Mpc^-3).
    !!  dlns : real    - Value of calculated 1-st log-derivative (optional).
    !!  d2lns: real    - Value of calculated 2-nd log-derivative (optional).
    !!  stat : integer - Status flag. Non-zero for failure.
    !! 
    subroutine get_variance(r, z, cm, sigma, dlns, d2lns, stat) 
        real(dp), intent(in) :: r !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: sigma
        real(dp), intent(out), optional :: dlns, d2lns
        integer , intent(out), optional :: stat 
        real(dp), dimension(1) :: args
        integer :: stat2 = 0

        args(1) = z !! 1-st argument is the redshift

        !! calculate variance
        call vc(ps_calculate, r, cm, sigma, dlns = dlns, d2lns = d2lns, args = args, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! normalization
        sigma = NORM * sigma 
        
    end subroutine get_variance

    !>
    !! Calculate sigma-8 normalization.
    !!
    !! Parameters:
    !!  cm  : cosmo_t - Cosmology parameters.
    !!  stat: integer - Status flag. Non-zero for failure.
    !! 
    subroutine set_normalization(cm, stat)
        class(cosmo_t), intent(inout) :: cm !! cosmology parameters
        integer , intent(out), optional :: stat 

        real(dp) :: calculated, r
        real(dp), dimension(1) :: args
        integer  :: stat2 = 0
        r = 8.0 / (0.01 * cm%H0) !! = 8 Mpc/h
        args(1) = 0.0_dp !! 1-st argument is the redshift

        !! calculating variance at 8 Mpc/h
        call vc(ps_calculate, r, cm, calculated, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! normalization
        NORM = 1. / calculated
        
    end subroutine set_normalization

    !>
    !! Get the current normalization factor.
    !!
    function get_normalization() result(retval)
        real(dp) :: retval
        retval = NORM
    end function get_normalization
    
end module matter_power_calculator