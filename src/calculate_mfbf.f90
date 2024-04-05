module calculate_mfbf
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: PI, RHO_CRIT0, DELTA_SC
    use cosmology, only: cosmo_t, get_cosmology
    use calculate_specmom, only: SMC_get_specmom
    implicit none

    private

    interface
        !! Interface to mass function or bias calculator
        subroutine fs_calculate(s, retval, args, stat)
            use iso_fortran_env, only: dp => real64
            real(dp), intent(in) :: s                 !! inputs
            real(dp), intent(in), optional :: args(:) !! inputs
            real(dp), intent(out) :: retval    !! outputs
            integer , intent(out), optional :: stat
        end subroutine fs_calculate
    end interface

    public :: calculate_massfunc_bias
    
contains

    !>
    !! Calculate the halo mass-function and bias for a given mass and redshift.
    !!
    !! Parameters:
    !!  mf      : procedure - Halo mass function model
    !!  bf      : procedure - Halo bias function model
    !!  m       : real      - Mass in Msun
    !!  s       : real      - Pre-calculated variance values
    !!  dlns    : real      - Pre-calculated log derivative of sigma w.r.to mass
    !!  z       : real      - Redshift
    !!  mf_args : real      - Arguments passed to halo mass-function model
    !!  bf_args : real      - Arguments passed to halo bias model
    !!  dndlnm  : real      - Calculated halo mass function in Mpc^-3
    !!  bm      : real      - Calculated halo bias 
    !!  fs      : real      - Calculated mass function, f(sigma)
    !!  stat    : integer   - Status flag
    !! 
    subroutine calculate_massfunc_bias(mf, bf, m, s, dlns, dndlnm, bm, fs, mf_args, bf_args, stat)
        procedure(fs_calculate) :: mf         !! mass function model 
        procedure(fs_calculate) :: bf         !! bias function model 
        real(dp), intent(in)    :: m, s, dlns !! mass varaiables
        real(dp), intent(in), optional :: mf_args(:) !! extra mass function arguments
        real(dp), intent(in), optional :: bf_args(:) !! extra bias function arguments

        real(dp), intent(out) :: dndlnm !! halo mass function
        real(dp), intent(out) :: bm     !! halo bias function
        real(dp), intent(out), optional :: fs 
        integer , intent(out), optional :: stat 

        real(dp) :: rho_m, fsigma, nu
        integer  :: stat2 = 0
        type(cosmo_t) :: cm

        !! get the global cosmology model
        cm = get_cosmology()

        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: calculate_massfunc_bias - cosmology model is not initialised'
            return
        end if
        !! check if k value is correct
        if ( m <= 0. ) then !! invalid value for m
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: calculate_massfunc_bias - mass m must be positive'
            return
        end if

        !! calculate mass-function f(sigma)
        call mf(s, fsigma, args = mf_args, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write(stderr,'(a)') 'error: calculate_massfunc_bias - failed to calculate massfunction f'
            return
        end if
        if ( present(fs) ) fs = fsigma

        !! calculate mass-function
        rho_m  = cm%Om0 * RHO_CRIT0 * (0.01*cm%H0)**2 !! Msun/Mpc^3 
        dndlnm = fsigma * abs(dlns / 6.) * rho_m / m  !! Mpc^-3

        !! calculate bias 
        nu = DELTA_SC / s
        call bf(nu, bm, args = bf_args, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write(stderr,'(a)') 'error: calculate_massfunc_bias - failed to calculate bias'
            return
        end if
        
    end subroutine calculate_massfunc_bias
    
end module calculate_mfbf
