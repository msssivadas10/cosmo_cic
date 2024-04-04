module transfer_bbks
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use calculate_growth, only: GC_get_growth
    use cosmology, only: cosmo_t
    implicit none

    private

    real(dp) :: theta     !! CMB temperaure in 2.7K unit
    real(dp) :: dplus0    !! Growth factor scaling
    real(dp) :: Gamma_eff !! Shape parameter

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_K  = 40 !! invalid value for wavenumber

    public :: tf_sugiyama95_init
    public :: tf_sugiyama95
    
contains

    !>
    !! Calculate the quantities related to BBKS linear transfer function.
    !!
    !! Parameters:
    !!  cm      : cosmo_t - Cosmology parameters.
    !!  version : integer - Tells if to use original BBKS (1) or corrected version (0).
    !!  stat    : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_sugiyama95_init(cm, version, stat) 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        integer, intent(in) , optional :: version
        integer, intent(out), optional :: stat
        
        real(dp) :: Om0, Ob0, h      
        integer  :: stat2 = 0
        
        !! growth factor at z=0
        call GC_get_growth(cm, 0.0_dp, dplus0, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_sugiyama95_init - failed to calculate growth.'
        end if

        Om0   = cm%Om0
        Ob0   = cm%Ob0
        h     = 0.01*cm%H0 !! hubble parameter in 100 km/sec/Mpc unit
        theta = cm%Tcmb0 / 2.7
        
        Gamma_eff = Om0*h**2 !! shape parameter
        
        if ( present(version) .and. ( version == 1) ) return !! use original BBKS function
        
        !! apply baryon correction factor
        Gamma_eff = Gamma_eff * exp(-Ob0 - sqrt(2*h)*Ob0/Om0)
        
    end subroutine tf_sugiyama95_init

    !>
    !! Transfer function model given by Bardeen et al (1986), including correction 
    !! by Sugiyama (1995).  BBKS.
    !!
    !! Parameters:
    !!  k    : real    - Wavenumebr in 1/Mpc.
    !!  z    : real    - Redshift.
    !!  cm   : cosmo_t - Cosmology model parameters.
    !!  tk   : real    - Transfer function.
    !!  dlntk: real    - 1-st log derivative of transfer function (optional).
    !!  stat : integer - Status flag. Non-zero for failure.
    !!
    subroutine tf_sugiyama95(k, z, cm, tk, dlntk, stat)
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        real(dp) :: dplus, q, t0, t1, t2, t3, t4
        integer  :: stat2 = 0

        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_sugiyama95 - cosmology model is not initialised'
            return
        end if
        !! check if k value is correct
        if ( k <= 0. ) then !! invalid value for k
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_sugiyama95 - wavenumber k must be positive'
            return
        end if

        !! linear growth
        call GC_get_growth(cm, z, dplus, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_sugiyama95 - failed to calculate growth.'
        end if
        dplus = dplus / dplus0 !! normalization

        q = k / Gamma_eff !! dimensionless scale

        t0 = 2.34*q
        t1 = 3.89*q
        t2 = (16.1*q)**2
        t3 = (5.46*q)**3
        t4 = (6.71*q)**4
        
        !! BBKS transfer function
        tk = log(1. + t0) / t0 * ( 1. + t1 + t2 + t3 + t4 )**(-0.25)
        tk = dplus * tk

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            t0    = t0 / ( (1. + t0) * log(1. + t0) )
            t1    = 0.25 * ( t1 + 2*t2 + 3*t3 + 4*t4 ) / (1 + t1 + t2 + t3 + t4) + 1.
            dlntk = t0 - t1
        end if
        
    end subroutine tf_sugiyama95

end module transfer_bbks
