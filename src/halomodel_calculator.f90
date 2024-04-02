!!
!! Halo model calculations
!!
module halomodel_calculator
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmo_t
    use halo_model, only: halomodel_t
    use numerical, only: generate_gaussleg
    use dist_time_calculator, only: calculate_comoving_distance, setup_distance_calculator
    use growth_calculator, only: setup_growth_calculator
    use matter_power_calculator, only: set_power_model,                     &
                                    &  set_normalization,                   &
                                    &  get_variance,                        &
                                    &  tf_calculate 
    use variance_calculator, only: setup_variance_calculator
    use massfunc_bias_calculator, only: setup_massfunc_bias_calculator,     &
                                    &   set_bias_model,                     &
                                    &   set_massfunc_model,                 &
                                    &   calculate_massfunc_bias,            &
                                    &   calculate_massfunc, calculate_bias, &
                                    &   fs_calculate
    implicit none

    private

    interface
        !! Redshift completeness function
        function z_distrib(z) result(retval)
            use iso_fortran_env, only: dp => real64
            real(dp), intent(in) :: z
            real(dp) :: retval
        end function z_distrib
    end interface

    interface
        !! A function of mass and halo models
        function halo_function(m, cm, hm) result(retval)
            use iso_fortran_env, only: dp => real64
            use halo_model, only: halomodel_t
            use objects, only: cosmo_t
            real(dp), intent(in)  :: m !! halo mass in Msun
            class(halomodel_t), intent(in) :: hm !! halo model parameters
            class(cosmo_t)    , intent(in) :: cm !! cosmology parameters
            real(dp) :: retval
        end function halo_function
    end interface

    !! Error flags
    integer, parameter  :: ERR_INVALID_VALUE_Z = 10 !! invalid value for redshift
    integer , parameter :: ERR_CALC_NOT_SETUP  = 51 

    real(dp), parameter :: LN_10 = 2.3025850929940455_dp !! ln(10)

    integer  :: nq = 0             !! number of points for integration
    real(dp), allocatable :: xq(:) !! nodes for integration
    real(dp), allocatable :: wq(:) !! weights for integration
    real(dp), allocatable :: mftab(:) !! mass function values
    real(dp), allocatable :: bftab(:) !! bias function values
    real(dp) :: redshift  = -99.0_dp  !! current redshift value
    logical  :: ready = .false.

    !! mass-function and bias calculators
    public :: calculate_massfunc, calculate_massfunc_bias, calculate_bias

    public :: setup_halomodel_calculator, reset_halomodel_calculator
    public :: calculate_mfbf_table
    public :: calculate_mf_integral, calculate_mfbf_integral
    public :: get_halo_density
    public :: get_average_galaxy_density
    public :: get_average_galaxy_bias
    public :: calculate_galaxy_params_zavg
    
contains

    !>
    !! Setup the halo model calculator.
    !!
    !! Parameters:
    !!  n   : integer   - Size of the integration rule.
    !!  stat: integer   - Status variable. 0 for success.
    !!
    subroutine setup_halomodel_calculator(nm, nk, nz, mf, bf, tf, cm, stat, filt)
        integer, intent(in)  :: nm, nk, nz
        integer, intent(out) ::  stat
        procedure(fs_calculate)  :: mf, bf
        procedure(tf_calculate)  :: tf
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        character(len=6), intent(in), optional :: filt

        !! allocate node array
        if ( .not. allocated(xq) ) allocate( xq(nm) )
        
        !! allocate weights array
        if ( .not. allocated(wq) ) allocate( wq(nm) )

        !! allocate mass-fcuntion array
        if ( .not. allocated(mftab) ) allocate( mftab(nm) )

        !! allocate bias array
        if ( .not. allocated(bftab) ) allocate( bftab(nm) )
        
        !! generating integration rule...
        nq = nm
        call generate_gaussleg(nm, xq, wq, stat = stat)
        if ( stat .ne. 0 ) return !! failed to generate integration rule 

        !! setup power spectrum
        call set_power_model(tf, stat = stat)
        if ( stat .ne. 0 ) return

        !! initialize growth factor calculator
        call setup_growth_calculator(nz, stat = stat)
        if ( stat .ne. 0 ) return

        !! initialize distance calculator
        call setup_distance_calculator(nz, stat = stat)
        if ( stat .ne. 0 ) return

        !! initialize variance calculator
        call setup_variance_calculator(nk, stat, filt = filt)
        if ( stat .ne. 0 ) return

        !! normalization
        call set_normalization(cm, stat = stat)
        if ( stat .ne. 0 ) return
        
        !! setup mass function calculator
        call setup_massfunc_bias_calculator(get_variance, stat = stat)
        if ( stat .ne. 0 ) return 
        
        !! set mass-function model
        call set_massfunc_model(mf, stat = stat)
        if ( stat .ne. 0 ) return 
        
        !! set bias model
        call set_bias_model(bf, stat = stat)
        if ( stat .ne. 0 ) return 

        ready = .true. !! ready for calculations
        
    end subroutine setup_halomodel_calculator

    !>
    !! Reset halo model calculator to initial state. 
    !!
    subroutine reset_halomodel_calculator()
        deallocate( xq )
        deallocate( wq )
        deallocate( mftab )
        deallocate( bftab )
        nq    = 0
        ready = .false.
        
        !! reset redshift to a negative value
        redshift = -99.0_dp

    end subroutine reset_halomodel_calculator

    subroutine calculate_mfbf_table(z, Delta, cm, hm, stat)
        real(dp), intent(in) :: z
        real(dp), intent(in) :: Delta
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters
        class(halomodel_t), intent(in) :: hm !! halo model parameters

        integer, intent(out), optional :: stat 

        real(dp) :: lnMa, lnMb, m_scale, m_shift, m
        integer  :: i, stat2
        stat2 = 0

        if ( z <= -1. ) stat =  ERR_INVALID_VALUE_Z
        if ( .not. ready ) stat2 = ERR_CALC_NOT_SETUP
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! mass integration is done in log scale, from M=Mmin to M=10^18 Msun
        lnMa    = hm%log_Mmin * LN_10
        lnMb    = log(1.0e+18) !! TODO: change to 5-sigma mass
        m_scale = 0.5*(lnMb - lnMa)
        m_shift = lnMa + m_scale
    
        !! calculate mass-function and bias at nodes
        do i = 1, nq
            m  = exp( m_scale*xq(i) + m_shift ) !! mass in Msun
            call calculate_massfunc_bias(m, z, Delta, cm, mftab(i), bftab(i), stat = stat2)
        end do

        redshift = z
        
    end subroutine calculate_mfbf_table

    subroutine calculate_mf_integral(cm, hm, retval, func, stat)
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters
        procedure(halo_function), optional :: func !! optional weight function

        real(dp), intent(out) ::  retval
        integer , intent(out), optional :: stat

        real(dp) :: lnMa, lnMb, m_scale, m_shift, m, wm, fm
        integer  :: i, stat2
        stat2 = 0

        if ( redshift <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( .not. ready ) stat2 = ERR_CALC_NOT_SETUP
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! mass integration is done in log scale, from M=Mmin to M=10^18 Msun
        lnMa    = hm%log_Mmin * LN_10
        lnMb    = log(1.0e+18) !! TODO: change to 5-sigma mass
        m_scale = 0.5*(lnMb - lnMa)
        m_shift = lnMa + m_scale

        retval = 0.0_dp
        do i = 1, nq
            wm = m_scale*wq(i)

            !! calculate halo mass-function
            fm = mftab(i)

            !! calculate the weight function
            if ( present(func) ) then
                m  = exp( m_scale*xq(i) + m_shift ) !! mass in Msun
                fm = fm * func(m, cm, hm)
            end if 

            retval = retval + wm * fm

        end do
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

    end subroutine calculate_mf_integral

    subroutine calculate_mfbf_integral(cm, hm, retval, func, stat)
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters
        procedure(halo_function), optional :: func !! optional weight function

        real(dp), intent(out) ::  retval
        integer , intent(out), optional :: stat

        real(dp) :: lnMa, lnMb, m_scale, m_shift, m, wm, fm
        integer  :: i, stat2
        stat2 = 0

        if ( redshift <= -1. ) stat2 = ERR_INVALID_VALUE_Z
        if ( .not. ready ) stat2 = ERR_CALC_NOT_SETUP
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        !! mass integration is done in log scale, from M=Mmin to M=10^18 Msun
        lnMa    = hm%log_Mmin * LN_10
        lnMb    = log(1.0e+18) !! TODO: change to 5-sigma mass
        m_scale = 0.5*(lnMb - lnMa)
        m_shift = lnMa + m_scale

        retval = 0.0_dp
        do i = 1, nq
            wm = m_scale*wq(i)
            
            !! calculate halo mass-function
            fm = mftab(i) * bftab(i)
            
            !! calculate the weight function
            if ( present(func) ) then
                m  = exp( m_scale*xq(i) + m_shift ) !! mass in Msun
                fm = fm * func(m, cm, hm)
            end if 

            retval = retval + wm * fm

        end do
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

    end subroutine calculate_mfbf_integral

    !=========================================================================================================

    !>
    !! Calculate the halo density at redshift z, for halos of mass > Mmin.
    !!
    !! Parameters:
    !!  z: real         - Redshift
    !!  Delta: real     - Halo overdensity relative to mean background density
    !!  cm: cosmo_t     - Cosmology model parameteres 
    !!  hm: halomodel_t - Halo model parameteres 
    !!  retval: real    - Calculated halo density in Mpc^-3
    !!  stat: integer   - Status flag
    !!
    subroutine get_halo_density(cm, hm, retval, stat)
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) ::  retval
        integer , intent(out), optional :: stat

        call calculate_mf_integral(cm, hm, retval, stat = stat)
        
    end subroutine get_halo_density

    !>
    !! Calculate the average galaxy density at redshift z, for halos of mass > Mmin.
    !!
    !! Parameters:
    !!  z: real         - Redshift
    !!  Delta: real     - Halo overdensity relative to mean background density
    !!  cm: cosmo_t     - Cosmology model parameteres 
    !!  hm: halomodel_t - Halo model parameteres 
    !!  retval: real    - Calculated galaxy density in Mpc^-3
    !!  stat: integer   - Status flag
    !!
    subroutine get_average_galaxy_density(cm, hm, retval, stat)
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) ::  retval
        integer , intent(out), optional :: stat
    
        call calculate_mf_integral(cm, hm, retval, func = get_average_galaxy_count, stat = stat)

    end subroutine get_average_galaxy_density

    !>
    !! Calculate the average galaxy bias factor at redshift z, for halos of mass > Mmin.
    !!
    !! Parameters:
    !!  z: real         - Redshift
    !!  Delta: real     - Halo overdensity relative to mean background density
    !!  cm: cosmo_t     - Cosmology model parameteres 
    !!  hm: halomodel_t - Halo model parameteres 
    !!  retval: real    - Calculated galaxy bias factor in Mpc^-3
    !!  stat: integer   - Status flag
    !!
    subroutine get_average_galaxy_bias(cm, hm, retval, stat)
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) ::  retval
        integer , intent(out), optional :: stat

        real(dp) :: res1, res2
        integer  :: stat2
        
        call calculate_mfbf_integral(cm, hm, res1, func = get_average_galaxy_count, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        call calculate_mf_integral(cm, hm, res2, func = get_average_galaxy_count, stat = stat2)
        if ( present(stat) ) stat = stat2
        if ( stat2 .ne. 0 ) return

        retval = res1 / res2 !! average galaxy bias

    end subroutine get_average_galaxy_bias

    !>
    !! Calculate the average total galaxy count inside a halo.
    !!
    !! Parameters:
    !!  m : real - Halo mass in Msun
    !!  hm: halomodel_t - Halo model parameters
    !!
    !! Returns:
    !!  retval: real
    !!
    function get_average_galaxy_count(m, cm, hm) result(retval)
        real(dp), intent(in)  :: m !! halo mass in Msun
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters
        real(dp) :: retval

        call hm%get_galaxy_count(m, retval)

    end function get_average_galaxy_count

    subroutine calculate_galaxy_params_zavg(za, zb, nz, fz, cm, hm, Delta, ng, bg, stat)
        real(dp), intent(in) :: za, zb !! redshift integration limits
        integer , intent(in) :: nz     !! redshift integration rule size
        procedure(z_distrib) :: fz     !! redshift distribution
        real(dp), intent(in) :: Delta  !! halo overdensity
        class(halomodel_t), intent(in) :: hm !! halo model parameters
        class(cosmo_t)    , intent(in) :: cm !! cosmology parameters

        real(dp), intent(out) :: ng !! galaxy density in Mpc^-3
        real(dp), intent(out) :: bg !! galaxy bias
        integer , intent(out), optional :: stat

        real(dp), dimension(nq) :: mf_save, bf_save
        real(dp), dimension(nz) :: xz, wz
        real(dp) :: fz_area, z_weight, dvdz, tz, z_save
        integer  :: i, stat2
        stat2 = 0

        !! save the current mass function and bias tables
        do i = 1, nq
            mf_save(i) = mftab(i) 
            bf_save(i) = bftab(i)
        end do
        z_save = redshift

        !! generate z integration rule
        call generate_gaussleg(nz, xz, wz, stat = stat2, a = za, b = zb)
        if ( present(stat) ) stat = stat2
        if (stat2 .ne. 0) return
        
        !! calculate integrals over z
        ng = 0.0_dp
        bg = 0.0_dp
        fz_area = 0.0_dp
        do i = 1, nz

            !! calculate comoving volume element
            call calculate_comoving_distance(xz(i), cm, tz, dvdz = dvdz, stat = stat2)
            if (stat2 .ne. 0) exit

            !! calculate z weight
            z_weight = fz( xz(i) ) * dvdz
            fz_area  = fz_area + z_weight * wz(i)

            !! setup calculator for this redshift
            call calculate_mfbf_table(xz(i), Delta, cm, hm, stat = stat2)
            if (stat2 .ne. 0) exit

            !! calculate galaxy density
            call get_average_galaxy_density(cm, hm, tz, stat = stat2)
            if (stat2 .ne. 0) exit
            ng = ng + ( z_weight * tz )*wz(i)

            !! calculate galaxy bias
            call get_average_galaxy_bias(cm, hm, tz, stat = stat2)
            if (stat2 .ne. 0) exit
            bg = bg + ( z_weight * tz )*wz(i)

        end do
        if ( present(stat) ) stat = stat2
        if (stat2 .ne. 0) return

        !! average galaxy density
        ng = ng / fz_area

        !! average galaxy bias
        bg = bg / fz_area

        !! reset mass function and bias tables to save values
        do i = 1, nq
            mftab(i) = mf_save(i) 
            bftab(i) = bf_save(i)
        end do
        redshift = z_save

    end subroutine calculate_galaxy_params_zavg
    
end module halomodel_calculator