module calculate_dist
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: generate_gaussleg, EPS, C_KMPS, PI
    use cosmology, only: cosmo_t, get_cosmology
    implicit none

    private

    ! Settings for redshift integration
    logical :: ready = .false.
    integer :: zInteg_N = 0
    real(dp), dimension(:), allocatable :: zInteg_X, zInteg_W
    
    public :: DC_init
    public :: DC_get_distance
    
contains

    !>
    !! Setup comoving distance calculator.
    !!
    !! Parameters:
    !!  size : integer - Size of the redshift integration rule.
    !!  stat : integer - Status.
    !!
    subroutine DC_init(size, stat)
        integer, intent(in) :: size
        integer, intent(out), optional ::  stat

        if ( size < 2 ) then
            if ( present( stat ) ) stat = 1
            write(stderr,'(a)') 'error: DC_init - size must be > 2'
            return
        end if

        !! allocate nodes array
        if ( allocated( zInteg_X ) ) deallocate( zInteg_X )
        allocate( zInteg_X( size ) )
        
        !! allocate weights array
        if ( allocated( zInteg_W ) ) deallocate( zInteg_W )
        allocate( zInteg_W( size ) )
        
        !! generate rule
        call generate_gaussleg(size, zInteg_X, zInteg_W, stat = stat)
        zInteg_N = size
        ready    = .true.

    end subroutine DC_init

    !>
    !! Calculate comoving distance at redshift z.
    !!
    !! Parameters:
    !!  z    : real    - Redshift (must be greater than -1).
    !!  r    : real    - Comoving distance in Mpc.
    !!  dvdz : real    - Comoving volume element in Mpc^3.
    !!  stat : integer - Status. 
    !!
    subroutine DC_get_distance(z, r, dvdz, stat) 
        real(dp), intent(in)  :: z !! redshift 
        
        real(dp), intent(out) :: r              !! distance in Mpc
        real(dp), intent(out), optional :: dvdz !! volume element in Mpc^3
        integer , intent(out), optional :: stat !! integration rule
        
        real(dp) :: res, drdz
        real(dp) :: ut_dist
        real(dp) :: a, fa, scale, shift, wa
        integer  :: i, nq
        type(cosmo_t) :: cm

        !! get the global cosmology model
        cm = get_cosmology()

        !! check if the calculator is setup properly
        if ( .not. ready ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: DC_get_distance - distance calculator is not setup :('
            return
        end if
        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: GC_get_growth - cosmology model is not initialised'
            return
        end if
        !! check if redshift value is correct
        if ( z <= -1. ) then !! invalid value for redshift
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: DC_get_distance - redshift z must be > -1'
            return
        end if

        r  = 0.0_dp; dvdz = 0.0_dp
        if ( abs( z ) < EPS ) return !! z = 0
        
        ut_dist = C_KMPS / cm%H0 !! factor to convert the value to Mpc units
        nq      = zInteg_N       !! size of the integration rule
        scale   = 0.5_dp * z / ( z + 1._dp ) 
        shift   = 1. - scale
        res     = 0.0_dp
        do i = 1, nq
            !! get i-the node and weight
            a  = scale * zInteg_X(i) + shift !! scale factor
            wa = zInteg_W(i)

            !! calculating integrand
            call cm%get_E2(1/a-1., fa) 
            fa  = 1./ a**2 / sqrt( fa )
            res = res + wa * fa
        end do
        res = res * scale
        r   = res * ut_dist !! comoving distance in Mpc

        !! calculate comoving volume element, dvdz
        if ( present(dvdz) ) then 
            !! distance derivative w.r.to z: 1/E(a)
            call cm%get_E2(z, fa)
            drdz = 1. / sqrt( fa ) * ut_dist
            dvdz = 4*PI * r**2 * drdz !! volume element
        end if

        if ( present(stat) ) stat = 0
        
    end subroutine DC_get_distance
    
end module calculate_dist