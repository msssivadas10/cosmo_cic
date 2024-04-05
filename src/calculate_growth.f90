module calculate_growth
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: generate_gaussleg
    use cosmology, only: cosmo_t, get_cosmology
    implicit none

    private

    ! Settings for redshift integration
    logical :: ready = .false.
    integer :: zInteg_N = 0
    real(dp), dimension(:), allocatable :: zInteg_X, zInteg_W
    
    public :: GC_init
    public :: GC_get_growth
    
contains

    !>
    !! Setup growth factor calculator.
    !!
    !! Parameters:
    !!  size : integer - Size of the redshift integration rule.
    !!  stat : integer - Status.
    !!
    subroutine GC_init(size, stat)
        integer, intent(in) :: size
        integer, intent(out), optional ::  stat

        if ( size < 2 ) then
            if ( present( stat ) ) stat = 1
            write(stderr,'(a)') 'error: GC_init - size must be > 2'
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

    end subroutine GC_init

    !>
    !! Calculate the linear growth and its log derivative.
    !!
    !! Parameters:
    !!  z     : real    - Redshift (must be greater than -1).
    !!  dplus : real    - Calculated value of growth factor.
    !!  fplus : real    - Calculated value of growth rate (optional).
    !!  stat  : integer - Status. 
    !!
    subroutine GC_get_growth(z, dplus, fplus, stat)
        real(dp), intent(in)  :: z !! redshift 
        
        real(dp), intent(out) :: dplus           !! growth factor
        real(dp), intent(out), optional :: fplus !! growth rate
        integer , intent(out), optional :: stat  
        
        real(dp) :: res, a, fa, wa, dlnfa, scale
        integer  :: i, nq
        type(cosmo_t) :: cm
        
        !! get the global cosmology model
        cm = get_cosmology()

        !! check if the calculator is setup properly
        if ( .not. ready ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: GC_get_growth - growth calculator is not setup :('
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
            write(stderr,'(a)') 'error: GC_get_growth - redshift z must be > -1'
            return
        end if

        nq    = zInteg_N  !! size of the integration rule
        scale = 0.5_dp / ( z + 1._dp ) 
        res   = 0.0_dp
        do i = 1, nq
            !! get i-the node and weight
            a  = scale * ( zInteg_X(i) + 1. ) !! scale factor
            wa = zInteg_W(i)

            !! calculating integrand
            call cm%get_E2(1/a-1., fa) !! E**2(z)
            fa  = 1./ ( a * sqrt( fa ) )**3
            res = res + wa * fa
        end do
        res = res * scale

        !! calculation of hubble parameter function E(z)
        call cm%get_E2(z, fa, dlnfa)

        !! calculating linear growth factor
        dplus = sqrt(fa) * res

        !! calculating linear growth rate
        if ( present(fplus) ) fplus = (z + 1.)**2 / fa / dplus - 0.5*dlnfa

        if ( present(stat) ) stat = 0

    end subroutine GC_get_growth
    
end module calculate_growth