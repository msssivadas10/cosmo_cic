module calculate_specmom
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use utils, only: generate_gaussleg, PI
    implicit none

    private

    interface
        !! Interface to power spectrum calculator
        subroutine ps_calculate(k, pk, args, stat)
            use iso_fortran_env, only: dp => real64
            real(dp), intent(in)            :: k       !! wavenumber in 1/Mpc unit 
            real(dp), intent(in), optional  :: args(:) !! additional arguments
            real(dp), intent(out)           :: pk
            integer , intent(out), optional :: stat
        end subroutine ps_calculate
    end interface

    ! Window function models for varience calculations
    integer, parameter :: WIN_TOPHAT = 1 !! Spherical top-hat window
    integer, parameter :: WIN_GAUSS  = 2 !! Gaussian window
    integer, parameter :: WIN_SHARPK = 3 !! Sharp-k window

    ! Settings for redshift integration
    integer  :: kInteg_N = 0
    real(dp), dimension(:), allocatable :: kInteg_X, kInteg_W
    real(dp) :: kInteg_lnka = -18.420680743952367_dp !! Lower limit of ln(k) integration
    real(dp) :: kInteg_lnkb =  18.420680743952367_dp !! Upper limit of ln(k) integration
    
    logical :: ready  = .false.

    public :: SMC_init
    public :: SMC_get_specmom
    
contains
    
    !>
    !! Setup spectral moment calculator.
    !!
    !! Parameters:
    !!  size : integer - Size of the wavenumber integration rule.
    !!  lnka : real    - Lower limit of integration.
    !!  lnkb : real    - Upper limit of integration.
    !!  stat : integer - Status.
    !!
    subroutine SMC_init(size, lnka, lnkb, stat)
        integer , intent(in) :: size
        real(dp), intent(in) , optional :: lnka, lnkb 
        integer , intent(out), optional ::  stat
        real(dp) :: lnkt

        if ( size < 2 ) then
            if ( present( stat ) ) stat = 1
            write(stderr,'(a)') 'error: SMC_init - size must be > 2'
            return
        end if

        !! set integration limits
        if ( present(lnka) ) kInteg_lnka = lnka
        if ( present(lnkb) ) kInteg_lnkb = lnkb
        if ( kInteg_lnka > kInteg_lnkb ) then
            lnkt        = kInteg_lnka
            kInteg_lnka = kInteg_lnkb 
            kInteg_lnkb = lnkt
        end if

        !! allocate nodes array
        if ( allocated( kInteg_X ) ) deallocate( kInteg_X )
        allocate( kInteg_X( size ) )
        
        !! allocate weights array
        if ( allocated( kInteg_W ) ) deallocate( kInteg_W )
        allocate( kInteg_W( size ) )

        !! generate rule
        call generate_gaussleg(size, kInteg_X, kInteg_W, stat = stat)
        kInteg_N = size
        ready    = .true.

    end subroutine SMC_init

    !>
    !! Calculate the spectral moment, smoothing over a scale r Mpc, using the tabulated 
    !! power spectrum values. Need proper setup for getting correct values.
    !!
    !! Parameters:
    !!  cm   : cosmo_t   - Cosmology parameters
    !!  ps   : procedure - Subroutine to calculate power spectrum. 
    !!  r    : real      - Smoothing scale in Mpc
    !!  j    : integer   - Order of the moment (j=0 gives the matter variance)
    !!  s    : real      - Calculated variance
    !!  dlns : real      - Calculatetd 1-st log-derivative (optional)
    !!  args : real      - Additional arguments to the power spectrum
    !!  stat : integer   - Status.
    !!
    subroutine SMC_get_specmom(ps, r, j, s, dlns, args, stat)
        procedure(ps_calculate)    :: ps 
        real(dp), intent(in)       :: r 
        integer , intent(in)       :: j
        real(dp), intent(in), optional :: args(:) !! additional arguments

        real(dp), intent(out) :: s
        real(dp), intent(out), optional :: dlns
        integer , intent(out), optional :: stat

        real(dp) :: k, kr, f2, f3, res1, res2, wk, pk, lnka, lnkb, scale, shift
        integer  :: i, nq, stat2 = 0, selected_win = WIN_TOPHAT

        !! check if the calculator is setup properly
        if ( .not. ready ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: SMC_get_specmom - growth calculator is not setup :('
            return
        end if
        !! check if r value is correct
        if ( r <= 0. ) then 
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: SMC_get_specmom - scale r must be positive'
            return
        end if
        !! check if j value is correct
        if ( j < 0 ) then 
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: SMC_get_specmom - j must be 0 or positive integer'
            return
        end if

        !! scale nodes and weight to the integration limit
        lnka  = kInteg_lnka
        lnkb  = kInteg_lnkb
        if ( selected_win == WIN_SHARPK ) lnkb = -log(r)
        scale = 0.5*( lnkb - lnka )
        shift = kInteg_lnka + scale
        nq    = kInteg_N
        res1  = 0.0_dp
        res2  = 0.0_dp
        do i  = 1, nq
            !! get i-the node and weight
            k  = exp( scale * kInteg_X(i) + shift ) !! wavenumber in Mpc^-1
            wk = scale * kInteg_W(i)
            kr = k*r 

            !! calculating weighted power spectrum
            call ps(k, pk, args = args, stat = stat2)
            if ( stat2 .ne. 0 ) then
                if ( present(stat) ) stat = stat2
                write(stderr,'(a)') 'error: SMC_get_specmom - failed to calculate power spectrum'
                return
            end if
            pk = pk * k**(3 + 2*j)

            !! calculating window function, w(kr)
            if ( selected_win == WIN_GAUSS ) then 
                !! gaussian window function
                f3 = exp(-0.5*kr**2)
            else 
                !! spherical tophat window function
                f3 = 3*( sin(kr) - kr*cos(kr) ) / kr**3
            end if
            f2   = pk * f3               !! p(k)*w(kr) 
            res1 = res1 + wk * ( f2*f3 ) !! sigma^2 integration

            ! no need to calculate derivatives, if not needed or sharp-k filter is used...
            if ( (.not. present(dlns)) .or. ( selected_win == WIN_SHARPK ) ) cycle !

            !! calculating window function 1-st derivative, dwdx
            if ( selected_win == WIN_GAUSS ) then 
                !! gaussian window function
                f3 = -kr*exp(-0.5*kr**2)
            else 
                !! spherical tophat window function
                f3 = 3*( ( kr**2 - 3. )*sin(kr) + 3*kr*cos(kr) ) / kr**4
            end if
            f3   = 2*f3 * k              !! 2*dwdx*k := 2*dwdr
            res2 = res2 + wk * ( f2*f3 ) !! ds2dr integration
        end do

        s = res1 / (2*PI**2)
        if ( present(stat) ) stat = 0
        if ( .not. present(dlns) ) return

        !! 1-st derivative for sharp-k filter
        if ( selected_win == WIN_SHARPK ) then
            k = 1./r
            call ps(k, pk, args = args, stat = stat2)
            if ( stat2 .ne. 0 ) then
                if ( present(stat) ) stat = stat2
                write(stderr,'(a)') 'error: SMC_get_specmom - failed to calculate power spectrum'
                return
            end if
            res2 = -pk * k**(4 + 2*j)
        end if

        !! 1-st log-derivative, dlnsdlnr
        res1 = r / res1
        res2 = res1 * res2 
        dlns = res2
    
    end subroutine SMC_get_specmom

end module calculate_specmom