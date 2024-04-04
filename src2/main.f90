program main
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use cosmology, only: cosmo_t
    use transfer_eh, only: tf_eisenstein98_init, tf_eisenstein98
    use calculate_growth, only: GC_init, GC_get_growth
    use calculate_specmom, only: SMC_init, SMC_get_specmom
    implicit none
    
    character(len=128) :: input_file = 'params.ini'
    integer :: ierror = 0

    !! global cosmology model (non initialised)
    type(cosmo_t) :: glob_cm = cosmo_t(H0 = 0._dp, Om0 = 0._dp, Ob0 = 0._dp, flat = .false.)
    character(len=16) :: ps_model = 'eisenstein98_zb', mf_model = 'tinker08', bf_model = 'tinker10'
    integer  :: zInteg_size = 128, kInteg_size = 512, mInteg_size = 512
    real(dp) :: kInteg_lnka = -18.420680743952367_dp, kInteg_lnkb = 18.420680743952367_dp
    real(dp) :: ps_norm = 1._dp

    !! get name of the input file from command line: 1-st arg
    call get_command_argument(1, input_file)
    if ( input_file == '' ) then
        write (*,'(a)') 'error: no input file'
        return
    end if 

    !! get parameters from input file
    call load_params()

    !! initialise calculator
    if ( ierror == 0 ) call calc_init()
    if ( ierror == 0 ) call display_settings()

    contains

        subroutine ps_calculate(cm, k, pk, args, stat)
            class(cosmo_t), intent(in) :: cm
            real(dp), intent(in)       :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in), optional :: args(:) !! additional arguments
            
            real(dp), intent(out) :: pk
            integer , intent(out), optional :: stat

            real(dp) :: z = 0._dp
            if ( present(args) ) z = args(1)
            
            call tf_eisenstein98(cm, k, z, pk, stat = stat)
            pk = pk**2 * k**cm%ns
        end subroutine ps_calculate

        subroutine get_matter_powerspec(k, z, pk, tk, dlnpk, stat)
            real(dp), intent(in)       :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in)       :: z !! redshift 
            
            real(dp), intent(out) :: pk
            real(dp), intent(out), optional :: tk, dlnpk
            integer , intent(out), optional :: stat

            call tf_eisenstein98(glob_cm, k, z, pk, stat = stat)
            if ( present(tk) ) tk = pk
            if ( present(dlnpk) ) dlnpk = 2*dlnpk + glob_cm%ns
            pk = pk**2 * k**glob_cm%ns * ps_norm
            
        end subroutine get_matter_powerspec

        subroutine load_params()
            character(len=128) :: buffer, label
            integer, parameter :: fh = 15 
            integer :: pos, ios = 0, line = 0
            
            write (*,'(a)') 'info: loading input parameters from file `'//trim(input_file)//'`'
            open(fh, file = input_file)
            do while ( ios == 0 )
                read(fh, '(a)', iostat = ios) buffer 
                if ( ios == 0 ) then
                    line = line + 1

                    !! remove comments, marked with charecters `#` or `;`
                    pos = scan( buffer, '#;' )
                    if ( pos > 0 ) buffer = buffer(1:pos-1)

                    !! skip empty lines
                    if ( len( trim(buffer) ) < 1 ) cycle

                    !! get key-value pairs, seperated by `=` 
                    pos = scan(buffer, '=') 
                    if ( pos < 1 ) then
                        ierror = 1
                        exit
                    end if
                    label  = buffer(1:pos-1) 
                    buffer = buffer(pos+1:)

                    select case ( label )
                    case ( 'Hubble' )
                        read(buffer, *, iostat = ios) glob_cm%H0
                    case ( 'OmegaMatter' )
                        read(buffer, *, iostat = ios) glob_cm%Om0
                    case ( 'OmegaBaryon' )
                        read(buffer, *, iostat = ios) glob_cm%Ob0
                    case ( 'OmegaNeutrino' )
                        read(buffer, *, iostat = ios) glob_cm%Onu0
                    case ( 'OmegaDarkEnergy' )
                        read(buffer, *, iostat = ios) glob_cm%Ode0
                    case ( 'NumNeutrino' )
                        read(buffer, *, iostat = ios) glob_cm%Nnu
                    case ( 'PowerIndex' )
                        read(buffer, *, iostat = ios) glob_cm%ns
                    case ( 'Sigma8' )
                        read(buffer, *, iostat = ios) glob_cm%sigma8
                    case ( 'w0' )
                        read(buffer, *, iostat = ios) glob_cm%w0
                    case ( 'wa' )
                        read(buffer, *, iostat = ios) glob_cm%wa
                    end select
                end if
            end do
        end subroutine load_params

        subroutine calc_init()
            real(dp) :: r, args(1)
            
            !! initialise cosmology
            ierror = glob_cm%init()
            if ( ierror .ne. 0 ) then
                write (*,'(a)') 'error: failed to initialise cosmology model'
                return
            end if

            !! initialise grwoth calculator
            write (*,'(a)') 'info: setting up growth calculator'
            call GC_init(zInteg_size, stat = ierror)
            if ( ierror .ne. 0 ) then
                write (*,'(a)') 'error: failed to setup growth calculator'
                return
            end if

            !! initialise power spectrum
            write (*,'(a)') 'info: setting up power spectrum calculator'
            select case ( ps_model )
            case ( 'eisenstein98_zb' )
                call tf_eisenstein98_init(glob_cm, version = 0, stat = ierror)
            case ( 'eisenstein98_bao' )
                call tf_eisenstein98_init(glob_cm, version = 1, stat = ierror)
            case ( 'eisenstein98_nu' )
                call tf_eisenstein98_init(glob_cm, version = 2, stat = ierror)
            case default
                ierror = 1
                write (*,'(a)') 'error: unknown power spectrum model, `'//ps_model//'`'
            end select
            if ( ierror .ne. 0 ) then
                write (*,'(a)') 'error: failed to setup power spectrum calculator'
                return
            end if

            !! initialise variance calculator
            write (*,'(a)') 'info: setting up variance calculator'
            call SMC_init(kInteg_size, lnka = kInteg_lnka, lnkb = kInteg_lnkb, stat = ierror)
            if ( ierror .ne. 0 ) then
                write (*,'(a)') 'error: failed to setup variance calculator'
                return
            end if
            
            !! normalising power spectrum
            write (*,'(a)') 'info: normalising power spectrum'
            r       = 800. / glob_cm%H0
            args(1) = 0._dp
            call SMC_get_specmom(glob_cm, ps_calculate, r, 0, ps_norm, args = args, stat = ierror)
            ps_norm = glob_cm%sigma8**2 / ps_norm
            if ( ierror .ne. 0 ) then
                write (*,'(a)') 'error: failed to normalise power spectrum'
                return
            end if

            write (*,'(a)') 'info: calculator initialised successfully :)'
        end subroutine calc_init

        subroutine display_settings()
            write (*,'(a,f8.3,a)')    '- Hubble parameter       : ', glob_cm%H0    , ' km/sec/Mpc'
            write (*,'(a,f8.3,a)')    '- Total matter density   : ', glob_cm%Om0   , ' '
            write (*,'(a,f8.3,a)')    '- Baryon density         : ', glob_cm%Ob0   , ' '
            write (*,'(a,f8.3,a)')    '- Neutrino density       : ', glob_cm%Onu0  , ' '
            write (*,'(a,f8.3,a)')    '- Curvature density      : ', glob_cm%Ok0   , ' '
            write (*,'(a,f8.3,a)')    '- Dark-energy density    : ', glob_cm%Ode0  , ' '
            write (*,'(a,f8.3,a)')    '- Neutrino number        : ', glob_cm%Nnu   , ' '
            write (*,'(a,f8.3,a)')    '- Power spectrum index   : ', glob_cm%ns    , ' '
            write (*,'(a,f8.3,a)')    '- sigma-8                : ', glob_cm%sigma8, ' '
            write (*,'(a,f8.3,a)')    '- CMB temperature        : ', glob_cm%Tcmb0 , ' K'
            write (*,'(a,f8.3,a)')    '- Dark-energy w0         : ', glob_cm%w0    , ' '
            write (*,'(a,f8.3,a)')    '- Dark-energy wa         : ', glob_cm%wa    , ' '
            write (*,'(a,i8,a)')      '- z integration rule size: ', zInteg_size   , ' '
            write (*,'(a,i8,a)')      '- k integration rule size: ', kInteg_size   , ' '
            write (*,'(a,i8,a)')      '- m integration rule size: ', mInteg_size   , ' '
            write (*,'(a,2(f8.3,a))') '- k integration range    : ', kInteg_lnka   , ' to ', kInteg_lnkb
            write (*,'(a,a)')         '- Power spectrum model   : ', ps_model
            write (*,'(a,a)')         '- Halo massfunction model: ', mf_model
            write (*,'(a,a)')         '- Halo bias model        : ', bf_model
            write (*,'(a,e12.3)')     '- Power spectrum norm    : ', ps_norm
        end subroutine display_settings
    
end program main