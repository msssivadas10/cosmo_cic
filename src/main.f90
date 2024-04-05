program main
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use load_options, only: options_t, load_options_from_file
    use utils, only: PI, RHO_CRIT0, DELTA_SC
    use cosmology, only: cosmo_t, get_cosmology, set_cosmology
    use calculate_dist,    only: DC_init, DC_get_distance
    use calculate_growth,  only: GC_init, GC_get_growth
    use calculate_specmom, only: SMC_init, SMC_get_specmom
    use calculate_mfbf, only: calculate_massfunc_bias
    use mod_mtf_eh, only: tf_eisenstein98_init, tf_eisenstein98
    use mod_hmf_tinker08, only: mf_tinker08_init, mf_tinker08
    use mod_hbf_tinker10, only: bf_tinker10_init, bf_tinker10
    implicit none
    
    integer, parameter :: PS_EH_FAMILY   = 10
    integer, parameter :: PS_BBKS_FAMILY = 11

    character(len=128) :: input_file
    integer  :: ierror = 0
    integer  :: ps_family = PS_EH_FAMILY
    real(dp) :: ps_norm = 1._dp
    type(options_t) :: opts

    !! set default values
    opts%Delta     = 200._dp
    opts%model_mps = 'eisenstein98_zb'
    opts%model_hmf = 'tinker08'
    opts%model_hbf = 'tinker10'
    opts%settings_vi_lnka = -18.420680743952367_dp
    opts%settings_vi_lnkb =  18.420680743952367_dp
    opts%settings_vi_size = 512
    opts%settings_gi_size = 128
    opts%settings_di_size = 128

    !! get name of the input file from command line: 1-st arg
    call get_command_argument(1, input_file)

    call initialize_calculator()
    call generate_redshift_data()

contains

    !>
    !! Calculate the un-normalised power spectrum
    !!
    subroutine ps_calculate(k, pk, args, stat)
        real(dp), intent(in)            :: k       !! wavenumber in 1/Mpc unit 
        real(dp), intent(in), optional  :: args(:) !! additional arguments
        real(dp), intent(out)           :: pk
        integer , intent(out), optional :: stat
        real(dp) :: z = 0._dp
        type(cosmo_t) :: cm

        !! get cosmology
        cm = get_cosmology()
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            return
        end if
        if ( present(args) ) z = args(1)

        !! calculate transfer function
        call tf_eisenstein98(k, z, pk, stat = stat)

        !! calculate power spectrum
        pk = pk**2 * k**cm%ns

    end subroutine ps_calculate

    !>
    !! Calculate the normalised power spectrum
    !!
    subroutine get_matter_powerspec(k, z, pk, tk, dlnpk, stat)
        real(dp), intent(in)            :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in)            :: z !! redshift 
        real(dp), intent(out)           :: pk
        real(dp), intent(out), optional :: tk, dlnpk
        integer , intent(out), optional :: stat
        type(cosmo_t) :: cm

        !! get cosmology
        cm = get_cosmology()
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            return
        end if

        !! calculate transfer function
        call tf_eisenstein98(k, z, pk, dlntk = dlnpk, stat = stat)
        if ( present(tk) ) tk = pk

        !! calculate power spectrum
        pk = pk**2 * k**cm%ns * ps_norm
        if ( present(dlnpk) ) dlnpk = 2*dlnpk + cm%ns
        
    end subroutine get_matter_powerspec

    !>
    !! Calculate massfunction and bias
    !!
    subroutine get_halo_mfbf(m, z, dndlnm, bm, r, fs, s, dlns, stat)
        real(dp), intent(in)  :: m, z
        real(dp), intent(out) :: dndlnm, bm 
        real(dp), intent(out), optional :: r, fs, s, dlns
        integer , intent(out), optional :: stat
        real(dp) :: args(3), rho_m, rho_h, rl, sig, dlnsdlnr
        type(cosmo_t) :: cm

        args(1) = z
        args(2) = opts%Delta 

        !! get cosmology
        cm = get_cosmology()
        if ( .not. cm%is_ready() ) then
            ierror = 1
            if ( present(stat) ) stat = 1
            return
        end if

        !! density
        rho_m = cm%Om0 * RHO_CRIT0 * (0.01*cm%H0)**2 !! Msun/Mpc^3 
        rho_h = rho_m
        
        !! calculate radius
        rl = ( 0.75*m / PI / rho_h)**(1./3.) ! Mpc
        if ( present(r) ) r = rl
        
        !! calculate variance
        call SMC_get_specmom(ps_calculate, rl, 0, sig, dlns = dlnsdlnr, args = args, stat = ierror)
        if ( ierror .ne. 0 ) then
            if ( present(stat) ) stat = 1
            write (stderr,'(a)') 'error: failed to calculate variance'
            return
        end if
        sig = sqrt(sig*ps_norm)
        if ( present(s) ) s = sig 
        if ( present(dlns) ) dlns = dlnsdlnr / 6.

        !! calculate mass function and bias
        call calculate_massfunc_bias(mf_tinker08, bf_tinker10, m, sig, dlnsdlnr, dndlnm, bm, &
                            &        fs = fs, mf_args = args, bf_args = args, stat = ierror)
        if ( ierror .ne. 0 ) then
            if ( present(stat) ) stat = 1
            write (stderr,'(a)') 'error: failed to calculate mass function and bias'
            return
        end if

    end subroutine get_halo_mfbf

    !>
    !! Load parameters and initialise the calculator
    !!
    subroutine initialize_calculator()
        real(dp) :: lnka, lnkb, r, args(1)
        type(cosmo_t) :: cm

        !! load parameters from file
        if ( input_file == '' ) then
            write (stderr,'(a)') 'error: no input file'
            ierror = 1
            return
        end if 
        call load_options_from_file(input_file, opts, stat = ierror)
        if ( ierror .ne. 0 ) return

        write (stderr,'(a)') 'loaded options: '
        write (stderr,'(a,a)')       ' - model_mps              : ', opts%model_mps
        write (stderr,'(a,a)')       ' - model_hmf              : ', opts%model_hmf
        write (stderr,'(a,a)')       ' - model_hbf              : ', opts%model_hbf
        write (stderr,'(a,a)')       ' - output_file_redshift   : ', opts%output_file_redshift
        write (stderr,'(a,a)')       ' - output_file_powerspec  : ', opts%output_file_powerspec
        write (stderr,'(a,a)')       ' - output_file_halos      : ', opts%output_file_halos
        write (stderr,'(a,i4)')      ' - cosmo_flat             : ', opts%cosmo_flat
        write (stderr,'(a,i4)')      ' - include_nu             : ', opts%include_nu
        write (stderr,'(a,i4)')      ' - settings_gi_size       : ', opts%settings_gi_size
        write (stderr,'(a,i4)')      ' - settings_di_size       : ', opts%settings_di_size
        write (stderr,'(a,i4)')      ' - settings_vi_size       : ', opts%settings_vi_size
        write (stderr,'(a,f8.3)')    ' - settings_vi_lnka       : ', opts%settings_vi_lnka
        write (stderr,'(a,f8.3)')    ' - settings_vi_lnkb       : ', opts%settings_vi_lnkb
        write (stderr,'(a,f8.3)')    ' - cosmo_Hubble           : ', opts%cosmo_H0
        write (stderr,'(a,f8.3)')    ' - cosmo_Om0              : ', opts%cosmo_Om0
        write (stderr,'(a,f8.3)')    ' - cosmo_Ob0              : ', opts%cosmo_Ob0
        write (stderr,'(a,f8.3)')    ' - cosmo_Ode0             : ', opts%cosmo_Ode0
        write (stderr,'(a,f8.3)')    ' - cosmo_ns               : ', opts%cosmo_ns
        write (stderr,'(a,f8.3)')    ' - cosmo_sigma8           : ', opts%cosmo_sigma8
        write (stderr,'(a,f8.3)')    ' - cosmo_w0               : ', opts%cosmo_w0
        write (stderr,'(a,f8.3)')    ' - cosmo_wa               : ', opts%cosmo_wa
        write (stderr,'(a,f8.3)')    ' - Delta                  : ', opts%Delta
        write (stderr,'(a,3(f8.3))') ' - z_range                : ', opts%z_range
        write (stderr,'(a,3(f8.3))') ' - k_range                : ', opts%k_range
        write (stderr,'(a,3(f8.3))') ' - m_range                : ', opts%m_range
        
        !! initialise cosmology
        cm%H0         = opts%cosmo_H0
        cm%Om0        = opts%cosmo_Om0
        cm%Ob0        = opts%cosmo_Ob0
        cm%Ode0       = opts%cosmo_Ode0
        cm%ns         = opts%cosmo_ns
        cm%sigma8     = opts%cosmo_sigma8
        cm%w0         = opts%cosmo_w0
        cm%wa         = opts%cosmo_wa
        cm%flat       = ( opts%cosmo_flat == 1 )
        cm%include_nu = ( opts%include_nu == 1 )
        call set_cosmology(cm, stat = ierror)
        if ( ierror .ne. 0 ) then
            write (stderr, '(a)') 'error: failed to set global cosmology'
            return
        end if
        write (stderr, '(a)') 'info: successfully set global cosmology'

        !! initialise growth calculator
        write (stderr,'(a)') 'info: setting up growth calculator'
        call GC_init(opts%settings_gi_size, stat = ierror)
        if ( ierror .ne. 0 ) then
            write (stderr,'(a)') 'error: failed to setup growth calculator'
            return
        end if
        write (stderr,'(a)') 'info: successfully setup growth calculator'

        !! initialise distance calculator
        write (stderr,'(a)') 'info: setting up distance calculator'
        call DC_init(opts%settings_di_size, stat = ierror)
        if ( ierror .ne. 0 ) then
            write (stderr,'(a)') 'error: failed to setup distance calculator'
            return
        end if
        write (stderr,'(a)') 'info: successfully setup distance calculator'

        !! initialise power spectrum
        write (stderr,'(a)') 'info: setting up power spectrum calculator'
        select case ( opts%model_mps )
        case ( 'eisenstein98_zb' )
            call tf_eisenstein98_init(version = 0, stat = ierror)
            ps_family = PS_EH_FAMILY
        case ( 'eisenstein98_bao' )
            call tf_eisenstein98_init(version = 1, stat = ierror)
            ps_family = PS_EH_FAMILY
        case ( 'eisenstein98_nu' )
            call tf_eisenstein98_init(version = 2, stat = ierror)
            ps_family = PS_EH_FAMILY
        case default
            ierror = 1
            write (stderr,'(a)') 'error: unknown power spectrum model - '//opts%model_mps//' '
        end select
        if ( ierror .ne. 0 ) then
            write (stderr,'(a)') 'error: failed to setup power spectrum calculator'
            return
        end if
        write (stderr,'(a)') 'info: sucessfully set power spectrum model - '//opts%model_mps//' '

        !! initialise variance calculator
        write (stderr,'(a)') 'info: setting up variance calculator'
        lnka = opts%settings_vi_lnka
        lnkb = opts%settings_vi_lnkb
        call SMC_init(opts%settings_vi_size, lnka = lnka, lnkb = lnkb, stat = ierror)
        if ( ierror .ne. 0 ) then
            write (stderr,'(a)') 'error: failed to setup variance calculator'
            return
        end if
        write (stderr,'(a)') 'info: successfully setup variance calculator'
        
        !! normalising power spectrum
        write (stderr,'(a)') 'info: normalising power spectrum'
        r       = 800. / cm%H0
        args(1) = 0._dp
        call SMC_get_specmom(ps_calculate, r, 0, ps_norm, args = args, stat = ierror)
        if ( ierror .ne. 0 ) then
            write (stderr,'(a)') 'error: failed to normalise power spectrum'
            return
        end if
        ps_norm = cm%sigma8**2 / ps_norm
        write (stderr,'(a)') 'info: successfully normalised power spectrum'
        write (stderr,'(a, e12.3)') 'info: normalisation factor = ', ps_norm

        !! initialising mass function
        write (stderr,'(a)') 'info: setting up massfunction'
        args(1) = opts%Delta
        call mf_tinker08_init(args)

        !! initialising bias
        write (stderr,'(a)') 'info: setting up bias'
        call bf_tinker10_init(args)

        write (*,'(a)') 'info: calculator initialised successfully :)'
    end subroutine initialize_calculator

    !>
    !! Calculate and save growth factor and comoving distance
    !!
    subroutine generate_redshift_data()
        integer, parameter :: fo = 10
        real(dp) :: data(5), z
        integer  :: i
        logical  :: save_zdata = .true.

        if ( ierror .ne. 0 ) return !! not initialised 
        if ( len( trim(opts%output_file_redshift) ) == 0 ) save_zdata = .false.

        write (stderr, '(a)') 'info: generating redshift data'

        !! open file
        if ( save_zdata ) then
            open(fo, file = opts%output_file_redshift)
            write(fo, '(a)') '# Col 1: Redshift' 
            write(fo, '(a)') '# Col 2: Comoving distance in Gpc' 
            write(fo, '(a)') '# Col 3: Comoving volume element in Gpc^3' 
            write(fo, '(a)') '# Col 4: Linear growth factor' 
            write(fo, '(a)') '# Col 4: Linear growth rate' 
        end if 
        
        !! generate data
        z = opts%z_range(1)
        do while ( z <= opts%z_range(2) )

            if ( save_zdata ) then
                data(1) = z

                !! calculate distance
                call DC_get_distance(z, data(2), dvdz = data(3), stat = ierror)
                if ( ierror .ne. 0 ) then
                    write (stderr, '(a,f0.3)') 'error: while calculating distance at z = ', z
                    return 
                end if
                data(2) = data(2) * 1E-03 !! in Gpc
                data(3) = data(3) * 1E-09 !! in Gpc^3

                !! calculate growth factor
                call GC_get_growth(z, data(4), fplus = data(5), stat = ierror)
                if ( ierror .ne. 0 ) then
                    write (stderr, '(a,f0.3)') 'error: while calculating growth at z = ', z
                    return 
                end if
                
                !! write to file
                write(fo, '(5(f12.6,a))') ( data(i), ', ', i = 1, 4 ), data(5)
            end if

            !! calculate power spectrum
            call generate_power_data(z)

            !! calculate mass function
            call generate_mffbf_data(z)

            if ( abs( opts%z_range(3) ) < 1e-08 ) exit !! zero-step size
            z = z + opts%z_range(3)
        end do
        if ( save_zdata ) close(fo)

        write (stderr, '(a)') 'info: successfully generated redshift data'

    end subroutine generate_redshift_data

    !>
    !! Calculate and save linear power spectrum
    !!
    subroutine generate_power_data(z)
        real(dp), intent(in) :: z
        integer, parameter  :: fo = 11
        character(len=32)   :: zstring
        character(len=128)  :: file
        integer  :: i, pos1, pos2, pos3
        real(dp) :: data(4), k, lnk

        file = opts%output_file_powerspec
        if ( ierror .ne. 0 ) return !! not initialised 
        if ( len( trim(file) ) == 0 ) return

        pos1 = scan(file, '{')
        pos2 = scan(file, '}')
        if ( ( ( pos1 == 0 ) .and. ( pos2 .ne. 0 ) ) .or. ( ( pos1 .ne. 0 ) .and. ( pos2 == 0 ) ) ) then
            ierror = 1
            write (stderr, '(a)') 'error: incorrect file name - '//file
            return
        else if ( ( pos1 .ne. 0 ) .or. ( pos2 .ne. 0 ) ) then 
            write(zstring, '('//file(pos1+1:pos2-1)//')') z
            pos3 = scan(zstring, '0123456789')
            file = file(1:pos1-1)//trim(zstring(pos3:))//file(pos2+1:)
        end if

        if ( ierror .ne. 0 ) return !! not initialised 

        !! open file
        open(fo, file = file)
        write(fo, '(a)') '# Col 1: Wavenumber in Mpc^-1' 
        write(fo, '(a)') '# Col 2: Matter transfer function' 
        write(fo, '(a)') '# Col 3: Matter power spectrum in Mpc^-3' 
        write(fo, '(a)') '# Col 4: Effective index (1-st log-derivative of power spectrum w.r.to k)' 

        !! generate data
        lnk = opts%k_range(1)
        do while ( lnk < opts%k_range(2) )
            k       = 10.**lnk
            data(1) = k

            !! calculate power
            call get_matter_powerspec(k, z, data(3), tk = data(2), dlnpk = data(4), stat = ierror)
            if ( ierror .ne. 0 ) then
                write (stderr, '(a,e0.6,a,f0.3)') 'error: while calculating power spectrum at k = ', k,', z = ', z
                return 
            end if

            !! write to file
            write(fo, '(4(e12.6,a))') ( data(i), ', ', i = 1, 3 ), data(4)
            
            lnk = lnk + opts%k_range(3)
        end do
        close(fo)

        write (stderr, '(a,f0.3)') 'info: successfully generated power spectrum data at z = ', z
        
    end subroutine generate_power_data

    !>
    !! Calculate and save massfunction data
    !!
    subroutine generate_mffbf_data(z)
        real(dp), intent(in) :: z
        integer, parameter  :: fo = 11
        character(len=32)   :: zstring
        character(len=128)  :: file
        integer  :: i, pos1, pos2, pos3
        real(dp) :: data(7), m, lnm

        file = opts%output_file_halos
        if ( ierror .ne. 0 ) return !! not initialised 
        if ( len( trim(file) ) == 0 ) return

        pos1 = scan(file, '{')
        pos2 = scan(file, '}')
        if ( ( ( pos1 == 0 ) .and. ( pos2 .ne. 0 ) ) .or. ( ( pos1 .ne. 0 ) .and. ( pos2 == 0 ) ) ) then
            ierror = 1
            write (stderr, '(a)') 'error: incorrect file name - '//file
            return
        else if ( ( pos1 .ne. 0 ) .or. ( pos2 .ne. 0 ) ) then 
            write(zstring, '('//file(pos1+1:pos2-1)//')') z
            pos3 = scan(zstring, '0123456789')
            file = file(1:pos1-1)//trim(zstring(pos3:))//file(pos2+1:)
        end if

        if ( ierror .ne. 0 ) return !! not initialised 

        !! open file
        open(fo, file = file)
        write(fo, '(a)') '# Col 1: Mass in Msun' 
        write(fo, '(a)') '# Col 2: Radius in Mpc' 
        write(fo, '(a)') '# Col 3: Variance, sigma' 
        write(fo, '(a)') '# Col 4: 1-st log-derivative of variance w.r.to mass' 
        write(fo, '(a)') '# Col 5: Mass function, f(sigma)' 
        write(fo, '(a)') '# Col 6: Mass function, dn/dlnm in Msun/Mpc^3' 
        write(fo, '(a)') '# Col 7: Halo bias' 

        !! generate data
        lnm = opts%m_range(1)
        do while ( lnm < opts%m_range(2) )
            m       = 10.**lnm
            data(1) = m

            !! calculate power
            call get_halo_mfbf(m, z, data(6), data(7), r = data(2), fs = data(5), s = data(3), &
                        &      dlns = data(4), stat = ierror)
            if ( ierror .ne. 0 ) then
                write (stderr, '(a,e0.6,a,f0.3)') 'error: while calculating mass-function at m = ', m,', z = ', z
                return 
            end if

            !! write to file
            write(fo, '(7(e12.6,a))') ( data(i), ', ', i = 1, 6 ), data(7)
            
            lnm = lnm + opts%m_range(3)
        end do
        close(fo)

        write (stderr, '(a,f0.3)') 'info: successfully generated halo data at z = ', z
        
    end subroutine generate_mffbf_data

end program main