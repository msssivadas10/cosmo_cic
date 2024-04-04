program main
    use iso_fortran_env, only: dp => real64
    use cosmology, only: cosmo_t
    use calculate_dist, only: DC_init, DC_get_distance
    use calculate_growth, only: GC_init, GC_get_growth
    use calculate_specmom, only: SMC_init, SMC_get_specmom
    use transfer_eh, only: tf_eisenstein98_init, tf_eisenstein98
    implicit none

    type(cosmo_t) :: cm
    integer  :: stat
    real(dp) :: norm

    call initialize()
    ! call generate_redshift_data()
    call generate_power_data()

    contains

    subroutine ps(cm1, k, pk, args, stat1)
        !! inputs:
        class(cosmo_t), intent(in) :: cm1
        real(dp), intent(in)       :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in), optional :: args(:) !! additional arguments
        !! outputs:
        real(dp), intent(out) :: pk
        integer , intent(out), optional :: stat1
    
        real(dp) :: z = 0._dp 
        
        if ( present(args) ) z = args(1)
        call tf_eisenstein98(cm1, k, z, pk, stat=stat1)
        pk = k**cm%ns * pk**2
        
    end subroutine ps

        subroutine initialize()
            real(dp) :: r, args(1)
            !! create cosmology model
            cm   = cosmo_t(H0=70.0_dp, Om0=0.3_dp, Ob0=0.05_dp, ns=1.0_dp, sigma8=0.8_dp)
            stat = cm%init() !! initialising the module
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup cosmology'
                return
            end if
            write (*,'(a,f8.3,a)') '- Hubble parameter     : ', cm%H0    , ' km/sec/Mpc'
            write (*,'(a,f8.3,a)') '- Total matter density : ', cm%Om0   , ' '
            write (*,'(a,f8.3,a)') '- Baryon density       : ', cm%Ob0   , ' '
            write (*,'(a,f8.3,a)') '- Neutrino density     : ', cm%Onu0  , ' '
            write (*,'(a,f8.3,a)') '- Curvature density    : ', cm%Ok0   , ' '
            write (*,'(a,f8.3,a)') '- Dark-energy density  : ', cm%Ode0  , ' '
            write (*,'(a,f8.3,a)') '- Neutrino number      : ', cm%Nnu   , ' '
            write (*,'(a,f8.3,a)') '- Power spectrum index : ', cm%ns    , ' '
            write (*,'(a,f8.3,a)') '- sigma-8              : ', cm%sigma8, ' '
            write (*,'(a,f8.3,a)') '- CMB temperature      : ', cm%Tcmb0 , ' K'
            write (*,'(a,f8.3,a)') '- Dark-energy w0       : ', cm%w0    , ' '
            write (*,'(a,f8.3,a)') '- Dark-energy wa       : ', cm%wa    , ' '

            !! initialize distance calculator
            write (*,*) 'info: setup distance'
            call DC_init(size=128, stat=stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup distance'
                return
            end if

            !! initialize growth calculator
            write (*,*) 'info: setup growth'
            call GC_init(size=128, stat=stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup growth'
                return
            end if

            !! set-up power
            write (*,*) 'info: setup power spectrum'
            call tf_eisenstein98_init(cm, stat=stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup power spectrum'
                return
            end if

            !! initialize variance calculator
            write (*,*) 'info: setup variance'
            call SMC_init(size=512, stat=stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup variance'
                return
            end if

            !! normalising power spectrum
            write (*,*) 'info: normalize power spectrum'
            r = 800.0 / cm%H0
            args(1) = 0._dp
            call SMC_get_specmom(cm, ps, r, 0, norm, args=args, stat=stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: normalize power spectrum'
                return
            end if
            norm = 1./norm
            write (*,'(a,e12.3)') 'Power normalization factor: ', norm

            write (*,*) 'info: successfull initialization'
            
        end subroutine initialize

        subroutine generate_redshift_data()
            integer, parameter :: fo = 15
            real(dp) :: z, r, dvdz, dplus, fplus
            open(fo, file = 'z_out.csv')
            !! header
            write (fo,'(5(a8,a))') 'z',', ','r',', ','dvdz',', ','dplus',', ','fplus'
            write (* ,'(5(a8,a))') 'z',', ','r',', ','dvdz',', ','dplus',', ','fplus'
            z = 0.0_dp
            do while ( z <= 10. )
                !! comoving distance 
                call DC_get_distance(cm, z, r, dvdz=dvdz, stat=stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating distance at z = ', z
                    return
                end if
                r    = r * 1E-03 ! Gpc
                dvdz = dvdz * 1E-09 ! Gpc^3
                !! linear growth 
                call GC_get_growth(cm, z, dplus, fplus=fplus, stat=stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating growth at z = ', z
                    return
                end if
                write (fo,'(5(f8.3,a))') z,', ',r,', ',dvdz,', ',dplus,', ',fplus
                write (* ,'(5(f8.3,a))') z,', ',r,', ',dvdz,', ',dplus,', ',fplus
                z = z + 1.0_dp
            end do
            close(fo)
            write (*,*) 'info: successfull redshift data generation'
        end subroutine generate_redshift_data
            
        subroutine generate_power_data()
            integer, parameter :: fo = 15
            real(dp) :: k, pk, tk, dlnpk
            real(dp) :: z = 0.0_dp
            open(fo, file = 'power.csv')
            !! header
            write (fo,'(4(a16,a))') 'k',', ','tk',', ','pk',', ','dlnpk'
            write (* ,'(4(a16,a))') 'k',', ','tk',', ','pk',', ','dlnpk'
            k = 1.0e-04_dp
            do while ( k <= 1.0e+04_dp )
                !! transfer
                call tf_eisenstein98(cm, k, z, tk, dlntk=dlnpk, stat=stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating transfer function'
                    return
                end if
                !! power
                call ps(cm, k, pk, stat1=stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating power spectrum'
                    return
                end if
                pk    = pk * cm%sigma8**2 * norm
                dlnpk = cm%ns + 2*dlnpk
                write (fo,'(4(e16.5,a))') k,', ',tk,', ',pk,', ',dlnpk
                write (* ,'(4(e16.5,a))') k,', ',tk,', ',pk,', ',dlnpk
                k = k * 10.0_dp
            end do
            close(fo)
            write (*,*) 'info: successfull power data generation'
        end subroutine generate_power_data
        
end program main
