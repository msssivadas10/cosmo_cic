program main
    use constants, only: dp
    use objects, only: cosmology_model
    implicit none

    type(cosmology_model) :: cm
    integer :: stat
    ! real(dp) :: z, y1, y2, y3, y4

    !! cosmology model
    cm = cosmology_model(H0=70.0_dp, Omega_m=0.3_dp, Omega_b=0.05_dp, ns=1.0_dp, sigma8=0.8_dp)
    call cm%initialize_cosmology(stat) !! initialising the module
    ! write (*,'(a,f8.4)') 'H0  : ', cm%H0
    ! write (*,'(a,f8.4)') 'Om0 : ', cm%Omega_m
    ! write (*,'(a,f8.4)') 'Ob0 : ', cm%Omega_b
    ! write (*,'(a,f8.4)') 'Ode0: ', cm%Omega_de

    call generate_redshift_data()
    call generate_power_data()

    contains

        subroutine generate_redshift_data()
            use dist_time_calculator, only: calculate_comoving_distance, &
                        &                   setup_distance_calculator
            use growth_calculator, only: calculate_linear_growth, &
                        &                setup_growth_calculator

            integer, parameter :: fo = 15
            real(dp) :: z, r, dvdz, dplus, fplus

            !! initialize distance calculator
            call setup_distance_calculator(128, stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup distance'
                return
            end if

            !! initialize growth calculator
            call setup_growth_calculator(128, stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup growth'
                return
            end if

            open(fo, file = 'test/z_out.csv')

            !! header
            write (fo,'(5(a,a))') 'redshift',', ','dist_Mpc',', ','vol_Mpc^3',', ','D_+',', ','f_+'
            write (* ,'(5(a,a))') 'redshift',', ','dist_Mpc',', ','vol_Mpc^3',', ','D_+',', ','f_+'

            z = 0.0_dp
            do while ( z <= 10. )

                !! comoving distance 
                call calculate_comoving_distance(z, cm, r, dvdz = dvdz, stat = stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating distance at z = ', z
                    return
                end if

                !! linear growth 
                call calculate_linear_growth(z, cm, dplus, fplus = fplus, stat = stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating growth at z = ', z
                    return
                end if

                write (fo,'(5(e16.5,a))') z,', ',r,', ',dvdz,', ',dplus,', ',fplus
                write (* ,'(5(e16.5,a))') z,', ',r,', ',dvdz,', ',dplus,', ',fplus
                z = z + 1.0_dp
            end do
            close(fo)
            
        end subroutine generate_redshift_data

        subroutine generate_power_data()
            use power_eh, only: get_power_spectrum,     &
                        &       get_variance,           &
                        &       set_normalization,      &
                        &       tf_eisenstein98_calculate_params
            use variance_calculator, only: setup_variance_calculator, &
                        &                  vc => calculate_variance
            use growth_calculator, only: setup_growth_calculator

            integer, parameter :: fo = 15
            real(dp) :: x, pk, tk, dlnpk, s, dlns, d2lns
            real(dp) :: z = 0.0_dp

            !! initialize growth calculator
            call setup_growth_calculator(128, stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup growth'
                return
            end if

            !! initialize variance calculator
            call setup_variance_calculator(128, stat, filt = 'tophat')
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup variance'
                return
            end if

            !! set-up power
            call tf_eisenstein98_calculate_params(cm, stat = stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup power spectrum'
                return
            end if

            !! normalization
            call set_normalization(vc, cm, stat = stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: normalizing power spectrum'
                return
            end if
            ! write (*,*) cm%ps_norm

            open(fo, file = 'test/power.csv')

            !! header
            write (fo,'(7(a,a))') 'x',', ','tk',', ','pk',', ','dlnpk',', ',', ','s',', ','dlns',', ','d2lns'
            write (* ,'(7(a,a))') 'x',', ','tk',', ','pk',', ','dlnpk',', ',', ','s',', ','dlns',', ','d2lns'

            x = 1.0e-04_dp
            do while ( x <= 1.0e+04_dp )
                
                !! power
                call get_power_spectrum(x, z, cm, pk, tk = tk, dlnpk = dlnpk)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating power spectrum'
                    return
                end if
                pk = pk * cm%sigma8**2
                
                !! variance
                call get_variance(vc, x, z, cm, s, dlns = dlns, d2lns = d2lns, stat = stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating variance'
                    return
                end if
                s = s * cm%sigma8**2

                write (fo,'(7(e16.5,a))') x,', ',tk,', ',pk,', ',dlnpk,', ',s,', ',dlns,', ',d2lns
                write (* ,'(7(e16.5,a))') x,', ',tk,', ',pk,', ',dlnpk,', ',s,', ',dlns,', ',d2lns
                x = x * 10.0_dp
            end do
            close(fo)

            ! x = 1e+03_dp
            ! call get_variance(x, z, cm, qrk, qrz, s, dlns = dlns, d2lns = d2lns)
            ! write (*,*) cm%ps_norm

            ! x = 1e+4_dp
            ! write (*,*) ( 3*( sin(x) - x*cos(x) ) / x**3 )

        end subroutine generate_power_data

end program main