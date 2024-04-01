program main
    use constants, only: dp
    use objects, only: cosmology_model
    use power_eh, only: get_power_spectrum,     &
                    &   ps => get_power_unnorm, &
                    &   get_variance,           & 
                    &   set_normalization,      &
                    &   get_normalization,      &
                    &   tf_eisenstein98_calculate_params
    use variance_calculator, only: setup_variance_calculator, vc => calculate_variance
    use dist_time_calculator, only: calculate_comoving_distance, setup_distance_calculator
    use growth_calculator, only: calculate_linear_growth, setup_growth_calculator
    use massfunc_bias_calculator, only: calculate_massfunc_bias
    use massfunc_models, only: mf => mf_tinker08
    use linbias_models, only: bf => bf_tinker10

    implicit none

    type(cosmology_model) :: cm
    integer :: stat
    real(dp) :: ps_norm
    ! real(dp) :: z, y1, y2, y3, y4

    !! cosmology model
    cm = cosmology_model(H0=70.0_dp, Omega_m=0.3_dp, Omega_b=0.05_dp, ns=1.0_dp, sigma8=0.8_dp)
    call cm%initialize_cosmology(stat) !! initialising the module
    ! write (*,'(a,f8.4)') 'H0  : ', cm%H0
    ! write (*,'(a,f8.4)') 'Om0 : ', cm%Omega_m
    ! write (*,'(a,f8.4)') 'Ob0 : ', cm%Omega_b
    ! write (*,'(a,f8.4)') 'Ode0: ', cm%Omega_de

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
    ps_norm = get_normalization()

    call generate_redshift_data()
    call generate_power_data()
    call generate_massfunc_data()

    contains

        subroutine generate_redshift_data()
            integer, parameter :: fo = 15
            real(dp) :: z, r, dvdz, dplus, fplus

            open(fo, file = 'z_out.csv')

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
            integer, parameter :: fo = 15
            real(dp) :: k, pk, tk, dlnpk
            real(dp) :: z = 0.0_dp

            open(fo, file = 'power.csv')

            !! header
            write (fo,'(4(a,a))') 'k',', ','tk',', ','pk',', ','dlnpk'
            write (* ,'(4(a,a))') 'k',', ','tk',', ','pk',', ','dlnpk'

            k = 1.0e-04_dp
            do while ( k <= 1.0e+04_dp )
                
                !! power
                call get_power_spectrum(k, z, cm, pk, tk = tk, dlnpk = dlnpk)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating power spectrum'
                    return
                end if
                pk = pk * cm%sigma8**2

                write (fo,'(4(e16.5,a))') k,', ',tk,', ',pk,', ',dlnpk
                write (* ,'(4(e16.5,a))') k,', ',tk,', ',pk,', ',dlnpk
                k = k * 10.0_dp
            end do
            close(fo)

        end subroutine generate_power_data
            
        subroutine generate_massfunc_data()
            integer, parameter :: fo = 15
            real(dp) :: m, s, dlns, dndlnm, bm, fs
            real(dp) :: z

            open(fo, file = 'mfbf_z0.csv')

            !! header
            write (fo,'(6(a,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'
            write (* ,'(6(a,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'

            z = 0._dp
            m = 1.0e+06_dp
            do while ( m <= 1.0e+15_dp )
                
                !! mass function 
                call calculate_massfunc_bias(mf, bf, m, z, 200._dp, cm, ps_norm, ps,      &
                                &            vc, dndlnm, bm, fs = fs, s = s, dlns = dlns, &
                                &            stat = stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating mass-function'
                    return
                end if
                ! s = s * cm%sigma8**2

                write (fo,'(6(e16.5,a))') m,', ',s,', ',dlns,', ',dndlnm,', ',fs,', ',bm
                write (* ,'(6(e16.5,a))') m,', ',s,', ',dlns,', ',dndlnm,', ',fs,', ',bm
                m = m * 10.0_dp
            end do
            close(fo)

            !!

            open(fo, file = 'mfbf_z2.csv')

            !! header
            write (fo,'(6(a,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'
            write (* ,'(6(a,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'

            z = 2._dp
            m = 1.0e+06_dp
            do while ( m <= 1.0e+15_dp )
                
                !! mass function 
                call calculate_massfunc_bias(mf, bf, m, z, 200._dp, cm, ps_norm, ps,      &
                                &            vc, dndlnm, bm, fs = fs, s = s, dlns = dlns, &
                                &            stat = stat)
                if ( stat .ne. 0 ) then
                    write (*,*) 'error: calculating mass-function'
                    return
                end if
                ! s = s * cm%sigma8**2

                write (fo,'(6(e16.5,a))') m,', ',s,', ',dlns,', ',dndlnm,', ',fs,', ',bm
                write (* ,'(6(e16.5,a))') m,', ',s,', ',dlns,', ',dndlnm,', ',fs,', ',bm
                m = m * 10.0_dp
            end do
            close(fo)

        end subroutine generate_massfunc_data

end program main