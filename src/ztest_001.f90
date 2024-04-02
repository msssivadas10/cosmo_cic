program main
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmo_t
    use dist_time_calculator, only: setup_distance_calculator, calculate_comoving_distance
    use growth_calculator, only: setup_growth_calculator, calculate_linear_growth
    use transfer_eh, only: tf_eisenstein98_calculate_params, tf_eisenstein98
    use massfunc_models, only: mf_tinker08
    use linbias_models , only: bf_tinker10
    use matter_power_calculator, only: get_power_spectrum,                  &
                                    &  get_variance,                        &
                                    &  set_power_model,                     &
                                    &  set_normalization
    use variance_calculator, only: setup_variance_calculator 
    use massfunc_bias_calculator, only: setup_massfunc_bias_calculator,     &
                                    &   set_bias_model,                     &
                                    &   set_massfunc_model,                 &
                                    &   calculate_massfunc_bias

    implicit none

    type(cosmo_t) :: cm
    integer :: stat

    call initialize()
    call generate_redshift_data()
    call generate_power_data()
    call generate_massfunc_data()

    contains

        subroutine initialize()

            !! create cosmology model
            cm = cosmo_t(H0=70.0_dp, Omega_m=0.3_dp, Omega_b=0.05_dp, ns=1.0_dp, sigma8=0.8_dp)
            call cm%initialize_cosmology(stat) !! initialising the module
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup cosmology'
                return
            end if
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

            !! set-up power
            call tf_eisenstein98_calculate_params(cm, stat = stat)
            call set_power_model(tf_eisenstein98, stat = stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup power spectrum'
                return
            end if

            !! initialize variance calculator
            call setup_variance_calculator(128, stat, filt = 'tophat')
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup variance'
                return
            end if

            !! normalization
            call set_normalization(cm, stat = stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: normalizing power spectrum'
                return
            end if

            !! setup mass function calculator
            call set_massfunc_model(mf_tinker08, stat = stat)
            call set_bias_model(bf_tinker10, stat = stat)
            call setup_massfunc_bias_calculator(get_variance, stat = stat)
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup mass function'
                return
            end if

            write (*,*) 'info: successfull initialization'
            
        end subroutine initialize

        subroutine generate_redshift_data()
            integer, parameter :: fo = 15
            real(dp) :: z, r, dvdz, dplus, fplus

            open(fo, file = 'z_out.csv')

            !! header
            write (fo,'(5(a16,a))') 'z',', ','r',', ','dvdz',', ','dplus',', ','fplus'
            write (* ,'(5(a16,a))') 'z',', ','r',', ','dvdz',', ','dplus',', ','fplus'

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

            write (*,*) 'info: successfull power data generation'

        end subroutine generate_power_data
                        
        subroutine generate_massfunc_data()
            integer, parameter :: fo = 15
            real(dp) :: m, s, dlns, dndlnm, bm, fs
            real(dp) :: z
            character(len=16) :: file

            z = 0.0_dp
            do while ( z < 3. )

                write(file,'(a,f3.1,a)') 'mfbf_z', z, '.csv'
                open(fo, file = file)

                !! header
                write (fo,'(6(a16,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'
                write (* ,'(6(a16,a))') 'm',', ','s',', ','dln',', ','dndlnm',', ','fs',', ','bm'

                m = 1.0e+06_dp
                do while ( m <= 1.0e+15_dp )
                    
                    !! mass function 
                    call calculate_massfunc_bias(m, z, 200._dp, cm, dndlnm, bm, &
                                            &    fs = fs, s = s, dlns = dlns, stat = stat)
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
                
                write (*,*) 'info: successfull mass function data generation at z =', z
                z = z + 2.0_dp
            end do

        end subroutine generate_massfunc_data
        
end program main
