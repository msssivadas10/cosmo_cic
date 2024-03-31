program main
    use constants, only: dp
    use numerical, only: generate_gaussleg, quadrule
    use objects, only: cosmology_model
    implicit none

    integer, parameter :: n  = 128
    integer, parameter :: n2 = 64
    type(quadrule) :: qrz, qrk
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

    !! integration rule
    call generate_gaussleg(n, qrz) !! for z
    call generate_gaussleg(n, qrk) !! for z
    qrk%a = -12.0_dp
    qrk%b =  12.0_dp

    ! call generate_redshift_data()
    ! call generate_power_data()

    contains

        subroutine generate_redshift_data()
            use distance_calculator, only: calculate_comoving_distance
            use growth_calculator, only: calculate_linear_growth
            integer, parameter :: fo = 15
            real(dp) :: z, r, dvdz, dplus, fplus

            open(fo, file = 'test/z_out.csv')

            !! header
            write (fo,'(5(a,a))') 'redshift',', ','dist_Mpc',', ','vol_Mpc^3',', ','D_+',', ','f_+'

            z = 0.0_dp
            do while ( z <= 10. )

                !! comoving distance 
                call calculate_comoving_distance(z, cm, qrz, r, dvdz)

                !! linear growth 
                call calculate_linear_growth(z, cm, qrz, dplus, fplus)

                write (fo,'(5(e12.6,a))') z,', ',r,', ',dvdz,', ',dplus,', ',fplus
                z = z + 1.0_dp
            end do
            close(fo)
            
        end subroutine generate_redshift_data

        subroutine generate_power_data()
            use power_eh, only: get_power_spectrum, get_variance, set_normalization, tf_eisenstein98_calculate_params
            integer, parameter :: fo = 15
            real(dp) :: x, pk, tk, dlnpk, s, dlns, d2lns
            real(dp) :: z = 0.0_dp

            !! set-up
            call tf_eisenstein98_calculate_params(cm, qrz)

            ! x = 8.0_dp / (0.01*cm%H0)
            ! call get_variance(x, z, cm, qrk, qrz, s)
            ! write (*,*) s, cm%ps_norm
            ! return

            !! normalization
            call set_normalization(cm, qrk, qrz)
            ! write (*,*) cm%ps_norm

            open(fo, file = 'test/power.csv')

            !! header
            write (fo,'(7(a,a))') 'x',', ','tk',', ','pk',', ','dlnpk',', ',', ','s',', ','dlns',', ','d2lns'

            x = 1.0e-04_dp
            do while ( x <= 1.0e+04_dp )
                
                !! power
                call get_power_spectrum(x, z, cm, qrz, pk, tk = tk, dlnpk = dlnpk)
                ! pk = pk * cm%sigma8**2
                
                !! variance
                call get_variance(x, z, cm, qrk, qrz, s, dlns = dlns, d2lns = d2lns)
                ! s = s * cm%sigma8**2

                write (fo,'(7(e12.3,a))') x,', ',tk,', ',pk,', ',dlnpk,', ',s,', ',dlns,', ',d2lns
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