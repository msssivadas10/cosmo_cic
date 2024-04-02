program main
    use iso_fortran_env, only: dp => real64
    use objects, only: cosmo_t
    use halo_model, only: halomodel_t
    use transfer_eh, only: tf_eisenstein98_calculate_params, tf_eisenstein98
    use massfunc_models, only: mf_tinker08
    use linbias_models , only: bf_tinker10
    use halomodel_calculator, only: setup_halomodel_calculator
    implicit none

    type(halomodel_t) :: hm
    type(cosmo_t) :: cm
    integer :: stat
    
    contains

        subroutine initialize()

            !! create cosmology model
            cm = cosmo_t(H0=70.0_dp, Omega_m=0.3_dp, Omega_b=0.05_dp, ns=1.0_dp, sigma8=0.8_dp)
            call cm%initialize_cosmology(stat) !! initialising the module
            if ( stat .ne. 0 ) then
                write (*,*) 'error: setup cosmology'
                return
            end if

            !! create halo model
            hm = halomodel_t(log_Mmin=11.68_dp, sigma=0.15_dp, log_M0=11.86_dp, log_M1=13.0_dp, a=1.02_dp)

            !! initialize halo calculator
            call setup_halomodel_calculator(128, 128, 128, mf_tinker08, bf_tinker10, tf_eisenstein98, cm, stat)

            write (*,*) 'info: successfull initialization'
            
        end subroutine initialize

end program main