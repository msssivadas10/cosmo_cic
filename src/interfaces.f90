module interfaces
    implicit none

    interface
        !! Interface to power spectrum calculator
        subroutine ps_calculate(k, z, cm, pk, stat)
            use constants, only: dp
            use objects, only: cosmology_model
            real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
            real(dp), intent(in) :: z !! redshift
            type(cosmology_model), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: pk
            integer , intent(out), optional :: stat
        end subroutine ps_calculate
    end interface

    interface
        !! Interface to variance calculator
        subroutine var_calculate(ps, r, z, cm, sigma, dlns, d2lns, stat)
            use constants, only: dp
            use objects, only: cosmology_model
            procedure(ps_calculate) :: ps !! transfer function 
            real(dp), intent(in) :: r !! scale in Mpc
            real(dp), intent(in) :: z !! redshift
            type(cosmology_model), intent(in) :: cm !! cosmology parameters

            real(dp), intent(out) :: sigma !! variance 
            real(dp), intent(out), optional :: dlns, d2lns 
            integer , intent(out), optional :: stat
        end subroutine var_calculate
    end interface

    interface
        !! Interface to mass function or bias calculator
        subroutine fs_calculate(s, z, Delta, cm, retval, stat)
            use constants, only: dp
            use objects, only: cosmology_model
            real(dp), intent(in) :: s 
            real(dp), intent(in) :: z !! redshift
            real(dp), intent(in) :: Delta !! overdensity w.r.to mean
            type(cosmology_model), intent(in) :: cm !! cosmology parameters
            real(dp), intent(out) :: retval
            integer , intent(out), optional :: stat
        end subroutine fs_calculate
    end interface
    
contains
    
end module interfaces