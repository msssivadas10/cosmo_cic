!!
!! Halo model type
!!
module halo_model
    use iso_fortran_env, only: dp => real64
    implicit none

    private
    
    !>
    !! Halo model parameters
    !!
    type, public :: halomodel_t

        !! Parameters for central galaxy count
        real(dp) :: log_Mmin !! Minimum mass
        real(dp) :: sigma    !! Spread of the central galaxy distribution??

        !! Parameters for the satellite galaxy count
        real(dp) :: log_M0 !! Mass scale of the drop
        real(dp) :: log_M1 !! Amplitude
        real(dp) :: a      !! Asymptotic slope

        !! Normalization for total galaxy count
        real(dp) :: dc = 1.0_dp

        contains
            procedure :: get_galaxy_count
    end type
    
contains

    subroutine get_galaxy_count(self, m, ntot, ncen, nsat)
        class(halomodel_t), intent(in) :: self
        real(dp), intent(in)  :: m !! halo mass in Msun
        real(dp), intent(out) :: ntot !! total galaxy count
        real(dp), intent(out), optional :: ncen !! central galaxy count
        real(dp), intent(out), optional :: nsat !! satellite galaxy count
        real(dp) :: M0, M1, ncen1, nsat1
        M0 = 10.0**(self%log_M0)
        M1 = 10.0**(self%log_M1)

        !! central galaxy count
        ncen1 = 0.5*( 1. + erf( log10(m) - self%log_Mmin ) / self%sigma )
        if ( present(ncen) ) ncen = ncen1

        !! satellite galaxy count
        nsat1 = 0.0_dp
        if ( m > M0 ) then
            nsat1 = ncen1 * ( (m - M0) / M1 )**(self%a)
        end if
        if ( present(nsat) ) nsat = nsat1

        !! total count
        ntot = ( ncen1 + nsat1 )*(self%dc)
        
    end subroutine get_galaxy_count
    
end module halo_model
