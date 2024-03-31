module growth_calculator
    use constants, only: dp, PI, SPEED_OF_LIGHT_KMPS, EPS
    use numerical, only: quadrule
    use objects, only: cosmology_model
    implicit none

contains

    !>
    !! Evaluate the integral expression related to linear growth_calculator calculation.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!  dplus: real            - Calculated value of growth factor.
    !!  fplus: real            - Calculated value of growth rate (optional).
    !!
    subroutine calculate_linear_growth(z, cm, qrule, dplus, fplus)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule

        real(dp), intent(out) :: dplus           !! growth factor
        real(dp), intent(out), optional :: fplus !! growth rate

        integer  :: i
        real(dp) :: integral !! value of the integral
        real(dp) :: a, fa, dlnfa, x_scale, zp1
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k 
        
        integral = 0.0_dp

        !! ditance by evaluating the integral of [ a * E(a) ]^-3 from 0 to a...
        x_scale = 0.5_dp / ( z + 1._dp ) 
        do i = 1, qrule%n
            !! scale factor
            a = x_scale * (qrule%x(i) + 1.) 

            !! calculating integrand
            call cm%calculate_hubble_func(1/a-1., fa) !! E**2(z)
            fa = 1./ ( a * sqrt( fa ) )**3
            
            integral = integral + qrule%w(i) * fa
        end do
        integral = integral * x_scale

        !! calculation of hubble parameter function E(z)
        call cm%calculate_hubble_func(z, fa, dlnfa)
        zp1 = z + 1

        !! calculating linear growth factor
        dplus = sqrt(fa) * integral

        if ( present(fplus) ) then !! calculating linear growth rate
            fplus = zp1**2 / fa / dplus - 0.5*dlnfa
        end if

    end subroutine calculate_linear_growth
    
end module growth_calculator