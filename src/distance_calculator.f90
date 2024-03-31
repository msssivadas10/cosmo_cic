module distance_calculator
    use constants, only: dp, PI, SPEED_OF_LIGHT_KMPS, EPS
    use numerical, only: quadrule
    use objects, only: cosmology_model
    implicit none

contains

    !>
    !! Calculate comoving distance at redshift z.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!  r    : real            - Comoving distance in Mpc.
    !!  dvdz : real            - Comoving volume element in Mpc^3.
    !!
    subroutine calculate_comoving_distance(z, cm, qrule, r, dvdz) 
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp), intent(out) :: r !! distance in Mpc
        real(dp), intent(out), optional :: dvdz !! volume element in Mpc^3

        real(dp) :: retval, drdz
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        real(dp) :: a, fa, x_scale
        integer  :: i

        r = 0.0_dp; dvdz = 0.0_dp
        if ( abs( z ) < EPS ) return

        H0   = cm%H0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k 

        !! ditance by evaluating the integral of [ a^2 * E(a) ]^-1 from a to 1...
        x_scale = 0.5_dp * z / ( z + 1._dp ) 
        do i = 1, qrule%n
            !! scale factor
            a = x_scale * qrule%x(i) + (1. - x_scale) 
            
            !! calculating integrand
            call cm%calculate_hubble_func(1/a-1., fa) !! E**2(z)
            fa = 1./ a**2 / sqrt( fa )
            
            retval = retval + qrule%w(i) * fa
        end do
        retval = retval * x_scale
        
        !! comoving distance in Mpc
        r = retval * SPEED_OF_LIGHT_KMPS / H0

        if ( present(dvdz) ) then !! calculate comoving volume element, dvdz
            
            !! distance derivative w.r.to z: 1/E(a)
            call cm%calculate_hubble_func(z, drdz)
            drdz = 1.0 / sqrt( drdz ) * SPEED_OF_LIGHT_KMPS / H0

            !! volume element
            dvdz = 4*PI * r**2 * drdz

        end if
        
    end subroutine calculate_comoving_distance

    !>
    !! Calculate comoving coordinate at redshift z.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!
    !! Returns:
    !!  retval: real - Comoving coordinate in Mpc.
    !!
    function comoving_coordinate(z, cm, qrule) result(retval)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp) :: retval !! value of coordinate
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        real(dp) :: k !! curvature
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k
        H0   = cm%H0

        call calculate_comoving_distance(z, cm, qrule, retval) !! comoving distance

        if ( abs(Ok0) < EPS ) return !! for flat cosmology, comoving distance = comoving coordinate

        if ( Ok0 < 0. ) then !! spherical or closed geometry
            k = sqrt( -Ok0 ) / SPEED_OF_LIGHT_KMPS * H0
            retval = sin( k*retval ) / k
        else !! hyperbolic or open geometry
            k = sqrt( Ok0 ) / SPEED_OF_LIGHT_KMPS * H0
            retval = sinh( k*retval ) / k
        end if
        
    end function comoving_coordinate

    !>
    !! Calculate luminocity distance at redshift z.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!
    !! Returns:
    !!  retval: real - Luminocity distance in Mpc.
    !!
    function luminocity_distance(z, cm, qrule) result(retval)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp) :: retval !! distance in Mpc
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k
        H0   = cm%H0

        retval =  comoving_coordinate(z, cm, qrule) * (z + 1.)
        
    end function luminocity_distance

    !>
    !! Calculate angular diameter distance at redshift z.
    !!
    !! Parameters:
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!
    !! Returns:
    !!  retval: real - Angular diameter distance in Mpc.
    !!
    function angular_diameter_distance(z, cm, qrule) result(retval)
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp) :: retval !! distance in Mpc
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k
        H0   = cm%H0

        retval =  comoving_coordinate(z, cm, qrule) / (z + 1.)
        
    end function angular_diameter_distance

    !>
    !! Calculate angular size corresponding to a physical size x, redshift z.
    !!
    !! Parameters:
    !!  x    : real            - Physical size in Mpc.
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!
    !! Returns:
    !!  retval: real - Angular size in arcsec.
    !!
    function angular_size(x, z, cm, qrule) result(retval)
        real(dp), intent(in)  :: x !! physical size in Mpc
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp) :: retval !! distance in Mpc
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k
        H0   = cm%H0

        retval = angular_diameter_distance(z, cm, qrule) !! angular diameter distance in Mpc
        retval = x / retval * ( 3600.0*180.0 / PI ) !! angular size in arcsec

    end function angular_size

    !>
    !! Calculate physical size corresponding to a angular size x, redshift z.
    !!
    !! Parameters:
    !!  x    : real            - Angular size in arcsec.
    !!  z    : real            - Redshift (must be greater than -1).
    !!  cm   : cosmology_model - Cosmology parameters
    !!  qrule: quadrule        - Integration rule 
    !!
    !! Returns:
    !!  retval: real - Physical size in Mpc.
    !!
    function physical_size(x, z, cm, qrule) result(retval)
        real(dp), intent(in)  :: x !! angular size in arcsec 
        real(dp), intent(in)  :: z !! redshift 
        type(cosmology_model), intent(in) :: cm !! cosmology parameters
        type(quadrule), intent(in) :: qrule     !! integration rule
        
        real(dp) :: retval !! size in arcsec
        real(dp) :: H0 !! hubble parameter value at z=0
        real(dp) :: Om0, Ode0, Ok0 !! density parameters at z=0
        Om0  = cm%Omega_m
        Ode0 = cm%Omega_de
        Ok0  = cm%Omega_k
        H0   = cm%H0
    
        retval = angular_diameter_distance(z, cm, qrule) !! angular diameter distance in Mpc
        retval = x * retval * ( PI / 3600.0 / 180.0 ) !! angular size

    end function physical_size
    
end module distance_calculator
