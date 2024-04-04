module transfer_eh
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    use calculate_growth, only: GC_get_growth
    use cosmology, only: cosmo_t
    implicit none

    private

    real(dp), parameter :: e = 2.718281828459045_dp

    !! Power spectrum model ids
    integer, parameter :: PS_EH98_ZB  = 1 !! Eisenstein & Hu without BAO
    integer, parameter :: PS_EH98_NU  = 2 !! Eisenstein & Hu including neutrino
    integer, parameter :: PS_EH98_BAO = 3 !! Eisenstein & Hu with BAO

    real(dp) :: h      !! Hubble parameter in unit of 100 km/sec/Mpc
    real(dp) :: Omh2   !! Total matter density parameter
    real(dp) :: Obh2   !! Baryon matter density parameter
    real(dp) :: theta  !! CMB temperaure in 2.7K unit
    real(dp) :: Nnu    !! Number of massive species
    real(dp) :: dplus0 !! Growth factor scaling
    real(dp) :: fc, fb, fnu, fcb,fnb, pc, pcb 
    real(dp) :: alpha_c, alpha_b, alpha_nu, alpha_g
    real(dp) :: beta_c, beta_b, beta_node
    real(dp) :: k_silk !! Silk damping scale
    real(dp) :: s      !! Sound horizon in Mpc
    real(dp) :: z_eq   !! Redshift at matter-radiation equality epoch
    real(dp) :: z_d    !! Redshift at drag epoch
    real(dp) :: yfs_over_q2   !! yfs / q^2 
    integer  :: ps_model = PS_EH98_ZB !! Power spectrum model to use

    !! Error flags
    integer, parameter :: ERR_INVALID_VALUE_Z  = 10 !! invalid value for redshift
    integer, parameter :: ERR_INVALID_VALUE_K  = 40 !! invalid value for wavenumber

    public :: tf_eisenstein98_init
    public :: tf_eisenstein98
    public :: tf_eisenstein98_nu
    public :: tf_eisenstein98_zb
    public :: tf_eisenstein98_bao
    
contains

    !>
    !! Calculate the quantities related linear transfer functions by Eisenstein & Hu (1998).
    !!
    !! Parameters:
    !!  cm      : cosmo_t - Cosmology parameters.
    !!  version : integer - Which version to use: BAO (1), neutrino (2) or zero baryon.  
    !!  stat    : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_eisenstein98_init(cm, version, stat) 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        integer, intent(in), optional :: version
        integer, intent(out), optional :: stat

        real(dp) :: c1, c2, yd, k_eq, R_eq, R_d, y, Gy
        integer  :: stat2 = 0

        !! growth factor at z=0
        call GC_get_growth(cm, 0.0_dp, dplus0, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_eisenstein98_init - failed to calculate growth.'
        end if

        h     = 0.01*cm%H0
        Omh2  = cm%Om0*h**2
        Obh2  = cm%Ob0*h**2
        theta = cm%Tcmb0 / 2.7_dp
        Nnu   = cm%Nnu
        fb    = Obh2 / Omh2
        fnu   = cm%Onu0 / cm%Om0
        fnb   = fnu + fb
        fc    = 1. - fnb
        fcb   = fc + fb

        !! setting model:
        ps_model = PS_EH98_ZB !! default: zero baryon model
        if ( present(version) ) then 
            if ( version == 1 ) then !! use model with BAO
                ps_model = PS_EH98_BAO
            else if ( ( version == 2 ) .and. ( cm%Onu0 > 0. ) ) then 
                !! use neutrino model, if model has non-zero neutrino content
                ps_model = PS_EH98_NU
            end if 
        end if

        !! redshift at matter-radiation equality (eqn. 1)
        z_eq = 2.5e+04 * Omh2 / theta**4 

        !! redshift at drag epoch (eqn. 2)
        c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
        c2  = 0.238*Omh2**0.223
        z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

        if ( ps_model == PS_EH98_BAO ) then !! model with bao

            !! scale of particle horizon at z_eq
            k_eq = 7.46e-02*Omh2/theta**2
            
            !! ratio of baryon - photon momentum density (eqn. 5)
            R_eq = 31.5*Obh2*theta**(-4.) * (z_eq/1.0e+03)**(-1) ! at z_eq
            R_d  = 31.5*Obh2*theta**(-4.) * (z_d /1.0e+03)**(-1) ! at z_d

            !! sound horizon (eqn. 6)
            s = (2./3./k_eq) * sqrt(6./R_eq) * log((sqrt(1. + R_d) + sqrt(R_d + R_eq)) / (1. + sqrt(R_eq)))
            
            !! silk damping scale (eqn. 7)
            k_silk = 1.6*Obh2**0.52 * Omh2**0.73 * (1. + (10.4*Omh2)**(-0.95))

            !! eqn. 11
            c1 = (46.9*Omh2)**0.670 * (1. + (32.1*Omh2)**(-0.532))
            c2 = (12.0*Omh2)**0.424 * (1. + (45.0*Omh2)**(-0.582))
            alpha_c = c1**(-fb) * c2**(-fb**3)

            !! eqn. 12
            c1 = 0.944*(1. + (458.0*Omh2)**(-0.708))**(-1)
            c2 = (0.395*Omh2)**(-0.0266)
            beta_c = (1. + c1*(fc**c2 - 1.))**(-1.)

            !! eqn. 14-15
            y  = (1. + z_eq) / (1. + z_d)
            Gy = y*( -6*sqrt(1. + y) + (2. + 3*y)*log((sqrt(1. + y) + 1.)/(sqrt(1. + y) - 1.)) )
            alpha_b = 2.07*k_eq*s*(1. + R_d)**(-0.75) * Gy

            !! eqn. 24
            beta_b = 0.5 + fb + (3. - 2*fb)*sqrt((17.2*Omh2)**2 + 1.)

            !! eqn. 23
            beta_node = 8.41*Omh2**0.435
            return

        end if

        !! sound horizon (eqn. 26)
        s = 44.5*log( 9.83/Omh2 ) / sqrt( 1 + 10*Obh2**0.75 ) 

        if ( ps_model == PS_EH98_NU ) then !! use model with neutrino
            
            !! eqn. 14 in EH98 paper
            pc  = 0.25*( 5 - sqrt( 1 + 24.0*fc  ) ) 
            pcb = 0.25*( 5 - sqrt( 1 + 24.0*fcb ) )
            
            !! small-scale suppression (eqn. 15)
            yd       = (1. + z_eq) / (1. + z_d) !! eqn. 3
            alpha_nu = (fc / fcb) * (5. - 2*(pc + pcb)) / (5 - 4*pcb)                                           &
                            & * (1. - 0.533*fnb + 0.126*fnb**3)/(1 - 0.193*sqrt(fnu*Nnu) + 0.169*fnu*Nnu**0.2)  &
                            & * (1. + yd)**(pcb - pc)                                                           &
                            & * (1. + 0.5*(pc - pcb) * (1. + 1./(3. - 4*pc)/(7. - 4*pcb)) * (1 + yd)**(-1))

            !! eqn. 21
            beta_c = (1. - 0.949*fnb)**(-1)

            !! growth factor suppression (eqn. 14)
            yfs_over_q2 = 17.2*fnu*( 1. + 0.488*fnu**(7./6.) ) * (Nnu/fnu)**2

        else !! use zero-baryon model

            !! eqn. 31
            alpha_g = 1. - 0.328*log( 431*Omh2 ) * fb + 0.38*log( 22.3*Omh2 ) * fb**2

        end if

    end subroutine tf_eisenstein98_init

    !>
    !! Calculate the linear transfer function by Eisenstein & Hu (1998), not including BAO.
    !!
    !! Parameters:
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  k     : real    - Wavenumber in 1/Mpc.
    !!  z     : real    - Redshift
    !!  tk    : real    - Value of calculated transfer function.
    !!  dlntk : real    - Value of calculated log-derivative
    !!  stat  : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_eisenstein98_zb(cm, k, z, tk, dlntk, stat)
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        real(dp) :: gamma_eff, dplus, q, t1, t2, t3, dt2, dt3
        integer  :: stat2 = 0

        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_zb - cosmology model is not initialised'
            return
        end if
        !! check if k value is correct
        if ( k <= 0. ) then !! invalid value for k
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_zb - wavenumber k must be positive'
            return
        end if

        !! linear growth
        call GC_get_growth(cm, z, dplus, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_eisenstein98_zb - failed to calculate growth.'
        end if
        dplus = dplus / dplus0 !! normalization

        !! eqn. 30  
        gamma_eff = Omh2 * ( alpha_g + ( 1 - alpha_g ) / ( 1 + ( 0.43*k*s )**4 ) )

        !! eqn. 28
        q = k * ( theta**2 / gamma_eff ) !! convert wavenumber to h/Mpc

        t1 = 2*e + 1.8*q
        t2 = log(t1)
        t3 = 14.2 + 731.0 / ( 1 + 62.5*q )
        
        !! EH transfer function
        tk = t2 / (t2 + t3*q**2) 
        tk = tk * dplus

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            dt2   = 1.8 / t1
            dt3   = -731.0 * 62.5 / ( 1 + 62.5*q )**2
            dlntk = ( dt2 / t2 - (dt2 + dt3*q**2 + 2*q*t3) / (t2 + t3*q**2) ) * q
        end if
        
    end subroutine tf_eisenstein98_zb

    !>
    !! Calculate the linear transfer function by Eisenstein & Hu (1998), not including massive
    !! neutrinos.
    !!
    !! Parameters:
    !!  cm    : cosmo_t - Cosmology parameters.
    !!  k     : real    - Wavenumber in 1/Mpc.
    !!  z     : real    - Redshift
    !!  tk    : real    - Value of calculated transfer function.
    !!  dlntk : real    - Value of calculated log-derivative
    !!  stat  : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_eisenstein98_nu(cm, k, z, tk, dlntk, stat) 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        real(dp) :: gamma_eff, sqrt_alpha, yfs, dnorm
        real(dp) :: dplus, bk, B, q, q_eff, q_nu, t1, t2, t3, dt2, dt3, dbk, dzk, dlndzk
        integer  :: stat2 = 0

        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_nu - cosmology model is not initialised'
            return
        end if
        !! check if k value is correct
        if ( k <= 0. ) then !! invalid value for k
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_nu - wavenumber k must be positive'
            return
        end if

        !! linear growth
        call GC_get_growth(cm, z, dplus, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_eisenstein_nu - failed to calculate growth.'
        end if
        dnorm = 2.5*Omh2*(z_eq + 1.)/h**2
        dplus = dnorm * dplus !! normalization
        
        q = k*theta**2 / Omh2 !! eqn. 5
        
        !! shape paremeter (eqn. 16)
        sqrt_alpha = sqrt(alpha_nu)
        gamma_eff  = Omh2*( sqrt_alpha + (1. - sqrt_alpha)/(1. + (0.43*k*s)**4) )

        !! transfer function T_sup (eqn. 17-20)
        q_eff = k*theta**2 / gamma_eff                  !! effective wavenumber (eqn. 17)
        t1    = e + 1.84*beta_c*sqrt_alpha*q_eff
        t2    = log(t1)                                 !! L (eqn. 19)
        t3    = 14.4 + 325. / (1. + 60.5*q_eff**1.11)   !! C (eqn. 20)
        tk    = t2 / (t2 + t3*q_eff**2)                 !! eqn. 18

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            dt2   = 1.84*beta_c*sqrt_alpha / t1
            dt3   = -325.0 * 60.5 * 1.11*q_eff**0.11 / ( 1 + 60.5*q_eff**1.11 )**2
            dlntk = ( dt2 / t2 - (dt2 + dt3*q_eff**2 + 2*q_eff*t3) / (t2 + t3*q_eff**2) ) * q_eff
        end if

        !! master function T_master (eqn. 22-24)
        q_nu = 3.92*q*sqrt(Nnu) / fnu               !! eqn. 23
        B    = 1.2*fnu**0.64 * Nnu**(0.3 + 0.6*fnu)
        bk   = 1. + B / (q_nu**(-1.6) + q_nu**0.8)  !! eqn. 22
        tk   = tk * bk                              !! eqn. 24

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            dbk   = -(bk - 1.)**2 * (-1.6*q_nu**(-2.6) + 0.8*q_nu**(-0.2)) / B
            dlntk = dlntk + ( dbk / bk ) * q_nu
        end if

        !! suppressed growth factor
        yfs  = yfs_over_q2 * q**2 !! eqn. 14
        if ( cm%include_nu ) then !! Dcbnu (eqn., 12), incl. neutrino
            dzk  = ( fcb**(0.7/pcb) + dplus / (1. + yfs) )**(pcb/0.7) * dplus**(1. - pcb) 
        else !! Dcb (eqn., 13), not incl. neutrino
            dzk  = ( (1. + dplus) / (1. + yfs) )**(pcb/0.7) * dplus**(1. - pcb) 
        end if
        tk = tk * dzk / (dnorm * dplus0)

        if ( present(dlntk) ) then !! 1-st log-derivative w.r.to k
            if ( cm%include_nu ) then !! for Dcbnu
                dlndzk = -(pcb/0.7) / ( fcb**(0.7/pcb) + dplus / (1. + yfs) ) 
                dlndzk =  dlndzk * dplus * 2*yfs / (1. + yfs)**2 
            else !! for Dcb
                dlndzk = -(pcb/0.7) * 2*yfs / (1. + yfs) 
            end if
            dlntk  = dlntk + dlndzk
        end if

    end subroutine tf_eisenstein98_nu

    !>
    !! Calculate the linear transfer function by Eisenstein & Hu (1998), including BAO.
    !!
    !! NOTE: Derivative calculations and equations not verified.
    !!
    !! Parameters:
    !!  cm   : cosmo_t - Cosmology parameters.
    !!  k    : real    - Wavenumber in 1/Mpc.
    !!  z    : real    - Redshift
    !!  tk   : real    - Value of calculated transfer function.
    !!  dlntk: real    - Value of calculated log-derivative
    !!  stat : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_eisenstein98_bao(cm, k, z, tk, dlntk, stat) 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        real(dp) :: dplus
        real(dp) :: q, t0b, t0ab, dt0b, dt0ab, t1, t2, t3, t4, dt2, dt3, dt4, f, df, st
        integer  :: stat2 = 0

        !! check if cosmology model is ready
        if ( .not. cm%is_ready() ) then
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_zb - cosmology model is not initialised'
            return
        end if
        !! check if k value is correct
        if ( k <= 0. ) then !! invalid value for k
            if ( present(stat) ) stat = 1
            write(stderr,'(a)') 'error: tf_eisenstein98_zb - wavenumber k must be positive'
            return
        end if

        !! linear growth
        call GC_get_growth(cm, z, dplus, stat = stat2)
        if ( stat2 .ne. 0 ) then
            if ( present(stat) ) stat = stat2
            write (stderr, '(a)') 'error: tf_eisenstein98_zb - failed to calculate growth.'
        end if
        dplus = dplus / dplus0 !! normalization

        !! eqn. 10
        q = k/Omh2 * theta**2

        !! cdm part (eqn. 17-20)
        f    = 1. / (1. + (k*s/5.4)**4)  !! eqn. 18
        t1   = e + 1.8*beta_c*q
        t2   = log(t1)
        t3   = 14.2 + 386.0/(1. + 69.9*q**1.08)         !! eqn. 20 with alpha_c=1
        t4   = 14.2/alpha_c + 386.0/(1. + 69.9*q**1.08) !! eqn. 20 with alpha_c
        t0b  = t2 / (t2 + t3*q**2)              !! eqn. 19 
        t0ab = t2 / (t2 + t4*q**2)              !! eqn. 19 
        tk   = fc*( f*t0b + (1. - f)*t0ab )     !! eqn. 17

        if ( present(dlntk) ) then ! 1-st q derivative
            df    = -f**2 * (s/5.4)**4 * 4*k**3 * (Omh2/theta**2)
            dt2   = 1.8*beta_c / t1
            dt3   = -386.0/(1. + 69.9*q**1.08)**2 * 69.9*1.08*q**0.08 
            dt4   = dt3
            dt0b  = dt2 / (t2 + t3*q**2) - t2 / (t2 + t3*q**2)**2 * (dt2 + 2*q*t3 + dt3*q**2)
            dt0ab = dt2 / (t2 + t4*q**2) - t2 / (t2 + t4*q**2)**2 * (dt2 + 2*q*t4 + dt4*q**2)
            dlntk = fc*( (f*dt0b + df*t0b) + ((1. - f)*dt0ab - df*t0ab) ) !! dt/dq
        end if

        !! baryon part (eqn. 21)
        f    = sin(k*st) / (k*st) !! spherical bessel function j0
        t1   = e + 1.8*q
        t2   = log(t1)
        t0ab = t2 / (t2 + t3*q**2)
        st   = s / (1. + (beta_node/(k*s))**3)**(1./3.)
        t0b  = t0ab / (1. + (k*s/5.2)**2) + alpha_b / (1. + (beta_b/(k*s))**3) * exp(-(k/k_silk)**1.4)
        tk   = tk + fb * t0b * f

        if ( present(dlntk) ) then ! 1-st q derivative 
            !! NOTE: check the equations!....
            df    = ( cos(k*st) - f ) / (k*st) * (Omh2/theta**2)
            dt2   = 1.8 / t1
            dt0ab = dt2 / (t2 + t3*q**2) - t2 / (t2 + t3*q**2)**2 * (dt2 + 2*q*t3 + dt3*q**2)
            dt0b  = 1.4*k**0.4 * (-1./k_silk)**1.4 +  (beta_b/(k*s))**3 * (3./k) / (1. + (beta_b/(k*s))**3)
            dt0b  = alpha_b / (1. + (beta_b/(k*s))**3) * exp(-(k/k_silk)**1.4) * dt0b
            dt0b  = dt0b - t0ab / (1. + (k*s/5.2)**2)**2 * (s/5.2)**2 * 2*k
            dt0b  = dt0ab / (1. + (k*s/5.2)**2)  + dt0b * (Omh2/theta**2)
            dlntk = dlntk + fb*( f*dt0b + df*t0b ) !! dt/dq

            !! log-derivative
            dlntk = dlntk / tk
        end if
        tk = tk * dplus
        
    end subroutine tf_eisenstein98_bao

    !>
    !! Calculate the linear transfer function by Eisenstein & Hu (1998).
    !!
    !! Parameters:
    !!  cm   : cosmo_t - Cosmology parameters.
    !!  k    : real    - Wavenumber in 1/Mpc.
    !!  z    : real    - Redshift
    !!  tk   : real    - Value of calculated transfer function.
    !!  dlntk: real    - Value of calculated log-derivative
    !!  stat : integer - Status flag. Non-zero for failure.
    !! 
    subroutine tf_eisenstein98(cm, k, z, tk, dlntk, stat) 
        class(cosmo_t), intent(in) :: cm !! cosmology parameters
        real(dp), intent(in) :: k !! wavenumber in 1/Mpc unit 
        real(dp), intent(in) :: z !! redshift
        
        real(dp), intent(out) :: tk
        real(dp), intent(out), optional :: dlntk
        integer , intent(out), optional :: stat

        if ( ps_model == PS_EH98_BAO ) then  
            !! model with bao
            call tf_eisenstein98_bao(cm, k, z, tk, dlntk = dlntk, stat = stat)
        else if ( ps_model == PS_EH98_NU  ) then  
            !! model with neutrino 
            call tf_eisenstein98_nu(cm, k, z, tk, dlntk = dlntk, stat = stat)
        else 
            !! zero-baryon model (default)
            call tf_eisenstein98_zb(cm, k, z, tk, dlntk = dlntk, stat = stat)
        end if
                
    end subroutine tf_eisenstein98
    
end module transfer_eh