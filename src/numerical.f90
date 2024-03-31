module numerical
    use constants
    implicit none

    private

    public :: generate_gaussleg, fzero

    interface 
        function obj_function(x, args) result(retval)
            use constants, only: dp
            real(dp), intent(in) :: x !! function argument
            real(dp), intent(in), optional :: args(:) !! additional arguments
            real(dp) :: retval 
        end function
    end interface

    !! Object to hold integration rule
    type, public :: quadrule
        integer :: n !! size of the rule
        real(dp), allocatable :: x(:) !! nodes 
        real(dp), allocatable :: w(:) !! weights
        real(dp) :: a = -1.0_dp !! lower integration limit
        real(dp) :: b =  1.0_dp !! upper integration limit
    end type 
    
contains

    !>
    !! Swap the arguments.
    !!
    subroutine swap(arg1, arg2)
        real(dp), intent(inout) :: arg1, arg2
        real(dp) :: tmp
        tmp  = arg1; arg1 = arg2; arg2 = tmp
    end subroutine swap

    !>
    !! Generate gauss-legendre integration points and weights
    !! 
    !! Parameters:
    !!  n  : integer  - Number of points to use. Must be a positive non-zero integer.
    !!  obj: quadrule - Object storing generted nodes and weights arrays.
    !!
    subroutine generate_gaussleg(n, obj)
        integer, intent(in) :: n    !! order of the rule: number of points
        type(quadrule), intent(out) :: obj !! storage object
        
        ! real(dp), intent(out) :: x(n) !! integration nodes (points)
        ! real(dp), intent(out) :: w(n) !! weights
        real(dp) :: wj, xj, xj_old, pm, pn, ptmp
        integer  :: j, k

        !! allocating memory for the rule
        obj%n = n
        allocate( obj%x(n) )
        allocate( obj%w(n) )

        !! for odd order, x = 0 is the {floor(n/2)+1}-th node
        if ( modulo(n, 2) == 1 ) then
            
            !! calculating lenegedre polynomial P_n(0) using its reccurence relation
            xj = 0.d0
            pm = 0.d0
            pn = 1.d0
            do k = 0, n-1
                ptmp = -k*pm / (k + 1.d0)
                pm   = pn
                pn   = ptmp
            end do
        
            obj%x(n/2 + 1) = 0.d0
            obj%w(n/2 + 1) = 2.d0 / (n*pm)**2 !! weight 
        end if

        !! other nodes
        do j = 1, n/2

            !! initial guess for j-th node (j-th root of n-th legendre polynomial)
            xj = cos( (2.d0*j - 0.5d0) * PI / (2.d0*n + 1.d0) ) 

            do 
                !! calculating lenegedre polynomial P_n(xj) using its reccurence relation
                pm = 0.d0
                pn = 1.d0
                do k = 0, n-1
                    ptmp = ( (2.d0*k + 1.d0)*xj*pn - k*pm ) / (k + 1.d0)
                    pm   = pn
                    pn   = ptmp
                end do
                
                !! next estimate of the root
                xj_old = xj
                xj     = xj - pn * (xj**2 - 1.d0) / (n*xj*pn - n*pm)

                if ( abs(xj - xj_old) < EPS ) exit !! result converged!

            end do

            wj = 2.d0 * (1.d0 - xj**2) / (n*xj*pn - n*pm)**2 !! weight for j-th node

            obj%x(j) = -xj
            obj%w(j) =  wj

            !! weights are symmetric about x = 0
            obj%x(n-j+1) = xj
            obj%w(n-j+1) = wj
            
        end do

    end subroutine generate_gaussleg

    !>
    !! Find a root of the function f(x) in the interval [a, b], if exist.
    !!
    !! Parameters:
    !!  f      : procedure - Function with signature y = f(x)
    !!  a, b   : real      - Interval containing the root.
    !!  retval : real      - Root  
    !!  status : integer   - Result status. 0=success, 1=no root exist, 2=maximum iterations reached.
    !!  xtol   : real      - Tolerance for the result.
    !!  maxiter: integer   - Maximum iterations to use.
    !!  args   : real      - Additional arguments to function call
    !!
    subroutine fzero(f, a, b, retval, status, xtol, maxiter, args)
        procedure(obj_function) :: f    !! function to find the root
        real(dp), intent(in)    :: a, b !! bracketting interval
        real(dp), intent(in), optional :: xtol    !! tolerance level
        integer, intent(in) , optional :: maxiter !! maximum iterations
        real(dp), intent(in), optional :: args(:) !! additional arguments to function call
        
        real(dp), intent(out)  :: retval  !! result: value of the root
        integer, intent(out) :: status  !! failure flag

        real(dp), parameter :: delta = 1e-08

        real(dp) :: xa, xb, xc, xd, xs, fa, fb, fc, fs, xtol_used 
        integer  :: mflag, iter, maxiter_used

        if ( present(xtol) ) then
            xtol_used = xtol
        else
            xtol_used = 1.0e-06_dp
        end if

        if ( present(maxiter) ) then
            maxiter_used = maxiter
        else
            maxiter_used = 1000
        end if


        status  = 0 !! default - meaning success

        xa = a
        xb = b
        fa = f(a, args)
        fb = f(b, args)
        if ( fa*fb >= 0. ) then !! root is not bracketed
            status = 1
            return
        end if

        if ( abs(fa) < abs(fb) ) then !! swap a and b
            call swap(xa, xb)
            call swap(fa, fb)
        end if 

        xc    = xa
        mflag = 1
        do iter = 1, maxiter_used
            fc = f(xc, args)

            if ( ( abs(fa - fc) > EPS ) .and. ( abs(fb - fc) > EPS ) ) then
                !! inverse quadratic interpolation
                xs = xa*fb*fc / (fa - fb) / (fa - fc) + fa*xb*fc / (fb - fa) / (fb - fc) + fa*fb*xc / (fc - fa) / (fc - fb)
            else 
                !! secant method
                xs = xb - fb * (xb - xa) / (fb - fa)
            end if

            if ( & 
                 &     (( xs < 0.25*(3*xa + xb) ) .or. ( xs > xb ))                 & !! condition 1
                 & .or. (( mflag == 1 ) .and. ( abs(xs - xb) >= abs(xb - xc)*0.5 )) & !! condition 2
                 & .or. (( mflag == 0 ) .and. ( abs(xs - xb) >= abs(xc - xd)*0.5 )) & !! condition 3
                 & .or. (( mflag == 1 ) .and. ( abs(xb - xc) < abs(delta) ))        & !! condition 4
                 & .or. (( mflag == 0 ) .and. ( abs(xc - xd) < abs(delta) ))        & !! condition 5
                 & ) then
                !! bisection method
                xs = 0.5 * (xa + xb)
                mflag = 1
            else !! not used bisection
                mflag = 0
            end if

            fs = f(xs, args)
            xd = xc
            xc = xb

            if ( fa*fs < 0. ) then
                xb = xs 
                fb = fs 
            else 
                xa = xs
                fa = fs
            end if

            if ( abs(fa) < abs(fb) ) then !! swap a and b
                call swap(xa, xb)
                call swap(fa, fb)
            end if 

            !! return the root if converged
            if (( abs(fs) < EPS ) .or. ( abs(xb - xa) < xtol_used )) then
                retval = xs
                return
            else if ( abs(fb) < EPS ) then
                retval = xb
                return
            end if

        end do

        !! root not found after all iterations
        status = 2
        
    end subroutine fzero
    
end module numerical
