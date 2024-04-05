module load_options
    use iso_fortran_env, only: dp => real64, stderr => error_unit
    implicit none

    private

    type, public :: options_t
        character(len=32) :: model_mps = ''
        character(len=32) :: model_hmf = ''
        character(len=32) :: model_hbf = ''
        character(len=32) :: output_file_redshift  = ''
        character(len=32) :: output_file_powerspec = ''   
        character(len=32) :: output_file_halos     = ''   
        integer  :: cosmo_flat         = 0
        integer  :: include_nu         = 0
        integer  :: settings_gi_size   = 0
        integer  :: settings_di_size   = 0
        integer  :: settings_vi_size   = 0
        real(dp) :: settings_vi_lnka   = 0._dp
        real(dp) :: settings_vi_lnkb   = 0._dp
        real(dp) :: cosmo_H0           = 0._dp
        real(dp) :: cosmo_Om0          = 0._dp
        real(dp) :: cosmo_Ob0          = 0._dp
        real(dp) :: cosmo_Ode0         = 0._dp
        real(dp) :: cosmo_ns           = 0._dp
        real(dp) :: cosmo_sigma8       = 0._dp
        real(dp) :: cosmo_w0           = 0._dp
        real(dp) :: cosmo_wa           = 0._dp
        real(dp) :: Delta              = 200._dp
        real(dp) :: z_range(3) = [ 0._dp, 0._dp, 0._dp ] 
        real(dp) :: k_range(3) = [ 0._dp, 0._dp, 0._dp ] 
        real(dp) :: m_range(3) = [ 0._dp, 0._dp, 0._dp ] 
    end type options_t

    public :: load_options_from_file
    
contains

    !>
    !! Load calculator options from a file. This file must contain the required  
    !! options in a key-value format. The charecters # or ; can be used for comments.
    !!
    !! Format: `property[.subproperty]=value`
    !!
    !! Parameters:
    !!  file : string    - Filename 
    !!  opts : options_t - Loaded options
    !!  stat : integer   - Status
    !!
    subroutine load_options_from_file(file, opts, stat)
        character(len=32), intent(in)  :: file
        type(options_t), intent(out)   :: opts
        integer, intent(out), optional :: stat
        
        integer, parameter :: fh = 15
        integer :: pos, ios = 0, line = 0, ierror = 0
        logical :: file_exist
        character(len=128) :: buffer, label, sublabel

        !! check if the file exist or not
        inquire(file = file, exist = file_exist)
        if ( .not. file_exist ) then
            write (stderr, '(a)') 'error: file '//trim(file)//' does not exist'
            if ( present(stat) ) stat = 1
            return
        end if
        write(stderr,'(a)') 'info: loading options from file - '//trim(file)//''

        !! parsing the options file
        open(fh, file = file)
        do while (ios == 0)
            read(fh, '(a)', iostat = ios) buffer
            if ( ios == 0 ) then
                line = line + 1

                pos = scan(buffer, '#;')                !! get comments, if any
                if ( pos > 0 ) buffer = buffer(1:pos-1) !! remove comments
                if ( len( trim(buffer) ) < 1 ) cycle    !! skip empty lines

                !! get key-value pairs: general line structure is label[.sublabel]=value
                pos    = scan(buffer, '=')
                ierror = 1
                if ( pos < 1 ) exit 
                label  = buffer(1:pos-1)
                buffer = buffer(pos+1:)
                if ( len( trim(buffer) ) < 1 ) exit !! no value for this item
                if ( len( trim(label ) ) < 1 ) exit !! no label for this item
                pos      = scan(label, '.')
                sublabel = ''
                if ( pos > 0 ) then 
                    sublabel = label(pos+1:)
                    label    = label(1:pos-1)
                    if ( len( trim(label   ) ) < 1 ) exit !! no main label for this item
                    if ( len( trim(sublabel) ) < 1 ) exit !! no sub label for this item
                end if
                ierror = 0

                select case ( label )
                case ( 'cosmology' )  !! cosmology model parameters
                    select case ( sublabel )
                    case ( 'flat' )
                        read(buffer, *, iostat = ios) opts%cosmo_flat
                    case ( 'Hubble' )
                        read(buffer, *, iostat = ios) opts%cosmo_H0
                    case ( 'Omega_matter' )
                        read(buffer, *, iostat = ios) opts%cosmo_Om0
                    case ( 'Omega_baryon' )
                        read(buffer, *, iostat = ios) opts%cosmo_Ob0
                    case ( 'Omega_darkEnergy' )
                        read(buffer, *, iostat = ios) opts%cosmo_Ode0
                    case ( 'power_index' )
                        read(buffer, *, iostat = ios) opts%cosmo_ns
                    case ( 'sigma8' )
                        read(buffer, *, iostat = ios) opts%cosmo_sigma8
                    case ( 'w0' )
                        read(buffer, *, iostat = ios) opts%cosmo_w0
                    case ( 'wa' )
                        read(buffer, *, iostat = ios) opts%cosmo_wa
                    end select
                case ( 'model' ) !! model id
                    pos = scan(buffer, '"')
                    if ( pos > 0 ) then
                        ierror = 1
                        buffer = buffer(pos+1:) 
                        pos    = scan(buffer, '"')
                        if ( pos == 0 ) exit !! no closing `"`
                        buffer = buffer(1:pos-1)
                        ierror = 0
                    end if
                    select case ( sublabel )
                    case ( 'power_spectrum' )
                        opts%model_mps = trim(buffer)
                    case ( 'halo_mass_function' )
                        opts%model_hmf = trim(buffer)
                    case ( 'halo_bias_function' )
                        opts%model_hbf = trim(buffer)
                    end select
                case ( 'output_file' ) !! output file
                    pos = scan(buffer, '"')
                    if ( pos > 0 ) then
                        ierror = 1
                        buffer = buffer(pos+1:) 
                        pos    = scan(buffer, '"')
                        if ( pos == 0 ) exit !! no closing `"`
                        buffer = buffer(1:pos-1)
                        ierror = 0
                    end if
                    select case ( sublabel )
                    case ( 'redshift' )
                        opts%output_file_redshift = trim(buffer)
                    case ( 'powerspec' )
                        opts%output_file_powerspec = trim(buffer)
                    case ( 'halos' )
                        opts%output_file_halos = trim(buffer)
                    end select
                case ( 'settings' )  !! main settings
                    select case ( sublabel )
                    case ( 'growth_integ_size' )
                        read(buffer, *, iostat = ios) opts%settings_gi_size
                    case ( 'dist_integ_size' )
                        read(buffer, *, iostat = ios) opts%settings_di_size
                    case ( 'var_integ_size' )
                        read(buffer, *, iostat = ios) opts%settings_vi_size
                    case ( 'var_integ_lnka' )
                        read(buffer, *, iostat = ios) opts%settings_vi_lnka
                    case ( 'var_integ_lnkb' )
                        read(buffer, *, iostat = ios) opts%settings_vi_lnkb
                    end select
                case ( 'redshift' ) !! redshift range
                    select case ( sublabel )
                    case ( 'start' )
                        read(buffer, *, iostat = ios) opts%z_range(1)
                    case ( 'stop' )
                        read(buffer, *, iostat = ios) opts%z_range(2)
                    case ( 'step' )
                        read(buffer, *, iostat = ios) opts%z_range(3)
                    end select
                case ( 'wavenum' ) !! wavenumber range (log-scale)
                    select case ( sublabel )
                    case ( 'start' )
                        read(buffer, *, iostat = ios) opts%k_range(1)
                    case ( 'stop' )
                        read(buffer, *, iostat = ios) opts%k_range(2)
                    case ( 'step' )
                        read(buffer, *, iostat = ios) opts%k_range(3)
                    end select
                case ( 'mass' ) !! mass range (log-scale)
                    select case ( sublabel )
                    case ( 'start' )
                        read(buffer, *, iostat = ios) opts%m_range(1)
                    case ( 'stop' )
                        read(buffer, *, iostat = ios) opts%m_range(2)
                    case ( 'step' )
                        read(buffer, *, iostat = ios) opts%m_range(3)
                    end select
                case ( 'include_nu' )
                    read(buffer, *, iostat = ios) opts%include_nu
                case ( 'Delta' )
                    read(buffer, *, iostat = ios) opts%Delta
                end select
            end if
        end do

        if ( present(stat) ) stat = ierror
        if ( ierror .ne. 0 ) then
            write (stderr, '(a,i0,a)') 'error: at line ', line, ' in '//trim(file)
            return
        end if
        write (stderr, '(a)') 'info: options loaded successfully'
        
    end subroutine load_options_from_file

end module load_options

