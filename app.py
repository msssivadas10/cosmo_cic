#!/usr/bin/python3
#
# Application for measuring 2D count-in-cells
# @author m. s. sūryan śivadās
#

__version__ = '2.0a'
prog_name   = 'cic'
prog_info   = 'Do 2D count-in-cells analysis on data.'

import sys

def raiseErrorAndExit(__msg: str):
    r"""
    Write an error message `__msg` to stderr and exit. 
    """
    
    sys.stderr.write("\033[1m\033[91mError:\033[m %s\n" % __msg)
    sys.exit(1)

if sys.version_info < (3, 10):
    raiseErrorAndExit("app requires python version >= 3.10")

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    raiseErrorAndExit("cannot import 'mpi4py', module not found.")

comm       = MPI.COMM_WORLD
rank, size = comm.rank, comm.size 

import os
import re
import logging
import yaml
from string import Template
from argparse import ArgumentParser
from cic.measure.counting import *


################################################################################################
# Default / constant values
################################################################################################

default_x_coord        = "ra"
default_y_coord        = "dec"
default_mask           = "%(band)s_mask"
default_allbands       = ['g','r','i','z','y']
default_magnitude      = "%(band)s_magnitude"
default_mag_offset     = "%(band)s_magnitude_offset"
default_redshift       = "redshift"
default_redshift_error = "redshift_err"

################################################################################################
# Argument parser object
################################################################################################

parser = ArgumentParser(prog = prog_name, description = prog_info)
parser.add_argument('--file', help = 'path to the input options file', type = str  )
parser.add_argument('--flag', help = 'flags to control the execution', type = int, default = 1)


################################################################################################
# Helper functions
################################################################################################

def loadOptions(file: str) -> dict:
    r"""
    Load options from a YAML / JSON file `file` and return as a dict.
    """
    
    if not os.path.exists( file ):
        logging.error( f"file does not exist '{ file }'; exiting" )
        raiseErrorAndExit( f"file does not exist '{ file }'" )

    try:
        with open( file, 'r' ) as fp:
            options = yaml.safe_load( fp )
    except Exception as _e:
        logging.error( f"failed to load options from '{ file }'. { _e }; exiting" )
        raiseErrorAndExit( f"failed to load options from '{ file }'. { _e }" )

    return options


def replaceFields(string: str, mapping: dict) -> str:
    r"""
    Variable substitutions on `string`. Variables are indicated by a `$` sign in front. 
    A dict `mapping` will tell the values to replace.

    >>> replaceFields("$x < 0", {"x": y})
    y < 0

    """

    fields = re.findall( r'\$(\w+)', string )
    for field in fields:
        assert field  in mapping.keys(), f"field '{ field }' has no replacement in the mapper" 
        
    return Template( string ).substitute( mapping )


def loadExpressions(__exprs: list[str], all_bands: list[str], mapping: dict = None) -> list[str]:
    r"""
    Finalise a set of expressions by substituting band names and other variables.
    """

    assert all( map( lambda __o: isinstance(__o, str), __exprs ) ), "an expression must be a 'str'"

    all_expressions = []
    for expr in __exprs:
        if re.search( r"\%\(band\)s", expr ):
            all_expressions.extend([ expr % { 'band': band } for band in all_bands])
        else:
            all_expressions.append( expr )

    if mapping:
        for i, expr in enumerate( all_expressions ):
            all_expressions[i] = replaceFields( expr, mapping )

    return all_expressions


def getValue(key: str, __dict: dict, default_value: object = None, dtype: type = None) -> object:
    r"""
    Look for a value with key `key` in the dict `__dict`. If not present, return the given 
    default value. Optionally check the data type also. 
    """

    def typename(dtype: type) -> str: # name of a type
        
        assert isinstance( dtype, type ), "'dtype' must be a python type"
        __t, = re.match(r'\<class \'([\w\.]+)\'\>', str( dtype )).groups()
        return __t
    
    def typename2(dtype: type | tuple[type]) -> str: # name of one or more types (joined)

        if not isinstance( dtype, tuple ):
            return typename( dtype )
        return ' or '.join( ', '.join( map(lambda __o: f"'{typename( __o )}'", dtype) ).rsplit(', ', maxsplit = 1) )


    value = __dict.get( key, default_value )
    if dtype is not None:
        if not isinstance( value, dtype ):
            raise TypeError( f"'{key}' must be a { typename2(dtype) }. got '{ typename(type(value)) }'" )

    return value


def getCSVReadOptions(csv_opts: dict) -> dict:
    r"""
    Return the settings for reading a CSV file using `pandas.read_csv` as dict. 
    """

    csv_opts["compression"] = getValue( 'compression', csv_opts, 'infer', str) 
    csv_opts["chunksize"]   = getValue( 'chunksize',   csv_opts, 1,       int)     

    # if both column names and header not given, first row is column names by default
    if csv_opts.get("names") is None and csv_opts.get("header") is None:
        csv_opts["header"] = 0

    return csv_opts


################################################################################################
# Counting process
################################################################################################

def initialize(file: str) -> dict:
    r"""
    Initialise the counting process: loading option file configuring logging. This will 
    return the loaded options as a dict.
    """

    # create log files if not exist
    if rank == 0:
        if not os.path.exists( "logs" ):
            try:
                os.mkdir( "logs" )
            except Exception as e:
                raiseErrorAndExit( "creating log directory raised exception '%s'" % e )
    comm.Barrier()

    log_file = os.path.join( "logs", "%d.log" % rank )
    if not os.path.exists( log_file ):
        open(log_file, 'w').close()

    # configure logger
    logging.basicConfig(level    = logging.INFO,
                        format   = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [
                            logging.FileHandler(log_file, mode = 'w'),
                            logging.StreamHandler()
                        ])

    # load options file
    logging.info( "loading options from file '%s'" % file )
    options = loadOptions( file )

    comm.Barrier()
    return options


def createPatches(options: dict) -> None:
    r"""
    Create patches on the specified region and calculate cell goodness factors using random
    objects. 
    """

    try:
        
        #
        # load / check options
        #

        # total region to used for calculations
        region = getValue( "region", options, [], list )

        # list of bad regions to cut out from total
        bad_regions = getValue( "bad_regions", options, [], list )

        # patchsizes
        patchsize_x = getValue( 'patchsize_x', options, dtype = (int, float) ) 
        patchsize_y = getValue( 'patchsize_y', options, dtype = (int, float) ) 

        # cellsize
        pixsize = getValue( "pixsize", options, dtype = (int, float) )

        # random catalog options
        cat_opts = getValue( "random_catalog", options, dtype = dict )

        # catalog file path
        path = getValue( "path", cat_opts, None, str ) 

        # csv reading options
        csv_opts = getCSVReadOptions( getValue( "csv_opts", cat_opts, {}, dict ) )

        # column names
        features = {
                        "mask"   : default_mask,
                        "x_coord": default_x_coord,
                        "y_coord": default_y_coord,
                   }
        for feature in features:
            features[ feature ] = getValue( feature, cat_opts, features[ feature ], str )

        # all bands used in the survey
        all_bands = getValue( "all_bands", options, default_allbands, list )
        assert all( map( lambda __o: isinstance(__o, str), all_bands ) ), "'all_bands' must be a array of 'str'"

        # set of masks to use
        mask_bands  = getValue( "random_masks", options, [], list )
        assert all( map( lambda __o: isinstance(__o, str), mask_bands ) ), "'random_masks' must be a array of 'str'"

        use_masks   = []
        for band in mask_bands:
            assert band in all_bands, "%s is not a valid band name" % band
            use_masks.append( features["mask"] % { 'band': band } )

        # data filters for random
        all_filters = loadExpressions( getValue( "filters", cat_opts, [], list ), all_bands )

        # expressions to evaluate on 
        all_expressions = loadExpressions( getValue( "expressions", cat_opts, [], list ), all_bands )

        # output file path (patches)
        output_path = getValue( "patch_details_path", options, None, str )


        #
        # creating the patches
        #
        prepareRegion(path         = path, 
                      output_path  = output_path, 
                      region       = region, 
                      patchsize_x  = patchsize_x, 
                      patchsize_y  = patchsize_y, 
                      pixsize      = pixsize, 
                      bad_regions  = bad_regions, 
                      use_masks    = use_masks, 
                      data_filters = all_filters, 
                      expressions  = all_expressions, 
                      x_coord      = features["x_coord"], 
                      y_coord      = features["y_coord"], 
                      **csv_opts      )

    except CountingError:
        pass # handled already!

    except Exception as _e:
        _msg = f"failed to create patches. '{ _e }'"
        logging.error( _msg )
        raiseErrorAndExit( _msg ) 

    return


def measureCountInCells(options: dict) -> None:
    r"""
    Count the number of objects in already created cells.
    """

    try:

        #
        # load / check options
        #

        # object catalog options
        cat_opts = getValue( "object_catalog", options, dtype = dict )

        # catalog file path
        path = getValue( "path", cat_opts, None, str ) 

        # csv reading options
        csv_opts = getCSVReadOptions( getValue( "csv_opts", cat_opts, {}, dict ) )

        # column names
        features = {
                        "mask"          : default_mask,
                        "x_coord"       : default_x_coord,
                        "y_coord"       : default_y_coord,
                        "magnitude"     : default_magnitude,
                        "mag_offset"    : default_mag_offset,
                        "redshift"      : default_redshift,
                        "redshift_error": default_redshift_error,
        }
        for feature in features:
            features[ feature ] = getValue( feature, cat_opts, features[ feature ], str )

        # all bands used in the survey
        all_bands = default_allbands if not options["all_bands"] else options["all_bands"]
        assert all( map( lambda __o: isinstance(__o, str), all_bands ) ), "'all_bands' must be a array of 'str'"

        # set of masks to use
        mask_bands  = [] if not options["object_masks"] else options["object_masks"]
        assert all( map( lambda __o: isinstance(__o, str), mask_bands ) ), "'object_masks' must be a array of 'str'"
        
        use_masks   = []
        for band in mask_bands:
            assert band in all_bands, "%s is not a valid band name" % band
            use_masks.append( features["mask"] % { 'band': band } )

        # mapping for replacing $-fields in filters and expressions
        mapping = {'redshift': features["redshift"], 'redshift_err': features["redshift_error"]}
        for band in all_bands:
            mapping[band]            = features["magnitude"]  % {'band': band}
            mapping['%s_off' % band] = features["mag_offset"] % {'band': band} 

        # data filters for random
        all_filters = loadExpressions( getValue( "filters", cat_opts, [], list ), all_bands )

        # expressions to evaluate on 
        all_expressions = loadExpressions( getValue( "expressions", cat_opts, [], list ), all_bands )

        # patch details file
        patch_details_path = getValue( "patch_details_path", options, None, str )

        # output file path (counts)
        output_path = getValue( "count_results_path", options, None, str )
        
        #
        # counting objects on cells
        #
        countObjects(path                  = path,
                     patch_details_path    = patch_details_path,
                     output_path           = output_path,
                     include_patch_details = True,
                     use_masks             = use_masks,
                     data_filters          = all_filters,
                     expressions           = all_expressions,
                     x_coord               = features["x_coord"],
                     y_coord               = features["y_coord"],
                     **csv_opts                                  )
        
    except CountingError:
        pass # handled already!

    except Exception as _e:
        _msg = f"failed to count objects. '{ _e }'"
        logging.error( _msg )
        raiseErrorAndExit( _msg ) 

    return


def main() -> None:

    # parse arguments
    args = parser.parse_args()
    flag = args.flag
    file = args.file

    if not file:
        return

    # initialising...
    options = initialize( file )
    if flag & 8:
        logging.info("exiting after successfull initialization :)")
        return
    
    # preparing patch data by counting randoms...
    if flag & 48:
        logging.warning("skipping patch calculations")
    else:
        createPatches( options )
        if flag & 4:
            logging.info("exiting after successfull patch generation :)")
            return

    # counting objects... 
    if flag & 32:
        logging.warning("skipping object counting")
    else:
        measureCountInCells( options )
        if flag & 2:
            logging.info("exiting after successfull object counting :)")
            return 
        
    return

if __name__ == '__main__':
    main()
