#!/usr/bin/python3

import os, time
import logging
import numpy as np, pandas as pd
from scipy.stats import binned_statistic_dd
from typing import Any 
from .utils2 import MeasurementError, Box, Rectangle, Box3D, CountResult

DISABLE_MPI = False # switch indicating if to disable parallel processing with MPI

try:
    from mpi4py import MPI    
except ModuleNotFoundError:
    DISABLE_MPI = True # disable using MPI
    logging.warning( "module 'mpi4py' is not found, disabling parallel processing" )


def getCommunicatorInfo():
    r"""
    Return the MPI communication details: the communicator object, rank and size 
    of the process. 
    """

    if DISABLE_MPI:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.rank, comm.size

def setBarrier(comm):
    r"""
    Wrapper over the MPI `Barrier` method.
    """

    if comm is None: 
        return 
    return comm.Barrier()

def raiseError(msg :str):
    r"""
    Raise a `CountingError` exception, and log the message.
    """

    logging.error( msg )
    raise MeasurementError( msg )


########################################################################################################
# Base function to count objects
########################################################################################################

def countObjectsRectangularCell(path: str, 
                                region: Box, 
                                patchsize: float | list[float], 
                                pixsize: float | list[float],
                                use_masks: list[str] = [], 
                                data_filters: list[str] = [], 
                                expressions: list[str] = [],
                                coord: list[str] = None,
                                **csv_opts: Any,       ) -> Any:
    r"""
    Count the number of objects in general n-D space, with recatangular cells.
    """

    if not isinstance(region, Box):
        raiseError( "region should be an instance of Box or its subclass" )

    patchsize = np.ravel( patchsize )
    pixsize   = np.ravel( pixsize )

    if coord is None:
        coord = [ 'x%d' % x for x in range( region.dim ) ]
    elif len( coord ) != region.dim:
        raiseError("coords list should have same size as the spatial dimension")


    # setting up MPI
    comm, RANK, SIZE = getCommunicatorInfo()

    # number of patches along each direction:
    x_min, x_max = np.ravel( region.min ), np.ravel( region.max )
    x_patches    = np.floor( ( x_max - x_min ) / patchsize ).astype('int')
    patch_count  = np.prod( x_patches ) # total number of patches

    # bin edges
    res_shape = np.floor( patchsize / pixsize ).astype('int') # number of cells along each direction
    x_bins    = []
    for x in range( region.dim ):
        x_bins.append( np.linspace( 0., patchsize[x], res_shape[x] + 1 ) ) # cell edges
    patch_bins = np.arange( patch_count + 1 ) - 0.5 # patch bins are centered at its index
    res_shape  = tuple( *res_shape, patch_count )   # shape of the count result array

    logging.info(f"counting with cellsize = {tuple(pixsize)}, cell arrangement shape = {res_shape[:-1]}, {patch_count} patches.")

    # extend data filters by adding boundary limits
    data_filters = [*data_filters, 
                    *[ f'{ coord[x] } >= { x_min[x] }' for x in range( region.dim ) ], 
                    *[ f'{ coord[x] } <= { x_max[x] }' for x in range( region.dim ) ], ]
    data_filters = ' & '.join( map( lambda _filt: "(%s)" % _filt, data_filters ) )
    
    # mask expressions is considered sperately from other expressions
    mask_expressions = ' & '.join( map( lambda _mask: "(%s == False)" % _mask, use_masks ) )

    logging.info( "data filter expression: '%s'", data_filters )
    if mask_expressions:
        logging.info( "mask expression: '%s'", mask_expressions )
    if expressions:
        logging.info( "other expression: '%s'", expressions )

    #
    # counting objects in parellel 
    #

    logging.info( "started counting objects in cell" )
    __t_init = time.time()

    # before starting calculations, try if the data can be loaded properly with given options:
    try:
        with pd.read_csv(path, **csv_opts ) as df_iter:
            pass
    except Exception as _e:
        raiseError( f"failed to load data from '{ path }'. { _e }" )

    total, unmasked = np.zeros( res_shape ), np.zeros( res_shape )
    with pd.read_csv(path, **csv_opts ) as df_iter:
        
        chunk = 0
        used_objects, total_objects = 0, 0
        for df in df_iter:

            # distribute chunks among different process
            if chunk % SIZE != RANK:
                chunk = chunk + 1
                continue
            chunk = chunk + 1

            total_objects += df.shape[0]

            # evaluate expressions: eg., magnitude corrections
            for expression in expressions:
                try:
                    df = df.eval( expression ) 
                except Exception as _e:
                    raiseError( f"cannot evaluate expression '{ expression }', { _e }" )

            # apply various data filters:   
            try:
                df = df.query( data_filters ).reset_index( drop = True )
            except Exception as _e:
                raiseError( f"cannot filter data, { _e }" )
            
            if not df.shape[0]:
                continue # no data to use
            used_objects += df.shape[0]

            # get mask weight: 
            if mask_expressions:
                mask_weight = df.eval( mask_expressions ).to_numpy().astype( 'float' )
            else:
                mask_weight = np.ones( df.shape[0] )

            # 
            # convert from nD physical coordinates (x1...xn) to (n+1)D patch coordinates (x'1...x'n, patch) 
            # with x'i in [0, patchsize_i] and patch in [0, patch_count]
            #  
            df[coord] = df[coord] - x_min # coordinates relative to region

            xp = np.floor( df[coord] / patchsize ).astype('int')  # nD patch index
            df['patch'] = np.ravel_multi_index( xp.T, x_patches ) # patch index (flattened)

            # coordinates relative to patch
            df[coord] = df[coord] - xp * patchsize 

            # chunk counts
            total_c, unmasked_c = binned_statistic_dd(df[[*coord, 'patch']].to_numpy(),
                                                      values    = [ np.ones(df.shape[0]), mask_weight ], # weights
                                                      statistic = 'sum',
                                                      bins      = [ *x_bins, patch_bins ], ).statistic
            
            total, unmasked = total + total_c, unmasked + unmasked_c

    logging.info( "used %d objects out of %d for counting", used_objects, total_objects )
    logging.info( "finished counting in %g seconds :)", time.time() - __t_init )

    setBarrier(comm) # syncing...

    #
    # since using multiple processes, combine counts from all processes
    #
    if RANK != 0:

        logging.info( "sending data to rank-0" )

        # send data to process-0
        comm.Send( total,    dest = 0, tag = 10 ) # total count
        comm.Send( unmasked, dest = 0, tag = 11 ) # unmasked count

    else:

        # recieve data from other processes
        tmp = np.zeros( res_shape ) # temporary storage
        for src in range(1, SIZE):

            logging.info( "recieving data from rank-%d", src )
            
            # total count
            comm.Recv( tmp, source = src, tag = 10,  )
            total = total + tmp

            # exposed count
            comm.Recv( tmp, source = src, tag = 11,  )
            unmasked = unmasked + tmp

    setBarrier(comm) # syncing...

    # 
    # return the result at rank-0
    #
    if RANK != 0: return (None, None)
    return (total, unmasked)


########################################################################################################
# Object counting for CIC
########################################################################################################

def prepareRegion(path: str, 
                  output_path: str,
                  region: Box, 
                  patchsize: float | list[float], 
                  pixsize: float | list[float],
                  bad_regions: list[Box] = [],
                  use_masks: list[str] = [], 
                  data_filters: list[str] = [], 
                  expressions: list[str] = [],
                  coord: list[str] = None,
                  **csv_opts: Any        ) -> None:
    r"""
    Prepare a region for counting. 
    """

    try:
        region = Box.create( region )
    except Exception as _e:
        raiseError( f"error converting region to 'Box': {_e}" )

    #
    # dividing the region into patches
    #

    logging.info( "checking and correcting patch sizes" )

    if np.size( patchsize ) == 1:
        patchsize = np.repeat( patchsize, region.dim )
    elif np.size( patchsize ) != region.dim:
        raiseError( "patchsize should have same size as the spatial dimension" )
    if np.any( patchsize <= 0 ):
        raiseError( "one or more patchsizes are zero or negative" )

    if np.size( pixsize ) == 1:
        pixsize = np.repeat( pixsize, region.dim )
    elif np.size( pixsize ) != region.dim:
        raiseError( "pixsize should have same size as the spatial dimension" )
    if np.any( pixsize <= 0 ) or np.any( pixsize > patchsize ):
        raiseError( "one or more pixsizes are zero, negative or bigger than the patchsize" )

    try:
        if bad_regions is None:
            bad_regions = []
        bad_regions = [ Box.create( bad_region ) for bad_region in bad_regions ]
    except Exception as _e:
        raiseError( f"error converting bad region to 'Box': {_e}" )

    # correcting the patchsizes to be exacly equal to a nearest multiple of pixsize
    # this make sure that the cell bins are correctly calculated in the base function
    x_cells   = np.floor( patchsize / pixsize ).astype('int')
    patchsize = x_cells * pixsize

    # diving region into patches not overlapping with a bad region
    patches      = [] # patch rectangles
    good_patches = [] # flag telling if the patch intesect with a bad region

    
    x_min, x_max = np.ravel( region.min ), np.ravel( region.max )
    x_patches    = np.floor( ( x_max - x_min ) / patchsize ).astype('int') # number of patches along each direction
    