#!/usr/bin/python3

import os, time
import logging
import numpy as np, pandas as pd
from scipy.stats import binned_statistic_dd
from typing import Any 
from .utils import MeasurementError, Box, CountResult

DISABLE_MPI = False # switch indicating if to disable parallel processing with MPI

try:
    from mpi4py import MPI    
except ModuleNotFoundError:
    DISABLE_MPI = True # disable using MPI
    logging.warning( "module 'mpi4py' is not found, disabling parallel processing" )


def getCommunicatorInfo() -> tuple:
    r"""
    Return the MPI communication details: the communicator object, rank and size 
    of the process. 
    """

    if DISABLE_MPI:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.rank, comm.size

def setBarrier(comm: Any) -> Any:
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

    # NOTE: chunksize is a required csv options, since loading data as iterable not work without it. 
    # a large number is used to make sure that the data will read as an iterable of minimum size 1
    if 'chunksize' not in csv_opts: 
        csv_opts['chunksize'] = 100000

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

            xp    = np.floor( df[coord] / patchsize ).astype('int')  # nD patch index
            patch = xp[:,0] # flat patch index
            for x in range( 1, region.dim ):
                patch = patch * x_patches[x] * xp[:,x]
            df['patch'] = patch # np.ravel_multi_index( xp.T, x_patches ) 

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

def prepareRegion(output_path: str,
                  region: Box, 
                  patchsize: float | list[float], 
                  pixsize: float | list[float],
                  bad_regions: list[Box] = [],  ) -> None:
    r"""
    Prepare a region for counting. 
    """

    try:
        region = Box.create( region )
    except Exception as _e:
        raiseError( f"error converting region to 'Box': {_e}" )
    dim = region.dim

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
    
    for bad_region in bad_regions:
        if not bad_region.samedim( region ):
            raiseError( "dimension mismatch between one bad region and region" )

    # correcting the patchsizes to be exacly equal to a nearest multiple of pixsize
    # this make sure that the cell bins are correctly calculated in the base function
    x_cells   = np.floor( patchsize / pixsize ).astype('int')
    patchsize = x_cells * pixsize

    # diving region into patches not overlapping with a bad region
    patches      = [] # patch rectangles
    good_patches = [] # flag telling if the patch intesect with a bad region

    x_min, x_max = np.ravel( region.min ), np.ravel( region.max )
    patch_coords = x_min + 0.
    stop         = False
    while 1:
        patch = Box(patch_coords + 0., patch_coords + patchsize)
        patches.append(patch)
        good_patches.append(True)
        for bad_region in bad_regions:
            if patch.intersect( bad_region ):
                good_patches[-1] = False
                break

        patch_coords[dim-1] += patchsize[dim-1]
        for x in reversed( range( dim ) ):
            if patch_coords[x] >= x_max[x]:
                patch_coords[x]    = x_min[x]
                patch_coords[x-1] += patchsize[x-1]
                if x == 0:
                    stop = True
        if stop:
            break

    patch_count = len( patches )
    if patch_count == 0: # no patches in this region
        raiseError( "cannot make patches with given sizes in the region" )

    badpatch_count = patch_count - sum(good_patches)
    if badpatch_count == patch_count: # all the patches are bad
        raiseError( "no good patches left in the region :(" )

    logging.info("created %d patches (%d bad patches)", patch_count, badpatch_count)


    if MPI.COMM_WORLD.rank != 0: return # file I/O only at rank 0

    #
    # saving the results for later use
    #
    res = CountResult(region      = region, 
                      patches     = patches,
                      patch_flags = good_patches,
                      pixsize     = pixsize, 
                      patchsize   = patchsize   )

    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try:
        res.save( output_path )
    except Exception as _e:
        raiseError( f"failed to write results to file '{ output_path }'. { _e }" )

    logging.info( "results saved to '%s' :)", output_path )
    
    return

def estimateCellQuality(path: str, 
                        patch_details_path: str,
                        output_path: str = None,
                        use_masks: list[str] = [], 
                        data_filters: list[str] = [], 
                        expressions: list[str] = [],
                        coord: list[str] = None,
                        **csv_opts: Any        ) -> None:
    r"""
    Calculate the cell quality (completeness) factor using random objects.
    """

    #
    # loading patch details
    #
    logging.info( "loading patch details from file '%s'", patch_details_path )   

    try:
        res = CountResult.load( patch_details_path )
    except Exception as _e:
        raiseError( f"failed to load patch details file '{ patch_details_path }'. { _e }"  )

    #
    # estimating the cell quality factor (fraction of non-masked objects)
    #
    logging.info("started counting objects")

    total, unmasked = countObjectsRectangularCell(path         = path, 
                                                  region       = res.region,
                                                  patchsize    = res.patchsize, 
                                                  pixsize      = res.pixsize, 
                                                  use_masks    = use_masks, 
                                                  data_filters = data_filters, 
                                                  expressions  = expressions, 
                                                  coord        = coord, 
                                                  **csv_opts,  )
    
    logging.info("finished counting objects!")

    if total is None: return # result is returned only at rank 0 (this is not 0!)

    # remove counts from bad patches
    # good_patches    = res.patch_flags
    # total, unmasked = total[..., good_patches], unmasked[..., good_patches]

    # goodness factor calculation 
    non_empty = ( total > 0 )
    unmasked[non_empty] = unmasked[non_empty] / total[non_empty] # re-write to save memory!
    res = res.add(value = unmasked, label = 'cell_quality')

    logging.info("calculated cell quality factor!")

    #
    # saving the results for later use
    #

    output_path = patch_details_path if output_path is None else output_path
    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try:
        res.save( output_path )
    except Exception as _e:
        raiseError( f"failed to write results to file '{ output_path }'. { _e }" )

    logging.info( "results saved to '%s' :)", output_path )
    return

def estimateObjectCount(path: str, 
                        patch_details_path: str,
                        output_path: str = None,
                        include_patch_details: bool = True,
                        use_masks: list[str] = [], 
                        data_filters: list[str] = [], 
                        expressions: list[str] = [],
                        coord: list[str] = None,
                        **csv_opts: Any        ) -> None:
    r"""
    Count objects in cells in a region.
    """

    #
    # loading patch details
    #
    logging.info( "loading patch details from file '%s'", patch_details_path )   

    try:
        res = CountResult.load( patch_details_path )
    except Exception as _e:
        raiseError( f"failed to load patch details file '{ patch_details_path }'. { _e }"  )
    
    # clear existing data, if not want to include patch data  
    if not include_patch_details:
        res.data.clear()
    
    logging.info( "loaded patch details" )

    #
    # counting objects in cells 
    #

    logging.info("started counting objects")

    total, unmasked = countObjectsRectangularCell(path         = path, 
                                                  region       = res.region,
                                                  patchsize    = res.patchsize, 
                                                  pixsize      = res.pixsize, 
                                                  use_masks    = use_masks, 
                                                  data_filters = data_filters, 
                                                  expressions  = expressions, 
                                                  coord        = coord, 
                                                  **csv_opts,  )
    
    logging.info("finished counting objects!")

    if total is None: return # result is returned only at rank 0 (this is not 0!)

    # remove counts from bad patches
    # good_patches    = res.patch_flags
    # total, unmasked = total[..., good_patches], unmasked[..., good_patches]

    res.add(value = total,    label = 'total_count'   )
    res.add(value = unmasked, label = 'unmasked_count')

    #
    # saving the results for later use
    #

    output_path = patch_details_path if output_path is None else output_path
    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try:
        res.save( output_path )
    except Exception as _e:
        raiseError( f"failed to write results to file '{ output_path }'. { _e }" )

    logging.info( "results saved to '%s' :)", output_path )
    
    return


