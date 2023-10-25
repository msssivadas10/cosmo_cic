#!/usr/bin/python3

import os, time
import logging
import numpy as np, pandas as pd
from scipy.stats import binned_statistic_dd
from cic.measure.utils import Rectangle, CountResult
from typing import Any 

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


class CountingError( Exception ):
    r"""
    Base class of exceptions raised during failure in counting processes.    
    """


########################################################################################################
# Base function to count objects
########################################################################################################

def countObjects_rectangularCell(path: str, 
                                 region: Rectangle, 
                                 patchsize_x: float, 
                                 patchsize_y: float, 
                                 pixsize: float,
                                 use_masks: list[str] = [], 
                                 data_filters: list[str] = [], 
                                 expressions: list[str] = [],
                                 x_coord: str = 'x',
                                 y_coord: str = 'y',
                                 **csv_opts: Any             ) -> Any:
    r"""
    Count the number of objects in 2D recatangular cells.

    Parameters
    ----------
    path: str
        Full path to the file containing the object data.
    region: Rectangle
        Region where the counting is done.
    patchsize_x, patchsize_y: float
        Size of a sub-region, if the counting is done on sub-divisions of the full 
        region.
    pixsize: float
        Size of a cell. The patchsizes must be an *integer multiple* of the pixsize 
        for the process to take place without any errors.
    use_masks: list[str]
        Names of the masks to use. A mask is a flag telling the that the object is in 
        a region that cannot be used. Each mask entry must be a string indcating the 
        column name in the object data.
    data_filters: list[str]
        Various filtering conditions to apply on the data. These expressions must be 
        in a format that can be evaluated by `pandas.DataFrame.query`.
    expressions: list[str]
        Various expressions to be evaluated on the data before the calculations. These 
        expressions must be in a format that can be evaluated by `pandas.DataFrame.query`.
    x_coord, y_coord: str
        Name of the X, Y coordinates of the objects in the data.
    csv_opts:
        Keyword arguments that are directly passed to the `pandas.read_csv`.

    Returns
    -------
    total: array_like
        Total measured counts in cells. This will be a 3D array of floats, where the first 
        two dimensions indicate the cell positions and the third dimension indicate the 
        patches. A value is returned only at the process of rank-0.
    unmasked: array_like
        Count of objects do not having any of the mask flags set. Have the same format as 
        the total counts. A value is returned only at the process of rank-0.

    Notes
    -----
        - A keyword argument `chunksize=1` must be given if the data is loaded all at once.

    """

    # setting up MPI
    comm, RANK, SIZE = getCommunicatorInfo()

    # number of patches along each direction:
    rx_min, ry_min, rx_max, ry_max = region.x_min, region.y_min, region.x_max, region.y_max
    x_patches = int( ( rx_max - rx_min ) / patchsize_x )
    y_patches = int( ( ry_max - ry_min ) / patchsize_y )
    n_patches = x_patches * y_patches # total number of patches

    # bin edges
    x_cells, y_cells = int( patchsize_x / pixsize ), int( patchsize_y / pixsize ) # number of cells
    x_bins           = np.linspace( 0., patchsize_x, x_cells + 1 ) # cells x edges
    y_bins           = np.linspace( 0., patchsize_y, y_cells + 1 ) # cells y edges
    patch_bins       = np.arange( n_patches + 1 ) - 0.5  # patch bins are centered at its index
    img_shape        = ( x_cells, y_cells, n_patches ) 

    logging.info("counting with cellsize = %f, cell arrangement shape = (%d, %d), %d patches.", pixsize, x_cells, y_cells, n_patches)
        
    # extend data filters by adding boundary limits
    data_filters = [*data_filters, 
                    f'{ x_coord } >= { rx_min }', 
                    f'{ y_coord } >= { ry_min }', 
                    f'{ x_coord } <= { rx_max }',
                    f'{ y_coord } <= { ry_max }', ]
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
        _msg = f"failed to load data from '{ path }'. { _e }"
        logging.error( _msg )
        raise CountingError( _msg )

    total, unmasked = np.zeros( img_shape ), np.zeros( img_shape )
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
                    _msg = f"cannot evaluate expression '{ expression }', { _e }"
                    logging.error( _msg )
                    raise CountingError( _msg )

            # apply various data filters:   
            try:
                df = df.query( data_filters ).reset_index( drop = True )
            except Exception as _e:
                _msg = f"cannot filter data, { _e }"
                logging.error( _msg )
                raise CountingError( _msg )
            
            if not df.shape[0]:
                continue # no data to use
            used_objects += df.shape[0]

            # get mask weight: 
            if mask_expressions:
                mask_weight = df.eval( mask_expressions ).to_numpy().astype( 'float' )
            else:
                mask_weight = np.ones( df.shape[0] )

            # 
            # convert from 2D physical coordinates (x, y) to 3D patch coordinates (x', y', patch) with 
            # x' in [0, patchsize_x], y' in [0, patchsize_y] and patch in [0, n_patches) 
            #  
            df[x_coord] = df[x_coord] - rx_min # coordinates relative to region
            df[y_coord] = df[y_coord] - ry_min

            xp, yp      = np.floor( df[x_coord] / patchsize_x ), np.floor( df[y_coord] / patchsize_y )
            df['patch'] = y_patches * xp + yp # patch index

            # coordinates relative to patch
            df[x_coord] = df[x_coord] - xp * patchsize_x 
            df[y_coord] = df[y_coord] - yp * patchsize_y 

            # chunk counts
            total_c, unmasked_c = binned_statistic_dd(df[[x_coord, y_coord, 'patch']].to_numpy(),
                                                      values    = [ np.ones(df.shape[0]), mask_weight ], # weights
                                                      statistic = 'sum',
                                                      bins      = [ x_bins, y_bins, patch_bins ], ).statistic
            
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
        tmp = np.zeros( img_shape ) # temporary storage
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
    if RANK != 0:
        return None, None
    return total, unmasked


########################################################################################################
# Object counting for CIC
########################################################################################################
            
def prepareRegion(path: str, 
                  output_path: str,
                  region: Rectangle, 
                  patchsize_x: float, 
                  patchsize_y: float, 
                  pixsize: float,
                  bad_regions: list[Rectangle] = [],
                  use_masks: list[str] = [], 
                  data_filters: list[str] = [], 
                  expressions: list[str] = [],
                  x_coord: str = 'x',
                  y_coord: str = 'y',
                  **csv_opts: Any                  ) -> None:
    r"""
    Prepare the region where counting take place. This includes dividing the region into patches, 
    and generating a completeness image. Results will be saved to the path given as JSON file. 
    This result can be loaded with `CountResult.load`

    Parameters
    ----------
    path: str
        Full path to the file containing the object data.
    output_path: str
        Full path to the output file.
    region: Rectangle
        Region where the counting is done.
    patchsize_x, patchsize_y: float
        Size of a sub-region, if the counting is done on sub-divisions of the full 
        region.
    pixsize: float
        Size of a cell. The patchsizes must be an *integer multiple* of the pixsize 
        for the process to take place without any errors.
    bad_regions: list[Rectangle]
        A list of regions that are excluded from the counting process. Any sub-region 
        overlapping with these are removed.
    use_masks: list[str]
        Names of the masks to use. A mask is a flag telling the that the object is in 
        a region that cannot be used. Each mask entry must be a string indcating the 
        column name in the object data.
    data_filters: list[str]
        Various filtering conditions to apply on the data. These expressions must be 
        in a format that can be evaluated by `pandas.DataFrame.query`.
    expressions: list[str]
        Various expressions to be evaluated on the data before the calculations. These 
        expressions must be in a format that can be evaluated by `pandas.DataFrame.query`.
    x_coord, y_coord: str
        Name of the X, Y coordinates of the objects in the data.
    csv_opts:
        Keyword arguments that are directly passed to the `pandas.read_csv`.

    Notes
    -----
        - A keyword argument `chunksize=1` must be given if the data is loaded all at once.

    """

    #
    # dividing the region into patches
    #

    logging.info( "checking and correcting patch sizes" )

    if patchsize_x <= 0 or patchsize_y <= 0:
        _msg = f"patch sizes must be positive (x_size = {patchsize_x}, y_size = {patchsize_y})"
        logging.error( _msg )
        raise CountingError( _msg )
    if pixsize <= 0 or pixsize > min( patchsize_x, patchsize_y ):
        _msg = f"pixsize (= {pixsize}) must be less than the patch sizes, min({patchsize_x}, {patchsize_y})"
        logging.error( _msg )
        raise CountingError( _msg )
    
    try:
        region = Rectangle.create( region )
    except Exception as _e:
        _msg = "'region' is not a valid rectangle-like object"
        logging.error( _msg )
        raise CountingError( _msg )

    try:
        bad_regions = [] if bad_regions is None else [ Rectangle.create( bad_region ) for bad_region in bad_regions ]
    except Exception as _e:
        _msg = "some of the bad regions is not a valid rectangle-like object"
        logging.error( _msg )
        raise CountingError( _msg )

    # correcting the patchsizes to be exacly equal to a nearest multiple of pixsize
    # this make sure that the cell bins are correctly calculated in the base function
    x_cells, y_cells = int( patchsize_x / pixsize ), int( patchsize_y / pixsize )
    patchsize_x      = x_cells * pixsize
    patchsize_y      = y_cells * pixsize     

    # diving region into patches not overlapping with a bad region
    patches      = [] # patch rectangles
    good_patches = [] # flag telling if the patch intesect with a bad region
    
    x_min = region.x_min
    while 1:

        x_max = x_min + patchsize_x
        if x_max > region.x_max:
            break

        y_min = region.y_min
        while 1:

            y_max = y_min + patchsize_y
            if y_max > region.y_max:
                break

            flag  = False
            patch = Rectangle( x_min, y_min, x_max, y_max )
            for bad_region in bad_regions:
                flag = bad_region.intersect( patch )
                if flag:
                    break

            patches.append( patch )
            good_patches.append( not flag )

            y_min = y_max
        # y-direction loop ends here

        x_min = x_max
    # x-direction loop ends here

    n_patches = len( patches )
    if not n_patches: # no patches in this region
        _msg = "cannot make patches with given sizes in the region"
        logging.error( _msg )
        raise CountingError( _msg )
    
    n_bad_patches = n_patches - sum( good_patches )
    if n_patches == n_bad_patches: # all the patches are bad
        _msg = "no good patches left in the region :("
        logging.error( _msg )
        raise CountingError( _msg )
    
    logging.info("created %d patches (%d bad patches)", n_patches, n_bad_patches)

    #
    # estimating the cell goodness factor (fraction of non-masked objects)
    #
    logging.info("started counting objects")

    total, unmasked = countObjects_rectangularCell(path         = path, 
                                                   region       = region,
                                                   patchsize_x  = patchsize_x, 
                                                   patchsize_y  = patchsize_y, 
                                                   pixsize      = pixsize, 
                                                   use_masks    = use_masks, 
                                                   data_filters = data_filters, 
                                                   expressions  = expressions, 
                                                   x_coord      = x_coord, 
                                                   y_coord      = y_coord, 
                                                   **csv_opts                 )
    
    logging.info("finished counting objects!")

    
    if MPI.COMM_WORLD.rank != 0: # result is returned only at rank 0
        return 

    # remove counts from bad patches
    total, unmasked = total[..., good_patches], unmasked[..., good_patches]

    # goodness factor calculation 
    non_empty = ( total > 0 )
    unmasked[ non_empty ] = unmasked[ non_empty ] / total[ non_empty ] # re-write to save memory!

    logging.info("calculated cell goodness factor!")

    #
    # saving the results for later use
    #
    res = CountResult(region      = region, 
                      patches     = patches,
                      patch_flags = good_patches,
                      pixsize     = pixsize, 
                      patchsize_x = patchsize_x,
                      patchsize_y = patchsize_y )
    res = res.add(value = unmasked, label = 'goodness')

    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try:
        res.save( output_path )
    except Exception as _e:
        _msg = f"failed to write results to file '{ output_path }'. { _e }"
        logging.error( _msg )
        raise CountingError( _msg )

    logging.info( "results saved to '%s' :)", output_path )
    
    return
        

def countObjects(path: str, 
                 patch_details_path: str,
                 output_path: str,
                 include_patch_details: bool = True,
                 use_masks: list[str] = [], 
                 data_filters: list[str] = [], 
                 expressions: list[str] = [],
                 x_coord: str = 'x',
                 y_coord: str = 'y',
                 **csv_opts: Any                   ) -> None:
    r"""
    Count objects in cells in a region. The region must be processed in advance using 
    `prepareRegion` and saved, so that the results can be loaded.

    Parameters
    ----------
    path: str
        Full path to the file containing the object data.
    patch_details_path: str
        Full path to the patch details file, as generated by `prepareRegion`.
    output_path: str
        Full path to the output file.
    include_patch_details: bool
        If set true (default), include the patch count data (i.e., unmasked area fraction) 
        to the output file.
    use_masks: list[str]
        Names of the masks to use. A mask is a flag telling the that the object is in 
        a region that cannot be used. Each mask entry must be a string indcating the 
        column name in the object data.
    data_filters: list[str]
        Various filtering conditions to apply on the data. These expressions must be 
        in a format that can be evaluated by `pandas.DataFrame.query`.
    expressions: list[str]
        Various expressions to be evaluated on the data before the calculations. These 
        expressions must be in a format that can be evaluated by `pandas.DataFrame.query`.
    x_coord, y_coord: str
        Name of the X, Y coordinates of the objects in the data.
    csv_opts:
        Keyword arguments that are directly passed to the `pandas.read_csv`.

    Notes
    -----
        - A keyword argument `chunksize=1` must be given if the data is loaded all at once.

    """

    #
    # loading patch details
    #
    logging.info( "loading patch details from file '%s'", patch_details_path )   

    try:
        patch_details = CountResult.load( patch_details_path )
    except Exception as _e:
        _msg = f"failed to load patch details file '{ patch_details_path }'. { _e }" 
        logging.error( _msg )
        raise CountingError( _msg )
    
    # clear existing data, if not want to include patch data  
    if not include_patch_details:
        patch_details.data.clear()
    
    logging.info( "loaded patch details" )

    #
    # counting objects in cells 
    #

    logging.info("started counting objects")

    total, unmasked = countObjects_rectangularCell(path         = path, 
                                                   region       = patch_details.region,
                                                   patchsize_x  = patch_details.patchsize_x, 
                                                   patchsize_y  = patch_details.patchsize_y, 
                                                   pixsize      = patch_details.pixsize, 
                                                   use_masks    = use_masks, 
                                                   data_filters = data_filters, 
                                                   expressions  = expressions, 
                                                   x_coord      = x_coord, 
                                                   y_coord      = y_coord, 
                                                   **csv_opts                              )
    
    logging.info("finished counting objects!")

    if MPI.COMM_WORLD.rank != 0: # result is returned only at rank 0
        return 

    # remove counts from bad patches
    good_patches    = patch_details.patch_flags
    total, unmasked = total[..., good_patches], unmasked[..., good_patches]
    patch_details.add(value = total,    label = 'total_count'   )
    patch_details.add(value = unmasked, label = 'unmasked_count')

    #
    # saving the results for later use
    #

    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try:
        patch_details.save( output_path )
    except Exception as _e:
        _msg = f"failed to write results to file '{ output_path }'. { _e }"
        logging.error( _msg )
        raise CountingError( _msg )

    logging.info( "results saved to '%s' :)", output_path )
    
    return









