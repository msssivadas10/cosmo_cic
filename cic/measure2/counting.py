#!/usr/bin/python3
r"""

A module for count-in-cells.

Utilities for count-in-cells measurement on a general N-dimensional point data, with support 
for patch-wise measurement for later re-samplings and parellel processing using MPI. 

Available methods:

- `countObjectsRectangularCell` for count-in-cells using rectangular cells (in any dimensions).
- `cicRectangularCell` - a complete count-in-cells workflow with rectangular cells.

"""

import os, time
import logging
import numpy as np, pandas as pd
from scipy.stats import binned_statistic_dd
from typing import Any, TypeVar 
from .utils import MeasurementError, Box, CountResult, get_parellel_process_info

Box_like = TypeVar('Box_like', Box, list[float])

def raiseError(msg :str):
    r"""
    Raise a `MeasurementError` exception, and log the message.
    """
    logging.error( msg )
    raise MeasurementError( msg )


########################################################################################################
# Base function to count objects
########################################################################################################

def _count_objects_from_df(df: pd.DataFrame,
                           expressions: list[str],
                           data_filters: list[str],
                           mask_expressions: list[str],
                           coord: list[str],
                           x_min: Any,
                           x_patches: Any,
                           patchsize: Any,
                           x_bins: Any, 
                           patch_bins: Any, 
                           extra_bins: dict) -> Any:
    r"""
    Count objects using data from a `pandas.DataFrame`.
    """

    total_objects = df.shape[0]

    # evaluate expressions: eg., magnitude corrections
    for expression in expressions:
        try: df = df.eval( expression ) 
        except Exception as _e: raiseError( f"cannot evaluate expression '{ expression }', { _e }" )

    # apply various data filters:   
    try: df = df.query( data_filters ).reset_index( drop = True )
    except Exception as _e: raiseError( f"cannot filter data, { _e }" )
    
    if not df.shape[0]: return total_objects, 0, None, None # no data to use

    used_objects = df.shape[0]

    # get mask weight: 
    if mask_expressions: mask_weight = df.eval( mask_expressions ).to_numpy().astype( 'float' )
    else: mask_weight = np.ones( df.shape[0] )

    # 
    # convert from nD physical coordinates (x1...xn) to (n+1)D patch coordinates (x'1...x'n, patch) 
    # with x'i in [0, patchsize_i] and patch in [0, patch_count]
    #  
    df[coord] = df[coord] - x_min # coordinates relative to region
    xp    = np.floor( df[coord].to_numpy() / patchsize ).astype('int')  # nD patch index
    patch = xp[:,0] # flat patch index
    for x in range( 1, len(coord) ): patch = patch * x_patches[x] + xp[:,x]
    df['patch'] = patch # np.ravel_multi_index( xp.T, x_patches ) 
    df[coord] = df[coord] - xp * patchsize # coordinates relative to patch

    # all coordinates and bins, including positions, other binned features and patch
    _all_coords, _all_bins = [*coord], [*x_bins]
    for _feature_name, _feature_bins in extra_bins.items():
        _all_coords.append(_feature_name)
        _all_bins.append(_feature_bins)
    _all_coords.append('patch')
    _all_bins.append(patch_bins)

    # chunk counts
    total_c, unmasked_c = binned_statistic_dd(df[_all_coords].to_numpy(),
                                              values    = [ np.ones(df.shape[0]), mask_weight ], # weights
                                              statistic = 'sum',
                                              bins      = _all_bins, ).statistic
            
    return total_objects, used_objects, total_c, unmasked_c

def _count_objects_asfull(path: str,
                          expressions: list[str], 
                          data_filters: list[str], 
                          mask_expressions: list[str], 
                          coord: list[str], 
                          x_min: Any, 
                          x_patches: Any, 
                          patchsize: Any, 
                          x_bins: Any, 
                          patch_bins: Any,
                          extra_bins: dict,
                          res_shape: tuple, 
                          csv_opts: dict, ) -> Any:
    r"""
    Count objects by loading full data from a file.
    """

    logging.info( "loading data from file '%s'", path )
    logging.info( "started counting objects in cell" )
    __t_init = time.time()

    df = pd.read_csv(path, **csv_opts)
    df_total, df_used, total, unmasked = _count_objects_from_df(df               = df, 
                                                                expressions      = expressions, 
                                                                data_filters     = data_filters, 
                                                                mask_expressions = mask_expressions, 
                                                                coord            = coord, 
                                                                x_min            = x_min, 
                                                                x_patches        = x_patches, 
                                                                patchsize        = patchsize, 
                                                                x_bins           = x_bins, 
                                                                patch_bins       = patch_bins, 
                                                                extra_bins       = extra_bins, )
    if total is None: total, unmasked = np.zeros( res_shape ), np.zeros( res_shape )

    logging.info( "used %d objects out of %d for counting", df_used, df_total )
    logging.info( "finished counting in %g seconds :)", time.time() - __t_init )
    return total, unmasked   

def _count_objects_aschunks(path: str,
                            expressions: list[str], 
                            data_filters: list[str], 
                            mask_expressions: list[str], 
                            coord: list[str], 
                            x_min: Any, 
                            x_patches: Any, 
                            patchsize: Any, 
                            x_bins: Any, 
                            patch_bins: Any,
                            extra_bins: dict,
                            res_shape: tuple, 
                            proc_size: int,
                            proc_rank: int, 
                            csv_opts: dict, ) -> Any:
    r"""
    Count objects by loading data as chunks from a file, if the csv options has `chunksize`
    specified, otherwise as full.
    """

    if 'chunksize' not in csv_opts: 
        total, unmasked = _count_objects_asfull(path             = path, 
                                                expressions      = expressions, 
                                                data_filters     = data_filters, 
                                                mask_expressions = mask_expressions, 
                                                coord            = coord, 
                                                x_min            = x_min, 
                                                x_patches        = x_patches, 
                                                patchsize        = patchsize, 
                                                x_bins           = x_bins, 
                                                patch_bins       = patch_bins,
                                                extra_bins       = extra_bins, 
                                                res_shape        = res_shape, 
                                                csv_opts         = csv_opts, )
        return total, unmasked

    logging.info( "loading data from file '%s' as chunks", path )
    logging.info( "started counting objects in cell" )
    __t_init = time.time()

    total, unmasked = np.zeros( res_shape ), np.zeros( res_shape )
    with pd.read_csv(path, **csv_opts ) as df_iter:
        chunk, used_objects, total_objects = 0, 0, 0
        for df in df_iter:
            # distribute chunks among different process
            this_chunk, chunk = chunk, chunk + 1
            if this_chunk % proc_size != proc_rank: continue
            df_total, df_used, tc, uc = _count_objects_from_df(df               = df, 
                                                               expressions      = expressions, 
                                                               data_filters     = data_filters, 
                                                               mask_expressions = mask_expressions, 
                                                               coord            = coord, 
                                                               x_min            = x_min, 
                                                               x_patches        = x_patches, 
                                                               patchsize        = patchsize, 
                                                               x_bins           = x_bins, 
                                                               patch_bins       = patch_bins, 
                                                               extra_bins       = extra_bins, )
            
            total_objects, used_objects = total_objects + df_total, used_objects + df_used
            if tc is not None: total, unmasked = total + tc, unmasked + uc

    logging.info( "used %d objects out of %d for counting", used_objects, total_objects )
    logging.info( "finished counting in %g seconds :)", time.time() - __t_init )
    return total, unmasked    

def countObjectsRectangularCell(path: str, 
                                region: Box_like, 
                                patchsize: float | list[float], 
                                pixsize: float | list[float],
                                use_masks: list[str] = None, 
                                data_filters: list[str] = None, 
                                expressions: list[str] = None,
                                coord: list[str] = None,
                                extra_bins: dict = None,
                                csv_opts: dict = None,   
                                use_mpi: bool = True, ) -> Any:
    r"""
    Count the number of objects in general n-D space, with recatangular cells.

    Parameters
    ----------
    path: str: 
        Path to the file containing object position and other features. Must be a csv file that can 
        be loaded with `pandas.read_csv`.
    region: Box
        Rectangular region containing the objects to count. Objects outside the region will be left.
    patchsize: float, array of float
        Size of a patch used. 
    use_masks: list of str, optional
        Names of the object mask features in the dataset. 
    data_filters: list of str, optional
        Expressions to filter the object dataset.
    expressions: list of str, optional
        Additional expressions to evaluate on the dataset.
    coord: list of str, optional
        Names of the object position coordinates in the dataset. If not specified, `x0, x1, ...` etc. 
        will be used.
    extra_bins: dict, optional
        Additional bins to use, as a dict.
    csv_opts: dict, optional
        Additional options passed to `pandas.read_csv`, such as the header.
    use_mpi: bool, optional
        Use MPI for multiprocessing (default = True)

    Returns
    -------
    total, masked: array_like
        Count of objects. `masked` is the counts, leaving the objects with no mask and `total` is the 
        count without filtering using mask. Other data filters are appied on both data.

    Notes
    -----
    - If using multiple processes using MPI, a value is returned only at rank 0. Return value will be 
      None at all other process. 

    """

    # pareller processing set-up
    comm, RANK, SIZE, _err = get_parellel_process_info(use_mpi)
    if _err: logging.warning( "module 'mpi4py' is not found, execution will be in serail mode" )

    if use_masks is None: use_masks = []
    if data_filters is None: data_filters = []
    if expressions is None: expressions = []
    if not isinstance(region, Box): region = Box.create(region)

    extra_shape = []
    if extra_bins is None: extra_bins = {}
    elif not isinstance(extra_bins, dict): raiseError("extra_bins must be a dict")
    else:
        for feature, bins in extra_bins.items():
            if np.ndim(bins) != 1: raiseError(f"bin edges of feature {feature} is 1d")
            nbins = len(bins) - 1
            if nbins < 2: raiseError(f"number of bins must be atleast 2, {feature} has only {nbins}")
            logging.info( "using extra %d bins for feature '%s'", nbins, feature )
            extra_shape.append(nbins)
    extra_shape = tuple(extra_shape)

    if not isinstance(region, Box):
        raiseError( "region should be an instance of Box or its subclass" )

    patchsize = np.ravel( patchsize )
    pixsize   = np.ravel( pixsize )
    
    if coord is None or len(coord) == 0:
        coord = [ 'x%d' % x for x in range( region.dim ) ]
    elif len( coord ) != region.dim:
        raiseError("coords list should have same size as the spatial dimension")


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
    res_shape  = ( *res_shape, *extra_shape, patch_count )   # shape of the count result array

    logging.info(f"counting with cellsize = {tuple(pixsize)}, cell arrangement shape = {res_shape[:-1]}, {patch_count} patches.")

    # extend data filters by adding boundary limits
    data_filters = [*data_filters, 
                    *[ f'{ coord[x] } >= { x_min[x] }' for x in range( region.dim ) ], 
                    *[ f'{ coord[x] } <= { x_max[x] }' for x in range( region.dim ) ], ]
    data_filters = ' & '.join( map( lambda _filt: "(%s)" % _filt, data_filters ) )
    
    # mask expressions is considered sperately from other expressions
    mask_expressions = ' & '.join( map( lambda _mask: "(%s == False)" % _mask, use_masks ) )

    logging.info( "data filter expression: '%s'", data_filters )
    if mask_expressions: logging.info( "mask expression: '%s'", mask_expressions )
    if expressions: logging.info( "other expression: '%s'", expressions )

    #
    # counting objects in parellel 
    #
    try:
        total, unmasked = _count_objects_aschunks(path             = path, 
                                                  expressions      = expressions, 
                                                  data_filters     = data_filters, 
                                                  mask_expressions = mask_expressions, 
                                                  coord            = coord, 
                                                  x_min            = x_min, 
                                                  x_patches        = x_patches, 
                                                  patchsize        = patchsize, 
                                                  x_bins           = x_bins, 
                                                  patch_bins       = patch_bins, 
                                                  extra_bins       = extra_bins,
                                                  res_shape        = res_shape, 
                                                  proc_size        = SIZE, 
                                                  proc_rank        = RANK, 
                                                  csv_opts         = csv_opts, )
    except Exception as _e: raiseError( f"counting failed with exception { _e }" )

    if comm is not None: comm.Barrier() # syncing...

    #
    # if using multiple processes, combine counts from all processes
    #
    if RANK != 0: 
        logging.info( "sending data to rank-0" )
        comm.Send( total,    dest = 0, tag = 10 ) # total count
        comm.Send( unmasked, dest = 0, tag = 11 ) # unmasked count
    else: # recieve data from other processes
        tmp = np.zeros( res_shape ) # temporary storage
        for src in range(1, SIZE):
            logging.info( "recieving data from rank-%d", src )
            
            comm.Recv( tmp, source = src, tag = 10,  ) # total count
            total = total + tmp

            comm.Recv( tmp, source = src, tag = 11,  ) # exposed count
            unmasked = unmasked + tmp

    if comm is not None: comm.Barrier() # syncing...

    # return the result at rank-0
    return (total, unmasked) if RANK == 0 else (None, None)


########################################################################################################
# Object counting for CIC
########################################################################################################

def prepareRegion(output_path: str,
                  region: Box_like, 
                  patchsize: float | list[float], 
                  pixsize: float | list[float],
                  bad_regions: list[Box_like] = None,  
                  use_mpi: bool = True,  ) -> None:
    r"""
    Prepare a region for counting. A rectangular region is divided into sub-regions of specifed size 
    and each of them again divided into cells of given size. This function will not return the results, 
    but save to a file.

    Parameters
    ----------
    output_path: str
        Path to the file to which output is written.
    region: Box
        Region to arrange cells and patches.
    patchsize:
        Size of each patches. If no patches are needed, set this value to the region size.
    pixsize:
        Size of each cells. This must be less than the patchsize. 
    bad_regions: list of Box, optional
        Regions to exclude.
    use_mpi: bool, optional
        Use MPI for multiprocessing (default = True)

    """

    if bad_regions is None: bad_regions = []

    try: region = Box.create( region )
    except Exception as _e: raiseError( f"error converting region to 'Box': {_e}" )
    dim = region.dim

    #
    # dividing the region into patches
    #

    logging.info( "checking and correcting patch sizes" )

    patchsize = np.asfarray(patchsize)
    if np.size( patchsize ) == 1:
        patchsize = np.repeat( patchsize, region.dim )
    elif np.size( patchsize ) != region.dim:
        raiseError( "patchsize should have same size as the spatial dimension" )
    if np.any( patchsize <= 0 ):
        raiseError( "one or more patchsizes are zero or negative" )

    pixsize = np.asfarray(pixsize)
    if np.size( pixsize ) == 1:
        pixsize = np.repeat( pixsize, region.dim )
    elif np.size( pixsize ) != region.dim:
        raiseError( "pixsize should have same size as the spatial dimension" )
    if np.any( pixsize <= 0 ) or np.any( pixsize > patchsize ):
        raiseError( "one or more pixsizes are zero, negative or bigger than the patchsize" )

    try:
        if bad_regions is None: bad_regions = []
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
    if patch_count == 0: raiseError( "cannot make patches with given sizes in the region" )

    badpatch_count = patch_count - sum(good_patches)
    if badpatch_count == patch_count: raiseError( "no good patches left in the region :(" )

    logging.info("created %d patches (%d bad patches)", patch_count, badpatch_count)

    if get_parellel_process_info(use_mpi).rank != 0: return # file I/O only at rank 0

    #
    # saving the results for later use
    #
    res = CountResult(region      = region, 
                      patches     = patches,
                      patch_flags = good_patches,
                      pixsize     = pixsize, 
                      patchsize   = patchsize,  )

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
                        use_masks: list[str] = None, 
                        data_filters: list[str] = None, 
                        expressions: list[str] = None,
                        coord: list[str] = None,
                        extra_bins: dict = None,
                        csv_opts: dict = None,   
                        use_mpi: bool = True,  ) -> None:
    r"""
    Calculate the cell quality (completeness) factor using random objects. Results will be saved 
    to a file.

    Parameters
    ----------
    path: str: 
        Path to the file containing random object position and other features. Must be a csv file that can 
        be loaded with `pandas.read_csv`. These random objects are used to calculate the cell area.
    patch_details_path: str
        Path to the file containing patch details, created with `prepareRegion` method.
    output_path: str, optional
        Path to the output file. If not given, use the `patch_details_file`. 
    use_masks: list of str, optional
        Names of the object mask features in the dataset. 
    data_filters: list of str, optional
        Expressions to filter the object dataset.
    expressions: list of str, optional
        Additional expressions to evaluate on the dataset.
    coord: list of str, optional
        Names of the object position coordinates in the dataset. If not specified, `x0, x1, ...` etc. 
        will be used. 
    extra_bins: dict, optional
        Additional bins to use, as a dict.
    csv_opts: dict, optional
        Additional options passed to `pandas.read_csv`, such as the header.
    use_mpi: bool, optional
        Use MPI for multiprocessing (default = True)

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
                                                  extra_bins   = extra_bins,
                                                  csv_opts     = csv_opts,  
                                                  use_mpi      = use_mpi,  )
    
    logging.info("finished counting objects!")

    if total is None: return # result is returned only at rank 0 (this is not 0!)

    # remove counts from bad patches
    # good_patches    = res.patch_flags
    # total, unmasked = total[..., good_patches], unmasked[..., good_patches]

    # goodness factor calculation 
    non_empty = ( total > 0 )
    unmasked[non_empty] = unmasked[non_empty] / total[non_empty] # re-write to save memory!
    res = res.add(value = unmasked, label = 'cell_quality')

    # add extra binning info, if given 
    if extra_bins is not None:
        for label, value in extra_bins.items(): res.extra_bins[label] = np.asfarray(value)

    logging.info("calculated cell quality factor!")

    #
    # saving the results for later use
    #

    output_path = patch_details_path if output_path is None else output_path
    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try: res.save( output_path )
    except Exception as _e:
        raiseError( f"failed to write results to file '{ output_path }'. { _e }" )

    logging.info( "results saved to '%s' :)", output_path )
    return

def estimateObjectCount(path: str, 
                        patch_details_path: str,
                        output_path: str = None,
                        include_patch_details: bool = True,
                        use_masks: list[str] = None, 
                        data_filters: list[str] = None, 
                        expressions: list[str] = None,
                        coord: list[str] = None,
                        extra_bins: dict = None,
                        csv_opts: dict = None,   
                        use_mpi: bool = True, ) -> None:
    r"""
    Count the number of objects in rectangulsr cells arranged in a given region.

    Parameters
    ----------
    path: str: 
        Path to the file containing object position and other features. Must be a csv file that can 
        be loaded with `pandas.read_csv`. These random objects are used to calculate the cell area.
    patch_details_path: str
        Path to the file containing patch details, created with `prepareRegion` method.
    output_path: str, optional
        Path to the output file. If not given, use the `patch_details_file`. 
    include_patch_details: bool, optional
        If set true, include the cell quality data to the output file (default).
    use_masks: list of str, optional
        Names of the object mask features in the dataset. 
    data_filters: list of str, optional
        Expressions to filter the object dataset.
    expressions: list of str, optional
        Additional expressions to evaluate on the dataset.
    coord: list of str, optional
        Names of the object position coordinates in the dataset. If not specified, `x0, x1, ...` etc. 
        will be used. 
    extra_bins: dict, optional
        Additional bins to use, as a dict.
    csv_opts: dict, optional
        Additional options passed to `pandas.read_csv`, such as the header.
    use_mpi: bool, optional
        Use MPI for multiprocessing (default = True)

    """

    #
    # loading patch details
    #
    logging.info( "loading patch details from file '%s'", patch_details_path )   

    try: res = CountResult.load( patch_details_path )
    except Exception as _e:
        raiseError( f"failed to load patch details file '{ patch_details_path }'. { _e }"  )
    
    # clear existing data, if not want to include patch data  
    if not include_patch_details: res.clear()
    
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
                                                  extra_bins   = extra_bins,
                                                  csv_opts     = csv_opts,  
                                                  use_mpi      = use_mpi,  )
    
    logging.info("finished counting objects!")

    if total is None: return # result is returned only at rank 0 (this is not 0!)

    # remove counts from bad patches
    # good_patches    = res.patch_flags
    # total, unmasked = total[..., good_patches], unmasked[..., good_patches]
    res.add(value = total,    label = 'total_count'   )
    res.add(value = unmasked, label = 'unmasked_count')
    if extra_bins is not None:
        for label, value in extra_bins.items(): res.extra_bins[label] = np.asfarray(value)

    #
    # saving the results for later use
    #

    output_path = patch_details_path if output_path is None else output_path
    if os.path.exists( output_path ):
        logging.warning( "file '%s' already exist, will be over-written", output_path )

    try: res.save( output_path )
    except Exception as _e:
        raiseError( f"failed to write results to file '{ output_path }'. { _e }" )

    logging.info( "results saved to '%s' :)", output_path )
    return



