#!/usr/bin/python3
r"""

`cicapp.py`: A Simple Application for N-D Count-in-cells
========================================================

`cicapp.py` can be used to measure count-in-cells in a general N-dimensional recatangular box 
region, with rectangular or cubic cells. 

Usage: ./cicapp.py [-h] [--flag FLAG] [--logs LOGS] [--no-mpi] file

Input options for counting are specified in the options file opts, in YAML or JSON format. 
Valid parameters are (NOTE: examples are in JSON format): 

================= ============================================================ ===================================                                                           
Field             Description                                                  Notes                                            
================= ============================================================ ===================================                                                                 
`id`              Optional task ID                                                           
`region`          Full box used for counting                                   e.g., `[[0.0, 0.0], [10.0, 20.0]]`                         
`badRegions`      Regions to exclude. List of objects in the form of `region`.                                                           
`patchsize`       Size of the sub-divisions                                    e.g., `[1.0]` or `[1.0, 1.0]`                      
`pixsize`         Size of the cells                                            similar to `patchsize`                         
`patchOutputFile` File to which spatial information is written (or read)                                                              
`countOutputFile` File to which counting results are written                                                              
`variables`       Mapping for $-variable replacement                           e.g., `{"x": "x_coordinate"}`                                         
`randomCatalog`   Specifications of the random catalog                                                                
`objectCatalog`   Specifications of the object catalog                                                           
================= =========================================================== ====================================                                                                 

Catalog specifications are:

================= =========================================================== ====================================
Field             Description                                                 Notes
================= =========================================================== ====================================
`path`            Filename of the catalog in CSV format                               
`coord`           Names of the coordinate features                                
`masks`           Names of the mask features                                
`filters`         Data filtering expressions                                  Can use substitute variable names  
`expressions`     Other expressions to evaluate on data, before processing    ''                                   
`csvOptions`      Other CSV read options, such as `chunksize`, `header`       Passed to `pandas.read_csv`   
`extraBins`       Extra bins used for counting, as key-value pairs            e.g., `{"z": [-1.0, 0.0, 1.0, 2.0]}`                      
================= =========================================================== ====================================

Counting process can be controlled be additional `--flags`:

- `1`: skip the cell preparation step, assuming the spatial details are already in `patchOutputFile`.
- `2`: stop after cell preparation step.

`--logs` specify the location to which runtime files are saved.

Disable parellel processing by using `--no-mpi` flag.

"""

import os
import re
import logging
import yaml
from random import choice
from string import Template
from argparse import ArgumentParser
from typing import Any 
from cic.measure2.counting import (estimateObjectCount,
                                   estimateCellQuality,
                                   prepareRegion, 
                                   get_parellel_process_info,
                                   Box_like,
                                   MeasurementError         )

########################################################################################################
# CIC workflows
########################################################################################################

def cicRectangularCell(region: Box_like = None,
                       patchsize: float | list[float] = None,
                       pixsize: float | list[float] = None, 
                       bad_regions: list[Box_like] = [], 
                       patch_details_path: str = None,
                       path_r: str = None,
                       use_masks_r: list[str] = None,
                       data_filters_r: list[str] = None,
                       expressions_r: list[str] = None,
                       coord_r: list[str] = None,
                       extra_bins_r: dict = None,
                       csv_opts_r: dict = None, 
                       path_o: str = None,
                       use_masks_o: list[str] = None,
                       data_filters_o: list[str] = None,
                       expressions_o: list[str] = None,
                       coord_o: list[str] = None,
                       extra_bins_o: dict = None,
                       csv_opts_o: dict = None, 
                       output_path: str = None, 
                       skip_cell_prepartion: bool = False, 
                       stop_after_cell_preparation: bool = False, 
                       use_mpi: bool = True,                    ) -> None:
    r"""
    Complete workflow of rectangular cell counting. Given a rectangular region and a other details, 
    this will do the following steps:
    
    1. Sub-divide the region into sub-regions (patches) and cells of given size.
    2. Calculate the quality of these cells using random points and masks.
    3. Count the numebr of objects in each of these cells.

    Results of these steps will be saved to files specied. 

    Parameters
    ----------
    region: Box_like
        Region used in the counting process. Must be a `Box` object or a list of shape (2, N), where N 
        is the number of dimensions of the counting space. This list has the format `[x_min, x_max]`, 
        where `x_min` is the minimum and `x_max` are the maximum coordinates.
    patchsize: float, list of float
        Sizes of the sub-regions. If no sub-divisions are needed, set this to the region size. 
    pixsize: float, list of float
        Sizes of the cells. Must be less than patchsize of that direction. 
    bad_regions: list of Box_like
    patch_details_path: str
        Path to file containing (or to write) the patch and cell details.
    path_r, path_o: str
        Path to the file containing random (with subscript `r`) and object (`o`) dataset. 
    use_masks_r, use_masks_o: str
        List of masks to use. Each entry must be a feture/column name in the corresponding dataset.   
    data_filters_r, data_filters_o: str
        List of data filtering expressions to apply on each dataset.
    expressions_r, expressions_o: str
        List of additional expressions to apply on each dataset. These are evaluated before counting.
    extra_bins_r, extra_bins_o: dict
        Additional bins to use, as a dict.
    coord_r, coord_o: str
        List of coordinate feature names in the datasets. This should have the same order as in the 
        region, patchsize etc specification.
    csv_opts_r, csv_opts_o: dict
        Additional csv reading options to use. 
    output_path: str
        Path to the file, to which output is written.
    skip_cell_prepartion: bool, default = False
        If set true, skip the cell preparation steps 1 and 2 and move directly to 3. Used when the cell 
        details are already available.
    stop_after_cell_preparation: bool, default = False
        If set true, stop after cell preparation steps, ignoring the step 3.
    use_mpi: bool, optional
        Use MPI for multiprocessing (default = True)
    
        
    """

    if not skip_cell_prepartion:

        # checking values
        if region is None            : raise MeasurementError("region is required for cell prepation")
        if patchsize is None         : raise MeasurementError("patchsize is required for cell prepation")
        if pixsize is None           : raise MeasurementError("pixsize is required for cell prepation")
        if path_r is None            : raise MeasurementError("path to random catalog 'path_r' is required for cell prepartion")
        if patch_details_path is None: raise MeasurementError("filename 'patch_details_path' is required for cell prepartion")

        # step 1: prepare the region and patches
        prepareRegion(patch_details_path, region, patchsize, pixsize, bad_regions, use_mpi)

        # step 2: estimate cell quality
        estimateCellQuality(path               = path_r, 
                            patch_details_path = patch_details_path, 
                            output_path        = patch_details_path, 
                            use_masks          = use_masks_r, 
                            data_filters       = data_filters_r, 
                            expressions        = expressions_r, 
                            coord              = coord_r, 
                            extra_bins         = extra_bins_r,
                            csv_opts           = csv_opts_r, 
                            use_mpi            = use_mpi,  )
        
        if stop_after_cell_preparation: return logging.info("stopping afetr cell preparation :)")
    else: logging.info("skipping cell preparation step :)")

    # checking values
    if path_o is None            : raise MeasurementError("path to object catalog 'path_o' is required for couting")
    if patch_details_path is None: raise MeasurementError("filename 'patch_details_path' is required for counting")
    if output_path is None       : raise MeasurementError("filename 'output_path' is required for counting")

    # step 3: estimate counts
    estimateObjectCount(path                  = path_o,
                        patch_details_path    = patch_details_path,
                        output_path           = output_path,
                        include_patch_details = False,
                        use_masks             = use_masks_o,
                        data_filters          = data_filters_o,
                        expressions           = expressions_o,
                        coord                 = coord_o,
                        extra_bins            = extra_bins_o,
                        csv_opts              = csv_opts_o, 
                        use_mpi               = use_mpi,  )
    return



def randstr(__length: int) -> str:
    r"""
    Generate a random string of given length.
    """
    __chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join([choice(__chars) for _ in range(__length)])

def substitute(__expressions: list[str], __vartable: dict[str, str]) -> list[str]:
    r"""
    Do a varaiable substitution on the given expressions using a variable table. 
    """
    if __expressions is None or __vartable is None: return __expressions
    __expressions = list(__expressions)
    for i, __str in enumerate(__expressions):
        __vars = re.findall( r'\$(\w+)', __str )
        for __var in __vars:
            if __var in __vartable.keys(): continue
            raise ValueError(f"no sustitution available for variable '{__var}'")
        __expressions[i] = Template(__str).substitute(__vartable)
    return __expressions

def rget(__name: str, __dict: dict, __default: Any = None) -> Any:
    r"""
    Recursively get a node value from a dict tree.
    """
    if not isinstance(__name, str ): raise TypeError("argument __name must be a str" )
    if __dict is None: return __default
    if not isinstance(__dict, dict): raise TypeError(f"argument __dict must be a dict")
    __name  = __name.split('.', maxsplit = 1)
    if len(__name) == 1: __name, __ext = __name[0], None 
    else: __name, __ext = __name
    if __ext is None: return __dict.get(__name, __default)
    __next_dict = __dict.get(__name)
    return rget(__ext, __next_dict, __default)

def rcopy(__src: dict, __dest: dict) -> None:
    r"""
    Recursively copy values from source dict tree to destination tree.
    """
    if not isinstance(__src , dict): raise TypeError("__src must be a dict" )
    if not isinstance(__dest, dict): raise TypeError("__dest must be a dict")
    for __key, __src_value in __src.items():
        if __key not in __dest: continue
        __dest_value = __dest[__key]
        if isinstance(__dest_value, dict): rcopy(__src_value, __dest_value)
        else: __dest[__key] = __src_value
    return

def load_options(file: str, opts: dict) -> None:
    r"""
    Load options in a file to a dict tree.
    """
    if not isinstance(file, str ): raise TypeError("argument file must be a str" )
    if not isinstance(opts, dict): raise TypeError("argument opts must be a dict")
    with open(file, 'r') as fp: return rcopy(yaml.safe_load(fp), opts)

def create_argument_parser() -> ArgumentParser:
    ap = ArgumentParser(prog         = "cicapp.py", 
                        description  = "A simple app to for count-in-cells in an n-dimensional space, with rectangular cells.")
    ap.add_argument('file'    , help = 'path to the input options file'    , type   = str         ,               )
    ap.add_argument('--flag'  , help = 'flags to control the execution'    , type   = int         , default  = 0  )
    ap.add_argument('--logs'  , help = 'path to create the log files'      , type   = str         , default  = '.')
    ap.add_argument('--no-mpi', help = 'disable multiprocessing using mpi' , action = 'store_true',               )
    return ap

def configure_logging(file: str, mode: str = 'a') -> None:
    if not isinstance(file, str ): raise TypeError("argument file must be a str" )
    if not os.path.exists(file): open(file, 'w').close() # NOTE: hack!
    logging.basicConfig(level    = logging.INFO, 
                        format   = "%(asctime)s [%(levelname)s] %(message)s", 
                        handlers = [logging.FileHandler(filename = file, mode = mode), logging.StreamHandler()])
    return

def __count_in_cells(opts: dict, flag: int, no_mpi: bool) -> None:
    region             = rget('region'                   , opts)
    patchsize          = rget('patchsize'                , opts)
    pixsize            = rget('pixsize'                  , opts)
    bad_regions        = rget('badRegions'               , opts)
    patch_details_path = rget('patchOutputFile'          , opts)
    path_r             = rget('randomCatalog.path'       , opts)
    use_masks_r        = rget('randomCatalog.masks'      , opts)
    data_filters_r     = rget('randomCatalog.filters'    , opts)
    expressions_r      = rget('randomCatalog.expressions', opts)
    extra_bins_r       = rget('randomCatalog.extraBins'  , opts)
    coord_r            = rget('randomCatalog.coord'      , opts)
    csv_opts_r         = rget('randomCatalog.csvOptions' , opts)
    path_o             = rget('objectCatalog.path'       , opts)
    use_masks_o        = rget('objectCatalog.masks'      , opts)
    data_filters_o     = rget('objectCatalog.filters'    , opts)
    expressions_o      = rget('objectCatalog.expressions', opts)
    extra_bins_o       = rget('objectCatalog.extraBins'  , opts)
    coord_o            = rget('objectCatalog.coord'      , opts)
    csv_opts_o         = rget('objectCatalog.csvOptions' , opts)
    output_path        = rget('countOutputFile'          , opts)
    variable_mapping   = rget('variables'                , opts)
    data_filters_r     = substitute(data_filters_r, variable_mapping)
    data_filters_o     = substitute(data_filters_o, variable_mapping)
    expressions_r      = substitute(expressions_r , variable_mapping)
    expressions_o      = substitute(expressions_o , variable_mapping)
    cicRectangularCell(region             = region,
                        patchsize          = patchsize,
                        pixsize            = pixsize,
                        bad_regions        = bad_regions,
                        patch_details_path = patch_details_path,
                        path_r             = path_r,
                        use_masks_r        = use_masks_r,
                        data_filters_r     = data_filters_r,
                        expressions_r      = expressions_r,
                        extra_bins_r       = extra_bins_r,
                        coord_r            = coord_r,
                        csv_opts_r         = csv_opts_r,
                        path_o             = path_o,
                        use_masks_o        = use_masks_o,
                        data_filters_o     = data_filters_o,
                        expressions_o      = expressions_o,
                        extra_bins_o       = extra_bins_o,
                        coord_o            = coord_o,
                        csv_opts_o         = csv_opts_o,
                        output_path        = output_path,
                        skip_cell_prepartion        = flag == 1,
                        stop_after_cell_preparation = flag == 2, 
                        use_mpi                     = not no_mpi, )        
    return 

def main() -> None:
    args = create_argument_parser().parse_args() # command line arguments namespace

    # loading options:
    opts = dict(region          = None,
                patchsize       = None,
                pixsize         = None,
                badRegions      = None,
                patchOutputFile = None,
                randomCatalog   = dict(path        = None, 
                                       masks       = None, 
                                       filters     = None, 
                                       expressions = None, 
                                       extraBins   = None, 
                                       coord       = None, 
                                       csvOptions  = None, ),
                objectCatalog   = dict(path        = None, 
                                       masks       = None, 
                                       filters     = None, 
                                       expressions = None, 
                                       extraBins   = None, 
                                       coord       = None, 
                                       csvOptions  = None, ),
                countOutputFile = None,
                variables       = None,
                id              = None, )
    load_options(args.file, opts)

    # save loaded options to a file:
    __id = rget('id', opts)
    if __id is None: __id = randstr(16)
    task_rootdir = os.path.join(os.path.abspath(args.logs), __id)
    if not os.path.exists(task_rootdir): os.makedirs(task_rootdir)
    with open(os.path.join(task_rootdir, 'used_options'), 'w') as fp: yaml.safe_dump(opts, fp)

    # configure logging:
    rank    = get_parellel_process_info(not args.no_mpi).rank
    logpath = os.path.join(task_rootdir, 'logs')
    if not os.path.exists(logpath): os.makedirs(logpath)
    logpath = os.path.join(logpath, "%d.log" % rank )
    configure_logging(logpath, mode = 'w')

    # counting: 
    try: 
        logging.info("starting counting mission '%s'" % __id)
        __count_in_cells(opts, args.flag, args.no_mpi)
        logging.info("counting mission '%s' completed successfully! :)" % __id)
    except Exception as _e: logging.error("counting mission '%s' failed with exception %s :(" % (__id, _e))
    return


if __name__ == '__main__': main()

