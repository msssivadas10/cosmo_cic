#!/usr/bin/python3
r"""

`cicapp.py`: A Simple Application for N-D Count-in-cells
========================================================

`cicapp.py` can be used to measure count-in-cells in a general N-dimensional recatangular box 
region, with rectangular or cubic cells. 

Usage: ./cicapp [-h] [--opts OPTS] [--flag FLAG] [--logs LOGS] [--use-mpi USE_MPI]

Input options for counting are specified in the options file OPTS, in YAML or JSON format. 
Valid parameters are: 

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
`variables`       Mapping for $-variable replacement                           e.g., `{'x': 'x_coordinate'}`                                         
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
================= =========================================================== ====================================

Counting process can be controlled be FLAG:

- `1`: skip the cell preparation step, assuming the spatial details are already in `patchOutputFile`.
- `2`: stop after cell preparation step.

LOGS specify the location to which runtime files are saved.

Disable parellel processing by setting USE_MPI to 0.

"""

import os, re
from random import choice
from string import Template
from cic.misc.appbuilder import Application
from cic.measure2.counting import cicRectangularCell, _get_parellel_process_info


def random_string(__len: int) -> str:
    r"""
    Generate a random string of given length.
    """
    __chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join([choice(__chars) for _ in range(__len)])

def substitute(__expressions: list[str], __vartable: dict[str, str]) -> list[str]:
    r"""
    Do a variable substitution on a list of expressions using a variable table.
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

class App(Application):
    r"""
    A simple app to for count-in-cells in an n-dimensional space, with rectangular cells.
    """
    
    def __init__(self) -> None:
        super().__init__(name = 'cic.app', description = self.__doc__)

    def create_argslist(self) -> None:
        self.add_argument('--opts'   , help = 'path to the input options file', type = str, required = 0  )
        self.add_argument('--flag'   , help = 'flags to control the execution', type = int, default  = 0  )
        self.add_argument('--logs'   , help = 'path to create the log files'  , type = str, default  = '.')
        self.add_argument('--use-mpi', help = 'use multiprocessing using mpi' , type = int, default  = 1  )

    def create_options(self) -> None:
        self.add_option('id'                        , help = 'id for the counting job'                  , cls = 'str',                                                    )
        self.add_option("region"                    , help = "region to do counting"                    , cls = 'num',                  ndim = 2, shape = (2, None)       )
        self.add_option("badRegions"                , help = "list of regions to exclude"               , cls = 'num', optional = True, ndim = 3, shape = (None, 2, None) )
        self.add_option("patchsize"                 , help = "size of a patch"                          , cls = 'num',                  ndim = 1,                         )
        self.add_option("pixsize"                   , help = "size of a cell"                           , cls = 'num',                  ndim = 1,                         )
        self.add_option("patchOutputFile"           , help = "path to the file containing patch details", cls = 'str',                  ndim = 0,                         )
        self.add_option("countOutputFile"           , help = "path to the file containing count details", cls = 'str',                  ndim = 0,                         )
        self.add_option("variables"                 , help = "variable mapping"                         , cls = 'any', optional = True,                                   )
        self.add_option("randomCatalog"             , help = "specifications for random object catalog" , cls = 'blk',                                                    )
        self.add_option("objectCatalog"             , help = "specifications for object catalog"        , cls = 'blk',                                                    )
        self.add_option("randomCatalog.path"        , help = "path to the random catalog file"          , cls = 'str',                            isfile = True           )
        self.add_option("randomCatalog.coord"       , help = "names of the coordinate features"         , cls = 'str',                  ndim = 1,                         )
        self.add_option("randomCatalog.masks"       , help = "names for mask features"                  , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("randomCatalog.filters"     , help = "data filtering expressions"               , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("randomCatalog.expressions" , help = "other expressions to evaluate on data"    , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("randomCatalog.extraBins"   , help = "Additional bins to use"                   , cls = 'any', optional = True,                                   )
        self.add_option("randomCatalog.csvOptions"  , help = "additional csv file options"              , cls = 'any', optional = True,                                   )
        self.add_option("objectCatalog.path"        , help = "path to the random catalog file"          , cls = 'str',                            isfile = True           )
        self.add_option("objectCatalog.coord"       , help = "names of the coordinate features"         , cls = 'str',                  ndim = 1,                         )
        self.add_option("objectCatalog.masks"       , help = "names for mask features"                  , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("objectCatalog.filters"     , help = "data filtering expressions"               , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("objectCatalog.expressions" , help = "other expressions to evaluate on data"    , cls = 'str', optional = True, ndim = 1                          )
        self.add_option("objectCatalog.extraBins"   , help = "Additional bins to use"                   , cls = 'any', optional = True,                                   )
        self.add_option("objectCatalog.csvOptions"  , help = "additional csv file options"              , cls = 'any', optional = True,                                   )
        return
    
    def run(self) -> None:
        self.parse_args() 
        if not self.args.opts: self.exit("no input files are given, exiting :)")

        # loading options:
        self.load_options(self.args.opts)     

        # save loaded options to a file:
        __id = self.options['id']
        if __id is None: __id = random_string(16)
        task_rootdir = os.path.join(os.path.abspath(self.args.logs), __id)
        if not os.path.exists(task_rootdir): os.makedirs(task_rootdir)
        with open(os.path.join(task_rootdir, 'used_options'), 'w') as fp: self.options.print(buffer = fp)

        # configure logging:
        rank    = _get_parellel_process_info(self.args.use_mpi).rank
        logpath = os.path.join(task_rootdir, 'logs')
        if not os.path.exists(logpath): os.makedirs(logpath)
        logpath = os.path.join(logpath, "%d.log" % rank )
        if not os.path.exists(logpath): open(logpath, 'a').close() # NOTE: hack!
        self.basic_logconfig(file = logpath, mode = 'w')

        # counting: 
        try: 
            self.log_info("starting counting mission '%s'" % __id)
            self._do_counting()
            self.log_info("counting mission '%s' completed successfully! :)" % __id)
        except Exception as _e: self.log_info("counting mission '%s' failed with exception %s :(" % (__id, _e))
        return

    def _do_counting(self) -> None:
        r"""
        Do count in cells with loaded options.
        """
        region             = self.options['region'                     ]
        patchsize          = self.options['patchsize'                  ]
        pixsize            = self.options['pixsize'                    ]
        bad_regions        = self.options['badRegions'                 ]
        patch_details_path = self.options['patchOutputFile'            ]
        path_r             = self.options['randomCatalog','path'       ]
        use_masks_r        = self.options['randomCatalog','masks'      ]
        data_filters_r     = self.options['randomCatalog','filters'    ]
        expressions_r      = self.options['randomCatalog','expressions']
        extra_bins_r       = self.options['randomCatalog','extraBins'  ]
        coord_r            = self.options['randomCatalog','coord'      ]
        csv_opts_r         = self.options['randomCatalog','csvOptions' ]
        path_o             = self.options['objectCatalog','path'       ]
        use_masks_o        = self.options['objectCatalog','masks'      ]
        data_filters_o     = self.options['objectCatalog','filters'    ]
        expressions_o      = self.options['objectCatalog','expressions']
        extra_bins_o       = self.options['objectCatalog','extraBins'  ]
        coord_o            = self.options['objectCatalog','coord'      ]
        csv_opts_o         = self.options['objectCatalog','csvOptions' ]
        output_path        = self.options['countOutputFile'            ]
        variable_mapping   = self.options['variables'                  ]
        if extra_bins_r is not None: extra_bins_r = {feature: [float(x) for x in value] for feature, value in extra_bins_r.items()}      
        if extra_bins_o is not None: extra_bins_o = {feature: [float(x) for x in value] for feature, value in extra_bins_o.items()} 
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
                           skip_cell_prepartion        = self.args.flag == 1,
                           stop_after_cell_preparation = self.args.flag == 2, 
                           use_mpi                     = self.args.use_mpi, )        
        return 

if __name__ == '__main__': App().run()
