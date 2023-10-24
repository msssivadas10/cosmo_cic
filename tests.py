import logging
from measure.utils import generateMockCatalog, Rectangle
from measure.counting import prepareRegion, countObjects



logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [ logging.StreamHandler() ])
# logging.disable()

generateMockCatalog(100_000).to_csv('mock_data.csv', index = False)

prepareRegion(path         = 'mock_data.csv', 
              output_path  = 'patch.json',
              region       = Rectangle(0, 0, 1, 1),
              patchsize_x  = 0.5, 
              patchsize_y  = 0.5, 
              pixsize      = 0.1,
              bad_regions  = [],
              use_masks    = ['u_mask', 'v_mask'], 
              data_filters = [], 
              expressions  = [], )

countObjects(path               = 'mock_data.csv', 
             patch_details_path = 'patch.json', 
             output_path        = 'count.json',
             use_masks          = ['u_mask', 'v_mask'], 
             data_filters       = ['redshift < 1.'], 
             expressions        = [], )

