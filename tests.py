#!/usr/bin/python3

def testCatelogGeneration():

    from cic.misc.generate_p3 import p3Generator1

    df = p3Generator1(ra1 = 0., ra2 = 40., dec1 = 0., dec2 = 10., density = 500.)
    df.to_csv('p3catalog.csv', index = False)
    return

def testCounting():

    import logging

    logging.basicConfig(level    = logging.INFO,
                        format   = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [ logging.StreamHandler() ])
    # logging.disable()

    from cic.measure.utils import Rectangle
    from cic.measure.counting import prepareRegion, countObjects

    prepareRegion(path         = 'p3catalog.csv', 
                  output_path  = 'test_counts.json',
                  region       = Rectangle(0., 0., 40., 10.),
                  patchsize_x  = 10., 
                  patchsize_y  = 10., 
                  pixsize      = 1.0,
                  bad_regions  = [],
                  use_masks    = ['g_mask', 'r_mask', 'i_mask'], 
                  data_filters = [], 
                  expressions  = [], 
                  x_coord      = 'ra', 
                  y_coord      = 'dec', 
                  chunksize    = 10000,
                  header       = 0, 
                )

    countObjects(path               = 'p3catalog.csv', 
                 patch_details_path = 'test_counts.json', 
                 output_path        = 'test_counts.json',
                 use_masks          = ['g_mask', 'r_mask', 'i_mask'], 
                 data_filters       = ['abs(redshift - 0.5) < 0.1', 'g_magnitude < 22.0'], 
                 expressions        = [], 
                 x_coord            = 'ra', 
                 y_coord            = 'dec', 
                 chunksize          = 10000,
                 header             = 0, 
                )
    
    return

if __name__ == '__main__':
    print("testing...")

    # testCatelogGeneration()
    # testCounting()