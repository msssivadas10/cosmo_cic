#!/usr/bin/python3

# add module location to path
import sys, os.path as path
sys.path.append(path.split(path.split(__file__)[0])[0])

import numpy as np
import matplotlib.pyplot as plt
from cic.misc.generate_p3 import p3Generator1

df = p3Generator1(ra1 = 0., ra2 = 40., dec1 = 0., dec2 = 10., density = 500.)
df.to_csv('p3catalog.csv', index = False)
