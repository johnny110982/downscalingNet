# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:58:22 2018

@author: chang
"""

import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np
import nco
from numpy import *

file=xr.open_dataset('D:trmm/3B43.20151201.7.HDF.nc')
file