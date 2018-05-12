# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:26:26 2018

@author: chang

comparison of TRMM and WRF
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

ERAI_SHAPE = [90,58]
ERAI_LENGTH = ERAI_SHAPE[0]*ERAI_SHAPE[1]

WRF_SHAPE = [199,119]
WRF_LENGTH = WRF_SHAPE[0]*WRF_SHAPE[1]

#take part of netcdf data and convert to new ssubset file
file=xr.open_dataset('D:1998wrf.nc')
train = file.incrain[151:181,:,:]
train.time
#train.to_netcdf('D:1998wrffeb.nc')

s = np.zeros((199,119))
for i in range(30):
    s = s + train[i,:,:]
print(s.shape)
s.to_netcdf('D:1998wrfjunsum.nc')

file = xr.open_dataset('D:TRMM_monthly/traintrmm1998to2007.nc')
train = file.precip[7,:,:,:]
train.time
train.to_netcdf('D:TRMM_monthly/1998trmmaug.nc')

#read training data
print('reading train trmm')
ds = xr.open_dataset('D:TRMM_monthly/trainTRMM1998to2007.nc')
er = ds.precip
erai_train = np.array(er).reshape(-1,ERAI_LENGTH)
print (erai_train.shape)

print('reading train wrf')
ds = xr.open_dataset('D:trainWRF1995to2005.nc')
er = ds.incrain
wrf_train = np.array(er).reshape(-1,WRF_LENGTH)
print (wrf_train.shape)


