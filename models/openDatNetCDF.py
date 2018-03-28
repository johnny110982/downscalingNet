# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:36:30 2018

@author: chang

process netcdf files and dat files.
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
#import netCDF4
#from netCDF4 import Dataset

#just want to open netcdf file
file=xr.open_dataset('D:newMMHC12K-DLY.nc')
file

#open dat file
file = np.fromfile('D:output/AlexNet_erai_input.dat')
file

#take part of netcdf data and convert to new ssubset file
file=xr.open_dataset('D:newMMHC12K-DLY.nc')
train = file.incrain[3653:7304,:,:]
train
train.to_netcdf('D:testWRF2005to2015.nc')

#plot
file=xr.open_dataset('D:newMMHC12K-DLY.nc')
sst = file.incrain[100,:,:].plot(levels=[-99999, 0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030])
plt.show()

#examine output images
from pylab import figure, axes, pie, title, show

wrf=np.fromfile('D:output/AlexNet_wrf_output.dat',float32)
a = wrf.reshape(146,23681)
b = a[5].reshape(199,119)
b.shape
plt.imshow(b)
plt.colorbar()
plt.savefig('D:output/1.jpg')
plt.show()

dplg=np.fromfile('D:output/AlexNet_dplg_output.dat',float32)
a = dplg.reshape(146,23681)
b = a[5].reshape(199,119)
b.shape
plt.imshow(b)
plt.colorbar()
plt.savefig('D:output/2.jpg')
plt.show()

erai=np.fromfile('D:output/AlexNet_erai_input.dat',float32)
a = erai.reshape(146,493)
b = a[5].reshape(29,17)
b.shape
plt.imshow(b)
plt.colorbar()
plt.savefig('D:output/3.jpg')
plt.show()

wrffile = xr.open_dataset('D:MMFC12K-DLY.nc')
wrffile
sst = wrffile.incrain[100,:,:].plot(levels=[-99999, 0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030])
plt.show()



#others...
wrf = file.time[0:3000]
wrf
file = fromfile('D:data/output/ERAItoWRF_DS_AlexNet_standardized_20000step_train_erai_NewData_17.dat')
file
file.shape
file=xr.open_dataset('D:trainHistoricalERAI1990to2000.nc')
file

#terminal command
'''
!ncdump -h D:erainterim/mmsfc-200908.nc
!pwd
ncks -d time,0,364 data.nc 1990.nc
'''



