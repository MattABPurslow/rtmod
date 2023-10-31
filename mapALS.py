import os, glob, pdb
import numpy as np
import xarray as xr
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt

modDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MODISnc'
alsNC = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.als.lonlat.nc'

for modTile in glob.glob(os.path.join(modDir, '*.nc')):
  modis = xr.open_dataset(modTile)
  als = xr.open_dataset(alsNC)
  res = 0.00416667
  chm = np.full(modis.lon.shape,np.nan)
  cv = np.full(modis.lon.shape,np.nan)
  for i in range(modis.y.shape[0]):
    for j in range(modis.x.shape[0]):
      dsij = als.sel(x=slice(modis.lon.values[i,j],
                             modis.lon.values[i,j]+res),
                     y=slice(modis.lat.values[i,j],
                             modis.lat.values[i,j]-res))
      chm[i,j] = dsij.chm.mean(skipna=True)
      cv[i,j] = dsij.cv.mean(skipna=True)
  modis['chm'] = (('y', 'x'), chm)
  modis['cv'] = (('y', 'x'), cv)
  modis.to_netcdf(modTile.replace('.nc', '.wALS.nc'))
