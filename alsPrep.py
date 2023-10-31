import os, glob
import numpy as np
import xarray as xr
import rioxarray as rx
import matplotlib.pyplot as plt

alsDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS'
todrop = ['band', 'spatial_ref']

ds = xr.Dataset()
ds['cv'] = rx.open_rasterio(os.path.join(alsDir,
           'sodankyla.cv.modisgrid.tif')).sel(band=1).drop_vars(todrop)
ds['dsm'] = rx.open_rasterio(os.path.join(alsDir,
            'sodankyla.dsm.modisgrid.tif')).sel(band=1).drop_vars(todrop)
ds['dtm'] = rx.open_rasterio(os.path.join(alsDir,
            'sodankyla.dtm.modisgrid.tif')).sel(band=1).drop_vars(todrop)
mask = (ds.cv >= 0.)&(ds.cv <= 100.)
ds = ds.where(mask)
ds['chm'] = ds.dsm - ds.dtm
ds['chm'] = ds.chm.where((ds.chm >= 0) & (ds.chm <= 100))
ds.to_netcdf(os.path.join(alsDir, 'sodankyla.als.nc'))
