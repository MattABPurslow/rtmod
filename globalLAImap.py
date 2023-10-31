import glob, os, pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import rioxarray as rx
from pyproj import CRS, Transformer, Proj

modisDir = '/exports/csce/datastore/geos/groups/3d_env/data/modis/lai/modis_tif'
gediFile = '/exports/csce/datastore/geos/users/s1503751/gedi/GEDI03_rh100_mean_2019108_2022019_002_03.tif'
chmDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/global/chm'
MODProj = Proj('+proj=sinu +R=6371007.181')
GEDIcrs = CRS.from_epsg(6933)
trans = Transformer.from_crs(MODProj.crs, GEDIcrs, always_xy=True)
modisTiles = glob.glob(os.path.join(modisDir, '*.tif'))
modisCells = np.unique([os.path.split(mT)[-1].split('.')[2] for mT in modisTiles])
filled = np.full(modisCells.shape[0], False)
gedi = rx.open_rasterio(gediFile).drop('band')
gedi = gedi.where((gedi > 0) & (gedi < 100))

for mT in modisTiles:
  if filled[modisCells==os.path.split(mT)[-1].split('.')[2]]==False:
    modis = rx.open_rasterio(mT).to_dataset(name='lai')
    modis['lai'] = (modis.lai.where(modis.lai<=100)/10.).fillna(-1)
    X, Y = np.meshgrid(modis.x, modis.y)
    x, y = trans.transform(X, Y)
    modis['e'], modis['n'] = (('y', 'x'), x), (('y', 'x'), y)
    modis['chm'] = gedi.sel(y=modis.n, x=modis.e, method='nearest')
    modis.chm.transpose('band', 'y', 'x').rio.to_raster(os.path.join(chmDir, os.path.split(mT)[-1].replace('MCD15A2H', 'chm')), dtype=np.float32, driver='GTiff', )
    filled[modisCells==os.path.split(mT)[-1].split('.')[2]] = True
