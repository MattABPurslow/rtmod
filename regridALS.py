import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
from pyproj import CRS, Proj, Transformer
import pvlib
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap

from multiprocessing import Pool
from functools import partial
from progress.bar import Bar

class RTMOD(object):
  def __init__(self, rootDir, ALSfile):
    self.rootDir = rootDir
    self.ALSfile = ALSfile
  
  def main(self):
    self.getMODISgrid()
    self.getALS()
    self.saveGrid()
  
  def getMODISgrid(self):
    lcFile = sorted(glob.glob(os.path.join(self.rootDir, 'MCD12Q1/*.hdf')))[-1]
    lc = rx.open_rasterio(lcFile)
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("4326"),
                                 always_xy=True)
    X, Y = np.meshgrid(lc.x, lc.y)
    lon, lat = trans.transform(X, Y)
    lc['lon'] = (('y', 'x'), lon)
    lc['lat'] = (('y', 'x'), lat)
    self.grid = lc[['lon', 'lat']].drop_vars('spatial_ref')
    self.grid.attrs.clear()
  
  def getALS(self):
    als = xr.open_dataset(self.ALSfile)
    als = als.sel(y=slice(self.grid.y.max(), self.grid.y.min()),
                  x=slice(self.grid.x.min(), self.grid.x.max()))
    xArr = np.sort(np.unique(self.grid.x.sel(x=als.x, method='ffill')))
    yArr = np.sort(np.unique(self.grid.y.sel(y=als.y, method='ffill')))[::-1]
    xArr = np.append(xArr, xArr[-1]+(xArr[-1]-xArr[-2]))
    yArr = np.append(yArr, yArr[-1]+(yArr[-1]-yArr[-2]))
    chm = np.full((len(yArr)-1, len(xArr)-1), 0.)
    cv = np.copy(chm)
    for i in range(len(yArr)-1):
      for j in range(len(xArr)-1):
        alsij = als.sel(x=slice(xArr[j],xArr[j+1]), y=slice(yArr[i],yArr[i+1]))
        chm[i,j] = alsij.chm.mean(skipna=True).values
        cv[i,j] = alsij.cv.mean(skipna=True).values
    self.grid = self.grid.sel(y=yArr[:-1], x=xArr[:-1])
    self.grid['chm'] = (('y', 'x'), chm)
    self.grid['cv'] = (('y', 'x'), cv/100.)
  
  def saveGrid(self):
    self.grid.to_netcdf(self.ALSfile.replace('.nc', 'modisgrid.nc'), format='NETCDF3_CLASSIC')

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.als.nc'
  rtmod = RTMOD(rootDir, ALSfile)
  rtmod.main()
