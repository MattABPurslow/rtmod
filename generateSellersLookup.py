import glob, os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from progress.bar import Bar
from pySellersTwoStream.sellersTwoStream import twoStream

def runSellers(zen, df, SA, LAI, leaf_r, leaf_t):
  if leaf_r+leaf_t > 1.:
    return np.nan
  else:
    t = twoStream()
    t.setupJULES()
    #cosine of solar zenith:
    t.mu=np.cos(np.deg2rad(zen))
    #leaf area index
    t.lai=LAI
    #leaf reflectance & tranmisttance
    t.leaf_r=leaf_r #=alpar
    t.leaf_t=leaf_t #=omega-alpar
    #soil reflectance
    t.soil_r=SA
    #number of layers
    t.nLayers=20
    #proportion of diffuse radiation
    t.propDif=df
    #do the radiative transfer calculation:
    IupPAR, _, _, _ = t.getFluxes()
    return IupPAR[0]

runSellers = np.vectorize(runSellers)

if __name__=='__main__':
  ds = xr.Dataset()
  ds.coords['zen'] = np.arange(50.0, 90., 0.5)
  ds.coords['wv'] = np.array(['PAR', 'NIR'])
  ds.coords['df'] = np.array([0, 1])
  ds.coords['LAI'] = np.arange(0.0, 3.01, 0.1)
  ds.coords['SA'] = np.arange(0.1, 0.91, 0.1)
  ds.coords['r'] = np.arange(0.1, 0.91, 0.1)
  ds.coords['t'] = np.arange(0.1, 0.91, 0.1)
  ds = ds.chunk(ds.dims)
  print(ds.head())
  print(np.prod([ds.dims[k] for k in ds.dims.keys()]), 'permutations')
  print('Starting Sellers @', datetime.datetime.now().strftime('%H:%M:%S'))
  ds['Albedo'] = xr.apply_ufunc(runSellers, ds.zen, ds.df, ds.SA, ds.LAI, ds.r, ds.t, dask="parallelized")
  ds = ds.to_netcdf('Sellers.lookup.coarse.nc')
  print('Sellers complete @', datetime.datetime.now().strftime('%H:%M:%S'))

  """
  ds = xr.Dataset()
  ds.coords['zen'] = np.arange(0.0, 90.0, 0.5)
  ds.coords['df'] = np.array([0, 1])
  ds.coords['wv'] = np.array(['PAR', 'NIR'])
  ds.coords['LAI'] = np.arange(0.0, 10.01, 0.1) #np.array([0., .1, .2, .5, 1., 2., 5.])
  ds.coords['SA'] = np.arange(0.1, 0.901, 0.1) # np.array([0.1, 0.3, 0.5])
  ds['r'] = (('wv', 'r_idx'), np.array([np.arange(0.05, 0.151, 0.02), np.arange(0.3, 0.4, 0.02)]))
  ds.coords['r_idx'] = np.arange(ds.dims['r_idx'])
  ds['t'] = (('wv', 't_idx'), np.array([np.arange(0.05, 0.151, 0.02), np.arange(0.3, 0.4, 0.02)]))
  ds.coords['t_idx'] = np.arange(ds.dims['t_idx'])
  ds = ds.chunk(ds.dims)
  print(ds.head())
  print(np.prod([ds.dims[k] for k in ds.dims.keys()]), 'permutations')
  print('Starting Sellers @', datetime.datetime.now().strftime('%H:%M:%S'))
  ds['Albedo'] = xr.apply_ufunc(runSellers, ds.zen, ds.df, ds.SA, ds.LAI, ds.r, ds.t, dask="parallelized")
  ds = ds.to_netcdf('Sellers.lookup.nc')
  print('Sellers complete @', datetime.datetime.now().strftime('%H:%M:%S'))
  """
