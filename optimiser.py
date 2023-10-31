import glob, os, pdb
import datetime
import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from progress.bar import Bar
from pySellersTwoStream.sellersTwoStream import twoStream

from scipy.optimize import least_squares

def runSellers(t, mu, LAI, SA, omega, leaf_r, leaf_t):
  #cosine of solar zenith:
  t.mu=mu
  #leaf area index
  t.lai=omega*LAI
  #leaf reflectance & tranmisttance
  t.leaf_r=leaf_r #=alpar
  t.leaf_t=leaf_t #=omega-alpar
  #soil reflectance
  t.soil_r=SA
  #do the radiative transfer calculation:
  IupPAR, _, _, _ = t.getFluxes()
  return IupPAR[0]

runSellers = np.vectorize(runSellers)

def getError(t, BSA, WSA, mu, LAI, SA, omega, leaf_r, leaf_t):
  #proportion of diffuse radiation
  t.propDif=0
  directAlbedo = runSellers(t, mu, LAI, SA, omega, leaf_r, leaf_t)
  #proportion of diffuse radiation
  t.propDif=1
  diffuseAlbedo = runSellers(t, mu, LAI, SA, omega, leaf_r, leaf_t)
  return np.array([((directAlbedo-BSA)**2)**0.5, ((diffuseAlbedo-WSA)**2)**0.5]).flatten()

def trySellers(params, mu, LAI, BSA, WSA):
  SA = params[:-3]
  omega, leaf_r, leaf_t = params[-3:]
  mu, LAI, BSA, WSA = data
  #def trySellers(params, mu, LAI, BSA, WSA, SA):
  #omega, leaf_r, leaf_t = params
  #mu, LAI, BSA, WSA, SA = data
  t = twoStream()
  t.setupJULES()
  #number of layers
  t.nLayers=1
  return getError(t, BSA, WSA, mu, LAI, SA, omega, leaf_r, leaf_t)

def getMODIS(MODISdir):
  MODISlist = sorted(glob.glob(os.path.join(MODISdir,'*.nc')))
  ds = xr.Dataset()
  varNames = ['zen', 'MODIS_LAI', 'MODIS_BSA_vis', 'MODIS_WSA_vis', 'MODIS_Snow_Albedo', 'IGBP_LC', 'PFT_LC', 'lon', 'lat']
  for MODISfile in MODISlist:
    ds = xr.merge([ds, xr.open_dataset(MODISfile)[varNames]])
  pft = np.array(list(ds.PFT_LC.values.flatten())*ds.date.shape[0]).astype(int)
  mu = np.cos(np.deg2rad(ds.zen.values.flatten()))
  LAI = ds.MODIS_LAI.values.flatten()
  BSA = ds.MODIS_BSA_vis.values.flatten()
  WSA = ds.MODIS_WSA_vis.values.flatten()
  SA = ds.MODIS_Snow_Albedo.values.flatten()
  SA[SA==0] = 1e-6 # Remove zero surface albedo values
  mask = (np.isnan(pft)==False) & \
         (np.isnan(mu)==False) & \
         (np.isnan(LAI)==False) & \
         (np.isnan(BSA)==False) & \
         (np.isnan(WSA)==False) & \
         (np.isnan(SA)==False)
  return ds, pft, mu, LAI, BSA, WSA, SA, np.argwhere(mask)

def getLCStats(LandCover, lc, var):
  lcMask = LandCover['LC_Type5'] == LandCover['LC_Type5'].attrs[lc]
  lcMean = float(LandCover[var].where(lcMask).mean(skipna=True))
  lcMedian = float(LandCover[var].where(lcMask).median(skipna=True))
  lcSigma = float(LandCover[var].where(lcMask).std(skipna=True))
  lcMin = float(LandCover[var].where(lcMask).min(skipna=True))
  lcMax = float(LandCover[var].where(lcMask).max(skipna=True))
  return lcMean, lcMedian, lcSigma, lcMin, lcMax

def getLCdata():
  Clumping = rx.open_rasterio('He2012/global_clumping_index.tif').sel(band=1)
  LandCover = rx.open_rasterio('He2012/MCD12Q1.A2006001.h19v02.006.2018145205215.hdf').sel(band=1)
  LandCover2019 = rx.open_rasterio('/home/matt/Documents/uni/phd/MODIS/sodankyla/MCD12Q1/MCD12Q1.A2019001.h19v02.006.2020212131416.hdf').sel(band=1)
  
  ## Select Land Cover tile clumping factors
  LandCover['Clumping_Index'] = (('y', 'x'), Clumping.sel(y=LandCover.y, x=LandCover.x, method='nearest').values)
  LandCover['Clumping_Index'] = LandCover.Clumping_Index.where(LandCover.Clumping_Index!=255) / 100
  
  ## Reduce to places with same land cover type now  
  noChange = LandCover2019['LC_Type5']==LandCover['LC_Type5']
  LandCover = LandCover.where(noChange)
  return LandCover
  
def getPFTparams(LandCover, lc):
  ## Extract PFT stats
  ciMean, ciMedian, ciSigma, ciMin, ciMax = getLCStats(LandCover, lc, 'Clumping_Index')
  rt_dict = {'Barren': 0.07,
             'Broadleaf Crop': 0.07,
             'Cereal Crop': 0.07,
             'Deciduous Broadleaf Trees': 0.07,
             'Deciduous Needleleaf Trees': 0.07,
             'Evergreen Broadleaf Trees': 0.07,
             'Evergreen Needleleaf Trees': 0.07,
             'Grass': 0.07,
             'Permanent Snow and Ice': 0.07,
             'Shrub': 0.07,
             'Urban and Built-up Lands': 0.07,
             'Water Bodies': 0.07}
  return ciMean, rt_dict[lc], rt_dict[lc]

def getPFTname(LandCover, pftVal):
  dropList = ['_FillValue', 'scale_factor', 'add_offset', 'long_name', 'valid_range', 'Unclassified']
  lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items() if k not in dropList}
  return lcDict[pftVal]

def unpackResult(result):
  SAout = result.x[:-3]
  omegaOut, rOut, tOut = result.x[-3:]
  return SAout, omegaOut, rOut, tOut

#def unpackResult(result):
#  omegaOut, rOut, tOut = result.x
#  return omegaOut, rOut, tOut
  
if __name__=='__main__':
  MODISdir = 'sodankyla/MODISnc'
  LandCover = getLCdata()
  ds, pft, mu, LAI, BSA, WSA, SA, idx = getMODIS(MODISdir)
  pftList = np.sort(np.unique(pft[idx]).astype(int))
  SA_fit = np.full(ds.MODIS_LAI.shape, np.nan)
  omega_fit =  np.full(ds.MODIS_LAI.shape, np.nan)
  r_fit =  np.full(ds.MODIS_LAI.shape, np.nan)
  t_fit =  np.full(ds.MODIS_LAI.shape, np.nan)
  for pftVal in pftList:
    pftName = getPFTname(LandCover, pftVal)
    print(pftName, '( PFT', pftVal, 'of', len(pftList), ')', end='\r')
    i = np.array([i for i in np.argwhere(pft.astype(int) == pftVal) if i in idx]).T[0]
    data = np.array([mu[i], LAI[i], BSA[i], WSA[i]])
    #data = np.array([mu[i], LAI[i], BSA[i], WSA[i], SA[i]])
    omega, r, t = getPFTparams(LandCover, pftName)
    params = np.append(SA[i], [omega, r, t])
    #params = np.array([omega, r, t])
    bounds = (np.full(params.shape, 1e-6), np.full(params.shape, 1.-1e-6))
    #bounds = (params-params/4., params+params/4.)
    result = least_squares(trySellers, params, args=data, bounds=bounds)
    SAout, omegaOut, rOut, tOut = unpackResult(result)
    #omegaOut, rOut, tOut = unpackResult(result)
    i = np.unravel_index(i, ds.MODIS_LAI.shape)
    SA_fit[i] = SAout
    omega_fit[i] = omegaOut
    r_fit[i] = rOut
    t_fit[i] = tOut
  print('', end='\r')
  print('All PFTs fitted')
  ds['SA_fit'] = (('date', 'y', 'x'), SA_fit)
  ds['omega_fit'] = (('date', 'y', 'x'), omega_fit)
  ds['r_fit'] = (('date', 'y', 'x'), r_fit)
  ds['t_fit'] = (('date', 'y', 'x'), t_fit)
  ds.to_netcdf('sodankyla.optimized.nc')

