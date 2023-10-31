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

def trySellers(omega, i, j, k):
  print('new iteration @', datetime.datetime.now(), '| Ω̅ =', omega.mean())
  t = twoStream()
  t.setupJULES()
  BSA = ds.MODIS_BSA_vis.values[(i,j,k)]
  WSA = ds.MODIS_WSA_vis.values[(i,j,k)]
  mu = ds.mu.values[(i,j,k)]
  LAI = ds.MODIS_LAI.values[(i,j,k)]
  SA = ds.MODIS_Snow_Albedo.values[(i,j,k)]
  SA[SA==0] = 1e-6
  leaf_r, leaf_t = getPFTparams(pftName)
  #number of layers
  t.nLayers=20
  return getError(t, BSA, WSA, mu, LAI, SA, omega, leaf_r, leaf_t)

def getMODIS(MODISdir):
  MODISlist = sorted(glob.glob(os.path.join(MODISdir,'*.nc')))
  ds = xr.Dataset()
  varNames = ['zen', 'MODIS_LAI', 'MODIS_BSA_vis', 'MODIS_WSA_vis', 'MODIS_Snow_Albedo', 'IGBP_LC', 'PFT_LC', 'lon', 'lat']
  for MODISfile in MODISlist:
    ds = xr.merge([ds, xr.open_dataset(MODISfile)[varNames]])
  LandCover = getLCdata()
  ds['omega'] = LandCover.Clumping_Index.sel(x=ds.x, y=ds.y, method='nearest')
  ds['mu'] = np.cos(np.deg2rad(ds.zen))
  mask = (np.isnan(np.full((ds.mu.shape), ds.omega.values))==False) & \
         (np.isnan(np.full((ds.mu.shape), ds.PFT_LC.values))==False) & \
         (np.isnan(ds.mu.values)==False) & \
         (np.isnan(ds.MODIS_LAI.values)==False) & \
         (np.isnan(ds.MODIS_BSA_vis.values)==False) & \
         (np.isnan(ds.MODIS_WSA_vis.values)==False) & \
         (np.isnan(ds.MODIS_Snow_Albedo.values)==False)
  return ds, LandCover, np.argwhere(mask)

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
  
def getPFTparams(lc):
  ## Extract PFT stats
  r_dict = {'Barren': 0.18,
            'Broadleaf Crop': 0.1,
            'Cereal Crop': 0.1,
            'Deciduous Broadleaf Trees': 0.1,
            'Deciduous Needleleaf Trees': 0.07,
            'Evergreen Broadleaf Trees': 0.1,
            'Evergreen Needleleaf Trees': 0.07,
            'Grass': 0.1,
            'Permanent Snow and Ice': 0.75,
            'Shrub': 0.1,
            'Urban and Built-up Lands': 0.18,
            'Water Bodies': 0.12}
  t_dict = {'Barren': 0.,
            'Broadleaf Crop': 0.15,
            'Cereal Crop': 0.17,
            'Deciduous Broadleaf Trees': 0.15,
            'Deciduous Needleleaf Trees': 0.15,
            'Evergreen Broadleaf Trees': 0.15,
            'Evergreen Needleleaf Trees': 0.15,
            'Grass': 0.15,
            'Permanent Snow and Ice': 0.,
            'Shrub': 0.15,
            'Urban and Built-up Lands': 0.,
            'Water Bodies': 0.}
  return r_dict[lc], t_dict[lc]

def getPFTname(LandCover, pftVal):
  dropList = ['_FillValue', 'scale_factor', 'add_offset', 'long_name', 'valid_range', 'Unclassified']
  lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items() if k not in dropList}
  return lcDict[pftVal]
  
if __name__=='__main__':
  MODISdir = 'sodankyla/MODISnc'
  ds, LandCover, idx = getMODIS(MODISdir)
  pftList = np.sort(np.unique(ds.PFT_LC.values).astype(int))
  omega_fit = np.full(ds.omega.shape, np.nan)
  for pftVal in pftList:
    pftName = getPFTname(LandCover, pftVal)
    print(pftName, '( PFT', pftVal, 'of', len(pftList), ')', end='\r')
    iPFT = np.argwhere(ds.PFT_LC.values.astype(int) == pftVal)
    iPFT_complex = iPFT[:,0]+iPFT[:,1]*1j
    idx_complex = idx[:,1]+idx[:,2]*1j
    data = np.array(idx[np.isin(idx_complex, iPFT_complex)]).T
    if data.shape[1] > 0:
      result = least_squares(trySellers, ds.omega.values[(data[1,:],data[2,:])],
                             args=data, bounds=(0., 1.))
      omega_fit[data] = result.x
  print('', end='\r')
  print('All PFTs fitted')
  ds['omega_fit'] = (('y', 'x'), omega_fit)

