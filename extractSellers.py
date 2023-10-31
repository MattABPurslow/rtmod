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

def unpackSellersOutput(ds, albedoArrFull, albedoArr, idxFull):
  print('Unpacking Sellers output at', datetime.datetime.now().strftime('%H:%M:%S'))
  ## Unpack into xarray and save
  albedoArr = np.array(albedoArr)
  for i in range(len(idxFull)):
    di, yi, xi, LAIi, sai, rPARi, tPARi, rNIRi, tNIRi = idxFull[i]
    albedoArrFull[di, yi, xi, LAIi, sai, rPARi, tPARi, rNIRi, tNIRi, :] = albedoArr[i]
  ds.coords['Sellers_LAI'] = LAIarr
  ds.coords['Sellers_SA'] = snowAlbedoArr
  ds.coords['A_Type'] = ['BSA_PAR', 'BSA_NIR',  'WSA_PAR', 'WSA_NIR']
  ds.coords['rPAR'] = rPAR
  ds.coords['tPAR'] = tPAR
  ds.coords['rNIR'] = rNIR
  ds.coords['tNIR'] = tNIR
  ds['Sellers_Albedo'] = (('date', 'y', 'x', 'Sellers_LAI', 'Sellers_SA',
                           'r_PAR', 't_PAR', 'r_NIR', 't_NIR', 'A_Type'), albedoArrFull)
  MODIS_Albedo = np.full((len(ds.date), len(ds.y), len(ds.x), 4), np.nan)
  MODIS_Albedo[:,:,:,0] = ds.MODIS_BSA_vis.values
  MODIS_Albedo[:,:,:,1] = ds.MODIS_BSA_nir.values
  MODIS_Albedo[:,:,:,2] = ds.MODIS_WSA_vis.values
  MODIS_Albedo[:,:,:,3] = ds.MODIS_WSA_nir.values
  ds['MODIS_Albedo'] = (('date', 'y', 'x', 'A_Type'), MODIS_Albedo)
  ds['Albedo_Difference'] = ds.Sellers_Albedo - ds.MODIS_Albedo
  idx = np.argmin(np.abs(ds.Albedo_Difference.values), axis=3)
  ds['Sellers_LAI_BestFit'] = (('date', 'y', 'x', 'Sellers_SA', 'r_PAR', 't_PAR', 'r_NIR', 't_NIR',  'A_Type'),
                               ds.Sellers_LAI.values[idx])
  return ds

def PlotLAIvLAI(ds):
  ## Plot Sellers best fit LAI against MODIS LAI
  fig = plt.figure(figsize=(7,5))
  ax = [[plt.subplot2grid((2,9), (0,0), colspan=4), plt.subplot2grid((2,9), (0,4), colspan=4)],
        [plt.subplot2grid((2,9), (1,0), colspan=4), plt.subplot2grid((2,9), (1,4), colspan=4)]]
  cax = plt.subplot2grid((2,9), (0,8), rowspan=2)
  cmap = 'coolwarm'
  ## BSA PAR
  im = ax[0][0].scatter(ds.MODIS_LAI, ds.sel(A_Type='BSA_PAR').Sellers_LAI_BestFit,
                  c=ds.sel(A_Type='BSA_PAR', 
                           Sellers_LAI=ds.sel(A_Type='BSA_PAR').Sellers_LAI_BestFit).Albedo_Difference,
                  vmin=-.1, vmax=.1, s=1, cmap=cmap)
  ax[0][0].set_xlabel('MODIS LAI')
  ax[0][0].set_ylabel('Sellers Best Fit LAI')
  ax[0][0].set_title('BSA PAR')
  ax[0][0].set_aspect('equal'); ax[0][0].set_xlim(0,10); ax[0][0].set_ylim(0,10)
  ## BSA NIR
  ax[0][1].scatter(ds.MODIS_LAI, ds.sel(A_Type='BSA_NIR').Sellers_LAI_BestFit,
                  c=ds.sel(A_Type='BSA_NIR', 
                           Sellers_LAI=ds.sel(A_Type='BSA_NIR').Sellers_LAI_BestFit).Albedo_Difference,
                  vmin=-.1, vmax=.1, s=1, cmap=cmap)
  ax[0][1].set_xlabel('MODIS LAI')
  ax[0][1].set_ylabel('Sellers Best Fit LAI')
  ax[0][1].set_title('BSA NIR')
  ax[0][1].set_aspect('equal'); ax[0][1].set_xlim(0,10); ax[0][1].set_ylim(0,10)
  ## WSA PAR
  ax[1][0].scatter(ds.MODIS_LAI, ds.sel(A_Type='WSA_PAR').Sellers_LAI_BestFit,
                  c=ds.sel(A_Type='WSA_PAR', 
                           Sellers_LAI=ds.sel(A_Type='WSA_PAR').Sellers_LAI_BestFit).Albedo_Difference,
                  vmin=-.1, vmax=.1, s=1, cmap=cmap)
  ax[1][0].set_xlabel('MODIS LAI')
  ax[1][0].set_ylabel('Sellers Best Fit LAI')
  ax[1][0].set_title('WSA PAR')
  ax[1][0].set_aspect('equal'); ax[1][0].set_xlim(0,10); ax[1][0].set_ylim(0,10)
  ## WSA NIR
  ax[1][1].scatter(ds.MODIS_LAI, ds.sel(A_Type='WSA_NIR').Sellers_LAI_BestFit,
                  c=ds.sel(A_Type='WSA_NIR', 
                           Sellers_LAI=ds.sel(A_Type='WSA_NIR').Sellers_LAI_BestFit).Albedo_Difference,
                  vmin=-.1, vmax=.1, s=1, cmap=cmap)
  ax[1][1].set_xlabel('MODIS LAI')
  ax[1][1].set_ylabel('Sellers Best Fit LAI')
  ax[1][1].set_title('WSA NIR')
  ax[1][1].set_aspect('equal'); ax[1][1].set_xlim(0,10); ax[1][1].set_ylim(0,10)
  ## Add colorbar & show
  fig.colorbar(im, cax=cax, label='Albedo difference (Sellers - MODIS)')
  cax.set_yticks([-.10,-.05, 0.00, .05, .10], [-.10,-.05, 0.00, .05, .10])
  fig.tight_layout()
  fig.show()

def PlotLAIMap(ds):
  ## Plot side-by-side LAI maps for days where Sellers best fit > 0
  for d in ds.date:
    if ds.sel(date=d).Sellers_LAI_BestFit.max() > 0.001:
      fig, ax = plt.subplots(2, 4, figsize=(8,8), sharex=True, sharey=True)
      ds.sel(A_Type='BSA_PAR', date=d).MODIS_LAI.plot(ax=ax[0,0],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_PAR', date=d).Sellers_LAI_BestFit.plot(ax=ax[0,1],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_NIR', date=d).MODIS_LAI.plot(ax=ax[0,2],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_NIR', date=d).Sellers_LAI_BestFit.plot(ax=ax[0,3],cmap='YlGn',vmin=0,vmax=1)    
      ds.sel(A_Type='WSA_PAR', date=d).MODIS_LAI.plot(ax=ax[1,0],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='WSA_PAR', date=d).Sellers_LAI_BestFit.plot(ax=ax[1,1],cmap='YlGn',vmin=0,vmax=1)    
      ds.sel(A_Type='WSA_NIR', date=d).MODIS_LAI.plot(ax=ax[1,2],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='WSA_NIR', date=d).Sellers_LAI_BestFit.plot(ax=ax[1,3],cmap='YlGn',vmin=0,vmax=1)    
      plt.show()
      import pdb; pdb.set_trace()
      plt.close()

def PlotAlbedoMap(ds):
  for d in ds.date:
    if ds.sel(date=d).Sellers_LAI_BestFit.max() > 0.001:
      fig, ax = plt.subplots(2, 4, figsize=(8,8), sharex=True, sharey=True)
      ds.sel(A_Type='BSA_PAR', date=d).MODIS_Albedo.plot(ax=ax[0,0],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_PAR', date=d, Sellers_LAI=ds.sel(A_Type='BSA_PAR', 
             date=d).Sellers_LAI_BestFit).Sellers_Albedo.plot(ax=ax[0,1],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_NIR', date=d).MODIS_LAI.plot(ax=ax[0,2],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='BSA_NIR', date=d, Sellers_LAI=ds.sel(A_Type='BSA_NIR', 
             date=d).Sellers_LAI_BestFit).Sellers_Albedo.plot(ax=ax[0,3],cmap='YlGn',vmin=0,vmax=1)    
      ds.sel(A_Type='WSA_PAR', date=d).MODIS_LAI.plot(ax=ax[1,0],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='WSA_PAR', date=d, Sellers_LAI=ds.sel(A_Type='WSA_PAR', 
             date=d).Sellers_LAI_BestFit).Sellers_Albedo.plot(ax=ax[1,1],cmap='YlGn',vmin=0,vmax=1)    
      ds.sel(A_Type='WSA_NIR', date=d).MODIS_LAI.plot(ax=ax[1,2],cmap='YlGn',vmin=0,vmax=1)
      ds.sel(A_Type='WSA_NIR', date=d, Sellers_LAI=ds.sel(A_Type='WSA_NIR', 
             date=d).Sellers_LAI_BestFit).Sellers_Albedo.plot(ax=ax[1,3],cmap='YlGn',vmin=0,vmax=1)    
      plt.show()
      import pdb; pdb.set_trace()
      plt.close()

def addSellers(ds, sellers):
  ds['r']= sellers.r
  ds['t']= sellers.t
  ds['Sellers_Albedo'] = sellers.sel(zen=ds.zen, method='nearest').Albedo.drop('zen')
  return ds

def openMODIS(MODISnc):
  ds = xr.open_dataset(MODISnc)
  ds['zen'] = ds.where(ds.zen<90.).zen
  ## Only use dates with non-zero LAI values
  date = ds.date[(ds.MODIS_LAI.max(dim=('x', 'y'))>0.05)]
  ds = ds.sel(date=date)
  return ds

def mapMODIS(ds):
  ds['MODIS_Albedo'] = (('df', 'wv', 'date', 'y', 'x'), np.array([[ds.MODIS_BSA_vis.values, ds.MODIS_BSA_nir.values],
                                                                  [ds.MODIS_WSA_vis.values, ds.MODIS_WSA_nir.values]]))
  ds = ds.drop_vars(['MODIS_BSA_vis', 'MODIS_BSA_nir', 'MODIS_WSA_vis', 'MODIS_WSA_nir',
                     'MODIS_BSA_shortwave', 'MODIS_WSA_shortwave', 'MODIS_BRDF_shortwave_QC'])
  return ds

def getSellersStats(ds):
  ds['Albedo_Difference_Abs'] = np.abs(ds.Sellers_Albedo - ds.MODIS_Albedo)
  idx = ds.Albedo_Difference_Abs.fillna(999).argmin(dim='LAI', skipna=True)
  ds['Sellers_LAI_BestFit'] = ds.LAI[idx].where(np.isnan(ds.Albedo_Difference_Abs.max(dim='LAI'))==False)
  ds = ds.drop_vars(['Albedo_Difference_Abs'])
  return ds
  
def iterateExtraction(MODISnc):
  ## Open MODIS tile
  ds = openMODIS(MODISnc)
  dates = ds.date.values
  for d in dates:
    outFile = os.path.join(outDir, os.path.split(MODISnc)[-1]\
                                   .replace('MODIS', 'combined')\
                                   .replace('nc', str(d)[:10].replace('-','')+'.nc'))
    dsi = ds.sel(date=[d])
    ## Extract Sellers albedo
    dsi = addSellers(dsi, sellers)
    ## Rearrange MODIS to match Sellers dimensions
    dsi = mapMODIS(dsi)
    ## Get Sellers stats
    dsi = getSellersStats(dsi)
    ## Convert stats to int32 with scale factor
    dsi = integerise(dsi)
    ## Output to NetCDF
    dsi.to_netcdf(outFile)

def integerise(ds):
  for k in ['Sellers_Albedo', 'MODIS_Albedo', 'Sellers_LAI_BestFit']:
    ds[k] = (ds[k] / 0.001).astype(np.int16)
    ds[k].attrs['scale_factor'] = 0.001
  return ds    
  
if __name__=='__main__':
  site = 'sodankyla'
  sellersNC = 'Sellers.lookup.coarse.nc'
  MODISdir = '%s/MODISnc' % site
  outDir = '%s/SellersCoarseNC' % site
  sellers = xr.open_dataset(sellersNC)
  MODISFiles = np.sort(glob.glob(os.path.join(MODISdir,'*.nc')))
  with Pool(4) as pool:
    pool.map(iterateExtraction, MODISFiles)

  """
  bins = np.arange(0, 10.1, 0.1)
  vals = np.empty(bins.shape[0]-1)
  for mf in MODISFiles:
    ds = xr.open_dataset(mf)
    v, _ = np.histogram(ds.MODIS_LAI, bins=bins)
    vals += v
  plt.plot(bins[:-1]+0.05, vals)
  plt.xlim(0, 10)
  plt.xlabel('MODIS LAI')
  plt.ylabel('Frequency')
  plt.show()
  
  
  for MODISnc in MODISFiles:
    print(MODISnc)
    ## Open MODIS tile
    ds = openMODIS(MODISnc)
    dates = ds.date.values
    for d in dates:
      outFile = os.path.join(outDir, os.path.split(MODISnc)[-1]\
                                     .replace('MODIS', 'combined')\
                                     .replace('nc', str(d)[:10].replace('-','')+'.nc'))
      dsi = ds.sel(date=[d])
      ## Extract Sellers albedo
      dsi = addSellers(dsi, sellers)
      ## Rearrange MODIS to match Sellers dimensions
      dsi = mapMODIS(dsi)
      ## Get Sellers stats
      dsi = getSellersStats(dsi)
      ## Reduce sizes of multi-dimensional arrays
      dsi = integerise(dsi)
      ## Output to NetCDF
      dsi.to_netcdf(outFile)
  """
