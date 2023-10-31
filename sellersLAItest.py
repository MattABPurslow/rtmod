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

def getRunData(ds, LAIarr, snowAlbedoArr, rPAR, tPAR, rNIR, tNIR):
  print('Generating run data at', datetime.datetime.now().strftime('%H:%M:%S'))
  ## Run Sellers for each point across possible MODIS LAIs
  albedoArrFull = np.full((len(ds.date), len(ds.y), len(ds.x), len(LAIarr), len(snowAlbedoArr), 
                           len(rPAR), len(tPAR), len(rNIR), len(tNIR), 4), np.nan)
  mask = (np.isnan(ds.MODIS_LAI)==False) & \
         (np.isnan(ds.MODIS_BSA_vis)==False) & \
         (np.isnan(ds.MODIS_BSA_nir)==False) & \
         (np.isnan(ds.MODIS_WSA_vis)==False) & \
         (np.isnan(ds.MODIS_WSA_nir)==False)
  idx = np.argwhere(np.isnan(ds.where(mask).zen.values)==False)
  idxFull = []
  for i in idx:
    di, yi, xi = i
    for LAIi in range(len(LAIarr)):
      for sai in range(len(snowAlbedoArr)):
        for rPARi in  range(len(rPAR)):
          for tPARi in  range(len(tPAR)):
            for rNIRi in  range(len(rNIR)):
              for tNIRi in  range(len(tNIR)):
                idxFull.append([di, yi, xi, LAIi, sai, rPARi, tPARi, rNIRi, tNIRi])
  idxFull = np.array(idxFull)
  PARmask = (rPAR[idxFull[:,5]] + tPAR[idxFull[:,6]]) <= 1.
  NIRmask = (rNIR[idxFull[:,7]] + tNIR[idxFull[:,8]]) <= 1.
  idxFull = idxFull[PARmask&NIRmask]
  return albedoArrFull, np.array(idxFull)

## MODIS VIS is 0.3-0.7μm, JULES PAR is ?
## MODIS NIR is 0.7-5.0μm, JULES NIR is ?
def getSellers(t, LAI, zenith, snowAlbedo, rPAR=0.07, tPAR=0.07,
               rNIR=0.35, tNIR=0.35, diffFrac=0.5):
  #cosine of solar zenith:
  t.mu=np.cos(np.deg2rad(zenith))
  #leaf area index
  t.lai=LAI
  #leaf reflectance & tranmisttance in PAR (JULES r=0.7 with assumed t=r from MODIS)  t.leaf_r=0.07#0.10 #=alpar
  t.leaf_r=rPAR #=alpar
  t.leaf_t=tPAR #=omega-alpar
  #soil reflectance
  t.soil_r=snowAlbedo
  #number of layers
  t.nLayers=20
  #proportion of diffuse radiation
  t.propDif=diffFrac
  #do the radiative transfer calculation:
  IupPAR, IdnPAR, IabPAR, Iab_dLaiPAR = t.getFluxes()
  #leaf reflectance & tranmisttance in NIR (JULES r=0.35 with assumed t=r from MODIS)
  t.leaf_r=rNIR #=alnir
  t.leaf_t=tNIR #=omnir-alnir
  #do the radiative transfer calculation:
  IupNIR, IdnNIR, IabNIR, Iab_dLaiNR = t.getFluxes()
  return  IupPAR[0], IupNIR[0]

def runSellers(idx):
  t=twoStream()
  t.setupJULES()
  di, yi, xi, LAIi, sai, rPARi, tPARi, rNIRi, tNIRi = idx
  IupPAR_dir, IupNIR_dir = getSellers(t, LAIarr[LAIi], ds.zen.values[di, yi, xi], snowAlbedoArr[sai],
                                      rPAR[rPARi], tPAR[tPARi], rNIR[rNIRi], tNIR[tNIRi], 0.)
  IupPAR_dif, IupNIR_dif = getSellers(t, LAIarr[LAIi], ds.zen.values[di, yi, xi], snowAlbedoArr[sai],
                                      rPAR[rPARi], tPAR[tPARi], rNIR[rNIRi], tNIR[tNIRi], 1.)
  return np.array([IupPAR_dir, IupNIR_dir, IupPAR_dif, IupNIR_dif])

def iterateSellers(idxFull):
  print('Running Sellers at', datetime.datetime.now().strftime('%H:%M:%S'))
  ## Run Sellers for all iterations
  with Pool(8) as pool:
    albedoArr = pool.map(runSellers, idxFull)
  return albedoArr

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

if __name__=='__main__':
  site = 'sodankyla'
  MODISdir = '%s/MODISnc' % site
  outDir = '%s/Sellersnc' % site
  MODISFiles = np.sort(glob.glob(os.path.join(MODISdir,'*.nc')))
  tileCount = 1
  for MODISnc in MODISFiles:
    print(' '.join(['Tile', str(tileCount), 'of', str(len(MODISFiles))])); tileCount += 1
    ds = xr.open_dataset(MODISnc)
    ds['zen'] = ds.where(ds.zen<90.).zen
    ## Only use dates with non-zero LAI values
    date = ds.date[(ds.MODIS_LAI.max(dim=('x', 'y'))>0.05)]
    ds = ds.sel(date=date)
    ## Arrays of Sellers inputs to loop over
    LAIarr = np.array([0., .1, .2, .5, 1., 2., 5.])
    snowAlbedoArr = np.array([0.1, 0.3, 0.5])
    rPAR = np.array([0.05, 0.1, 0.15])
    tPAR = np.array([0.05, 0.1, 0.15])
    rNIR = np.array([0.3, 0.35, 0.4])
    tNIR = np.array([0.3, 0.35, 0.4])
    ## Generate array to hold Sellers output
    albedoArrFull, idxFull = getRunData(ds, LAIarr, snowAlbedoArr, rPAR, tPAR, rNIR, tNIR)
    ## Run Sellers in parallel
    albedoArr = iterateSellers(idxFull)
    ## Unpack Sellers output into dataset
    ds = unpackSellersOutput(ds, albedoArrFull, albedoArr, idxFull)
    ## Save Sellers output to NetCDF
    ##ds.to_netcdf(os.path.join(outDir, os.path.split(MODISnc)[-1].replace('MODIS', 'Sellers')))
