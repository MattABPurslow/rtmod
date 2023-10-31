import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap

from multiprocessing import Pool
from functools import partial
from progress.bar import Bar
from pySellersTwoStream.sellersTwoStream import twoStream

from scipy.optimize import least_squares

def runSellers(t, mu, LAI, SA, omega, leaf_r, leaf_t):
  if np.isnan(mu)|np.isnan(LAI)|np.isnan(SA)|np.isnan(omega):
    return np.nan
  elif SA==0.:
    return np.nan
  else:
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
    Iup, _, _, _ = t.getFluxes()
    return Iup[0]

class RTMOD(object):
  def __init__(self, MODISdir, testing=False, als=False):
    ## Load in MODIS data
    self.MODISdir = MODISdir
    self.getMODIS(testing=testing, als=als)
    ## Create Sellers instance
    self.createSellers()
    self.runSellers = np.vectorize(runSellers)

  def main(self):
    ## Fit Sellers for each PFT
    print('PFTs to fit:', ', '.join([self.lcDict[k] for k in self.pftList]))
    self.ds['Sellers_BSA_vis'] = (('date', 'y', 'x'), np.full(self.ds.MODIS_LAI.shape, np.nan))
    self.ds['Sellers_BSA_nir'] = (('date', 'y', 'x'), np.full(self.ds.MODIS_LAI.shape, np.nan))
    self.ds['Sellers_WSA_vis'] = (('date', 'y', 'x'), np.full(self.ds.MODIS_LAI.shape, np.nan))
    self.ds['Sellers_WSA_nir'] = (('date', 'y', 'x'), np.full(self.ds.MODIS_LAI.shape, np.nan))
    for pftVal in self.pftList[::-1]:
      self.trySellers()
    print('All PFTs fitted')

  def createSellers(self):
    # create Sellers instance
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1
  
  def sellersUfunc(self, mu, LAI, SA, omega, pft, wv):
    if np.isnan(mu)|np.isnan(LAI)|np.isnan(SA)|np.isnan(omega)|np.isnan(pft):
      return np.nan
    elif SA==0.:
      return np.nan
    else:
      pftName = self.lcDict[pft]
      #cosine of solar zenith:
      self.t.mu=mu
      #leaf area index
      self.t.lai=LAI/omega
      #leaf reflectance & tranmisttance
      self.t.leaf_r, self.t.leaf_t = self.rtDict[pftName]['r'][wv], self.rtDict[pftName]['r'][wv]
      #soil reflectance
      self.t.soil_r=SA
      #do the radiative transfer calculation:
      Iup, _, _, _ = self.t.getFluxes()
      return Iup[0]

  def trySellers(self):
    #proportion of diffuse radiation
    self.t.propDif=0
    for df in [0, 1]:
      alb = ['BSA', 'WSA'][df]
      for wv in ['vis', 'nir']:
        self.t.propDif = df
        self.ds['Sellers_%s_%s'%(alb,wv)] = xr.apply_ufunc(np.vectorize(self.sellersUfunc), self.ds.mu,
                                                           self.ds.MODIS_LAI, self.ds.MODIS_Snow_Albedo,
                                                           self.ds.omega, self.ds.PFT_LC, wv)

  def getMODIS(self, testing=False, als=False):
    print('Loading MODIS data')
    ## Read in the processed MODIS data
    MODISlist = sorted(glob.glob(os.path.join(self.MODISdir,'*.nc')))
    self.ds = xr.Dataset()
    varNames = ['zen', 'MODIS_LAI', 'IGBP_LC', 'PFT_LC', 'lon', 'lat',
                'MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover',
                'MODIS_BSA_vis', 'MODIS_WSA_vis',
                'MODIS_BSA_nir', 'MODIS_WSA_nir']
    for MODISfile in MODISlist:
      self.ds = xr.merge([self.ds, xr.open_dataset(MODISfile)[varNames]])
    if als:
      als = xr.open_dataset(os.path.join(self.MODISdir,'../ALS/sodankyla.als.nc'))
      trans = Transformer.from_crs(CRS.from_epsg(32635), CRS.from_epsg(4326),
      
                             always_xy=True)
      X, Y = np.meshgrid(als.x, als.y)
      lon, lat = trans.transform(X,Y)
      lonMin, lonMax, latMin, latMax = lon.min(), lon.max(), lat.min(), lat.max()
      alsMask = (self.ds.lon >= lonMin) & (self.ds.lon <= lonMax) &\
                (self.ds.lat >= latMin) & (self.ds.lat <= latMax)
      self.ds = self.ds.where(alsMask)
      alsIDX = np.argwhere(alsMask.values)
      self.ds = self.ds.isel(y=slice(alsIDX[:,0].min(), alsIDX[:,0].max()),
                             x=slice(alsIDX[:,1].min(), alsIDX[:,1].max()))
    if testing:
      yMin, xMin = np.random.randint(0, self.ds.y.shape[0]-3), np.random.randint(0, self.ds.x.shape[0]-3)
      self.ds = self.ds.isel(y=slice(yMin,yMin+2), x=slice(xMin, xMin+2))
    self.ds['mu'] = np.cos(np.deg2rad(self.ds.zen))
    self.getClumping()
    mask = (np.isnan(np.full((self.ds.mu.shape), self.ds.omega.values))==False) & \
           (np.isnan(np.full((self.ds.mu.shape), self.ds.PFT_LC.values))==False) & \
           (np.isnan(self.ds.mu.values)==False) & \
           (np.isnan(self.ds.MODIS_LAI.values)==False) & \
           (np.isnan(self.ds.MODIS_BSA_vis.values)==False) & \
           (np.isnan(self.ds.MODIS_WSA_vis.values)==False) & \
           (np.isnan(self.ds.MODIS_Snow_Albedo.values)==False)
    self.idx = np.argwhere(mask)
    self.getRTdict()
    ## Identify available PFTs
    self.pftList = np.sort(np.unique(self.ds.PFT_LC.values[np.isnan(self.ds.PFT_LC.values)==False]).astype(int))
    self.pftList = self.pftList[self.pftList > 0]
    self.pftList = np.array([k for k in self.pftList
                             if (self.lcDict[k] in self.rtDict.keys())])
    self.ds['PFT_LC'] = self.ds.PFT_LC.where(np.isin(self.ds.PFT_LC, self.pftList))
  
  def getClumping(self):
    print('Retrieving He et al (2012) clumping index')
    Clumping = rx.open_rasterio(os.path.join(self.MODISdir,
                                '../../global_clumping_index.tif')).sel(band=1)
    LCdir = os.path.join(self.MODISdir, '../MCD12Q1')
    LandFile2006, LandFileNow = sorted(glob.glob(os.path.join(LCdir,'*.hdf')))
    LC2006 = rx.open_rasterio(LandFile2006).sel(band=1)
    LCNow = rx.open_rasterio(LandFileNow).sel(band=1)
    self.getLCdict(LCNow)
    ## Select Land Cover tile clumping factors
    LC2006['Clumping_Index'] = (('y', 'x'), Clumping.sel(y=LC2006.y, x=LC2006.x,
                                                     method='nearest').values)
    LC2006['Clumping_Index'] = LC2006.Clumping_Index\
                                     .where(LC2006.Clumping_Index!=255) / 100
    ## Reduce to places with same land cover type now  
    noChange = LCNow['LC_Type5']==LC2006['LC_Type5']
    LC2006 = LC2006.where(noChange)
    self.ds['omega'] = LC2006.Clumping_Index.sel(x=self.ds.x, y=self.ds.y,
                                                        method='nearest')
    
  def getRTdict(self):
    ## Extract initial estimates of leaf reflectance and transmittance for PFT
    """
    ## Based on JULES parameters
    rtDict = {'Barren':                     {'r': {'vis': 0.18, 'nir': 0.01},
                                             't': {'vis': 0.01, 'nir': 0.01}},
              'Broadleaf Crop':             {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Cereal Crop':                {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.17, 'nir': 0.01}},
              'Deciduous Broadleaf Trees':  {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Deciduous Needleleaf Trees': {'r': {'vis': 0.07, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Evergreen Broadleaf Trees':  {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Evergreen Needleleaf Trees': {'r': {'vis': 0.07, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Grass':                      {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Permanent Snow and Ice':     {'r': {'vis': 0.75, 'nir': 0.01},
                                             't': {'vis': 0.01, 'nir': 0.01}},
              'Shrub':                      {'r': {'vis': 0.10, 'nir': 0.01},
                                             't': {'vis': 0.15, 'nir': 0.01}},
              'Urban and Built-up Lands':   {'r': {'vis': 0.18, 'nir': 0.01},
                                             't': {'vis': 0.01, 'nir': 0.01}},
              'Water Bodies':               {'r': {'vis': 0.12, 'nir': 0.01},
                                             't': {'vis': 0.01, 'nir': 0.01}}}
    """
    ## Based on Majasalmi & Bright (2019)
    rtDict = {'Broadleaf Crop':             {'r': {'vis': 0.08, 'nir': 0.42}, ## Crop
                                             't': {'vis': 0.05, 'nir': 0.40}},
              'Cereal Crop':                {'r': {'vis': 0.08, 'nir': 0.42}, ## Crop
                                             't': {'vis': 0.05, 'nir': 0.40}},
              'Deciduous Broadleaf Trees':  {'r': {'vis': 0.09, 'nir': 0.40}, ## BDT boreal
                                             't': {'vis': 0.05, 'nir': 0.42}},
              'Deciduous Needleleaf Trees': {'r': {'vis': 0.08, 'nir': 0.39}, ## NDT boreal
                                             't': {'vis': 0.06, 'nir': 0.42}},
              'Evergreen Broadleaf Trees':  {'r': {'vis': 0.11, 'nir': 0.46}, ## BET temperate
                                             't': {'vis': 0.06, 'nir': 0.33}},
              'Evergreen Needleleaf Trees': {'r': {'vis': 0.08, 'nir': 0.41}, ## NET boreal
                                             't': {'vis': 0.06, 'nir': 0.33}},
              'Grass':                      {'r': {'vis': 0.08, 'nir': 0.42}, ## Crop
                                             't': {'vis': 0.04, 'nir': 0.40}},
              'Shrub':                      {'r': {'vis': 0.08, 'nir': 0.42}, ## Crop
                                             't': {'vis': 0.05, 'nir': 0.40}}}
    self.rtDict = rtDict
    
  def getLCdict(self, LandCover):
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}
  
if __name__=='__main__':
  site = 'sodankyla'
  rt = RTMOD('/exports/csce/datastore/geos/users/s1503751/MODIS/%s/MODISnc'
             % site, testing=False, als=False)
  rt.main()
  rt.ds.to_netcdf('/exports/csce/datastore/geos/users/s1503751/MODIS/%s/sellers.large.nc' % site)

  rt.ds['BSA_vis_diff'] = rt.ds.Sellers_BSA_vis - rt.ds.MODIS_BSA_vis
  rt.ds['BSA_nir_diff'] = rt.ds.Sellers_BSA_nir - rt.ds.MODIS_BSA_nir
  rt.ds['WSA_vis_diff'] = rt.ds.Sellers_WSA_vis - rt.ds.MODIS_WSA_vis
  rt.ds['WSA_nir_diff'] = rt.ds.Sellers_WSA_nir - rt.ds.MODIS_WSA_nir 
  zenStep = 1
  zenBin = np.arange(0, 91, zenStep)
  BSA_vis = np.empty(zenBin.shape[0]-1)
  WSA_vis = np.empty(zenBin.shape[0]-1)
  BSA_nir = np.empty(zenBin.shape[0]-1)
  WSA_nir = np.empty(zenBin.shape[0]-1)
  for i in range(len(zenBin)-1):
    ds = rt.ds.where((rt.ds.zen>=zenBin[i])&(rt.ds.zen<zenBin[i+1]))
    BSA_vis[i] = ds.BSA_vis_diff.mean(skipna=True)
    WSA_vis[i] = ds.WSA_vis_diff.mean(skipna=True)
    BSA_nir[i] = ds.BSA_nir_diff.mean(skipna=True)
    WSA_nir[i] = ds.WSA_nir_diff.mean(skipna=True)
  plt.plot(zenBin[:-1]+zenStep/2., BSA_vis, label='BSA vis', color=cmap(0.2), lw=3)
  plt.scatter(rt.ds.zen, rt.ds.BSA_vis_diff, color=cmap(0.2), alpha=0.01, s=1, edgecolor='none')
  plt.plot(zenBin[:-1]+zenStep/2., WSA_vis, label='WSA vis', color=cmap(0.4), lw=3)
  plt.scatter(rt.ds.zen, rt.ds.WSA_vis_diff, color=cmap(0.4), alpha=0.01, s=1, edgecolor='none')
  plt.plot(zenBin[:-1]+zenStep/2., BSA_nir, label='BSA nir', color=cmap(0.6), lw=3)
  plt.scatter(rt.ds.zen, rt.ds.BSA_nir_diff, color=cmap(0.6), alpha=0.01, s=1, edgecolor='none')
  plt.plot(zenBin[:-1]+zenStep/2., WSA_nir, label='WSA nir', color=cmap(0.8), lw=3)
  plt.scatter(rt.ds.zen, rt.ds.WSA_nir_diff, color=cmap(0.8), alpha=0.01, s=1, edgecolor='none')
  plt.legend(loc='best')
  plt.show()
