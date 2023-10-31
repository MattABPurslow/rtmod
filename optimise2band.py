import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
from pyproj import CRS, Transformer
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
  t.lai=LAI/omega
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
    ## Create dictionary to track fitting trajectory
    self.statTracker = {}

  def main(self):
    ## Create date-by-y-by-x arrays to hold fitted values
    self.getOutArrs()
    ## Fit Sellers for each PFT
    print('PFTs to fit:', ', '.join([self.lcDict[k] for k in self.pftList]))
    for pftVal in self.pftList[::-1]:
      self.fit(pftVal)
    print('All PFTs fitted')
    ## Add as variables in MODIS dataset
    self.addFitToDataset()

  def getOutArrs(self):
    ## Create empty arrays for fitted parameters
    self.SA_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)
    self.omega_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)
    self.r_vis_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)
    self.t_vis_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)
    self.r_nir_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)
    self.t_nir_fit = np.full(self.ds.MODIS_LAI.shape, np.nan)

  def createSellers(self):
    #create Sellers instance
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1

  def trySellers(self, params, i, j, k):
    print('new iteration @', datetime.datetime.now())
    #unpack omega
    #inefficient as will adjust omega values of pixels with missing data
    omega = params[len(i):-4].reshape(self.ds.omega.shape)[(j,k)]
    #proportion of diffuse radiation
    self.t.propDif=0
    albVISdir = self.runSellers(self.t, self.mu, self.LAI, params[:len(i)], omega, params[-4], params[-2])
    albNIRdir = self.runSellers(self.t, self.mu, self.LAI, params[:len(i)], omega, params[-3], params[-1])
    #proportion of diffuse radiation
    self.t.propDif=1
    albVISdif = self.runSellers(self.t, self.mu, self.LAI, params[:len(i)], omega, params[-4], params[-2])
    albNIRdif = self.runSellers(self.t, self.mu, self.LAI, params[:len(i)], omega, params[-3], params[-1])
    #array of absolute difference to MODIS albedo
    albVISdirdiff = albVISdir-self.BSA_vis
    albNIRdirdiff = albNIRdir-self.BSA_nir
    albVISdifdiff = albVISdir-self.WSA_vis
    albNIRdifdiff = albNIRdir-self.WSA_nir
    error = np.array([(albVISdirdiff**2)**0.5,
                      (albNIRdirdiff**2)**0.5,
                      (albVISdifdiff**2)**0.5,
                      (albNIRdifdiff**2)**0.5]).flatten()
    stats = {'vis_dir_bias': np.mean(albVISdirdiff),
             'nir_dir_bias': np.mean(albNIRdirdiff),
             'vis_dif_bias': np.mean(albVISdifdiff),
             'nir_dif_bias': np.mean(albNIRdifdiff),
             'rmse': np.mean(error),
             'sa_mean': np.mean(params[:len(i)]),
             'omega_mean': np.mean(omega),
             'r_vis': params[-4],
             'r_nir': params[-3],
             't_vis': params[-2],
             't_nir': params[-1]}
    self.statTracker[self.pftName] = pd.concat([self.statTracker[self.pftName],
                                                pd.DataFrame(stats,
                                                  index=[self.nIter])])
    self.nIter += 1
    return error

  def unpackResult(self, result, dyx):
    ## Extract variables from params array
    self.SA_fit[dyx] = result.x[:np.shape(dyx)[1]]
    self.omega_fit[dyx] = result.x[np.shape(dyx)[1]:-4].reshape(self.ds.omega.shape)[(dyx[1], dyx[2])]
    self.r_vis_fit[dyx], self.r_nir_fit[dyx], self.t_vis_fit[dyx], self.t_nir_fit[dyx] = result.x[-4:]

  def getMODIS(self, testing=False, als=False):
    print('Loading MODIS data')
    ## Read in the processed MODIS data
    MODISlist = sorted(glob.glob(os.path.join(self.MODISdir,'*.nc')))
    self.ds = xr.Dataset()
    varNames = ['zen', 'MODIS_LAI', 'IGBP_LC', 'PFT_LC', 'lon', 'lat',
                'MODIS_Snow_Albedo', 'MODIS_BSA_vis', 'MODIS_WSA_vis',
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
  
  def fit(self, pftVal):
    pftName = self.lcDict[pftVal]
    print('Fitting sellers for', pftName)
    self.pftName = pftName
    self.statTracker[pftName] = pd.DataFrame()
    self.nIter = 0
    iPFT = np.argwhere(self.ds.PFT_LC.values.astype(int) == pftVal)
    iPFT_complex = iPFT[:,0]+iPFT[:,1]*1j
    idx_complex = self.idx[:,1]+self.idx[:,2]*1j
    ## Get indexes for good pixels within PFT
    data = np.array(self.idx[np.isin(idx_complex, iPFT_complex)]).T
    ## Check if any pixels:
    if data.shape[1] > 0:
      ## Add number of dates, latitudes and longitudes to data array
      r_vis, r_nir = self.rtDict[pftName]['r']['vis'], self.rtDict[pftName]['r']['nir']
      t_vis, t_nir = self.rtDict[pftName]['t']['vis'], self.rtDict[pftName]['t']['nir']
      ## Extract parameters to vary
      dyx = (data[0,:], data[1,:], data[2,:])
      SA = self.ds.MODIS_Snow_Albedo.values[dyx]
      SA[SA<1e-6] = 1e-6
      params = np.append(np.append(SA, self.ds.omega.fillna(1.0).values.flatten()),
                                   [r_vis, r_nir, t_vis, t_nir])
      ## Set bounds to prevent non-physical solutions
      bounds = (np.full(params.shape, 1e-7), np.full(params.shape, 1.))
      # Unpack data into self (so not done every iteration)
      self.BSA_vis = self.ds.MODIS_BSA_vis.values[(data[0,:],data[1,:],data[2,:])]
      self.WSA_vis = self.ds.MODIS_WSA_vis.values[(data[0,:],data[1,:],data[2,:])]
      self.BSA_nir = self.ds.MODIS_BSA_vis.values[(data[0,:],data[1,:],data[2,:])]
      self.WSA_nir = self.ds.MODIS_WSA_vis.values[(data[0,:],data[1,:],data[2,:])]  
      self.mu = self.ds.mu.values[(data[0,:],data[1,:],data[2,:])]
      self.LAI = self.ds.MODIS_LAI.values[(data[0,:],data[1,:],data[2,:])]
      ## Fit Sellers for PFT
      result = least_squares(self.trySellers, params, args=data, bounds=bounds)
      ## Save Sellers parameters
      self.unpackResult(result, dyx)
  
  def addFitToDataset(self):
    ## Add Sellers best-fit parameters to dataset
    self.ds['SA_fit'] = (('date', 'y', 'x'), self.SA_fit)
    self.ds['omega_fit'] = (('y', 'x'), np.nanmean(self.omega_fit, axis=0))
    self.ds['r_vis_fit'] = (('y', 'x'), np.nanmean(self.r_vis_fit, axis=0))
    self.ds['t_vis_fit'] = (('y', 'x'), np.nanmean(self.t_vis_fit, axis=0))
    self.ds['r_nir_fit'] = (('y', 'x'), np.nanmean(self.r_nir_fit, axis=0))
    self.ds['t_nir_fit'] = (('y', 'x'), np.nanmean(self.t_nir_fit, axis=0))

if __name__=='__main__':
  site = 'sodankyla'
  rt = RTMOD('/exports/csce/datastore/geos/users/s1503751/MODIS/%s/MODISnc1deg'
             % site, testing=False, als=True)
  rt.main()
  rt.ds.to_netcdf('/exports/csce/datastore/geos/users/s1503751/MODIS/%s/fitted.als.nc' % site)
  for pft in rt.pftList:
    rt.statTracker[pft].to_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/%s/%s.als.pkl' % (site, pft))
