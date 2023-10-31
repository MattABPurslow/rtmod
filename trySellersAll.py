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
from pySellersTwoStream.sellersTwoStream import twoStream


class RTMOD(object):
  def __init__(self, rootDir):
    self.rootDir = rootDir
    self.nCPUs = 10
    self.createSellers()
    self.sellers = np.vectorize(self.sellersUfunc)
    self.getRTdict()
    self.getAlbDict()
    self.getMODISdirs()
    self.getDOYs()
    self.getLandCover()
    self.getSummerLAI()
  
  def main(self):
    with Pool(self.nCPUs) as pool:
      pool.map(self.runDOY, self.doyList)
    #for doy in self.doyList:
    #  self.runDOY(doy)
  
  def runDOY(self, doy):
    self.getMODIS(doy)
    self.getSellers()
    self.saveOutput()
 
  def getMODIS(self, doy):
    self.doy = doy
    self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                 .replace(tzinfo=datetime.timezone.utc)
    self.getMODISfiles()
    #self.getLAI()
    self.getSnow()
    self.getAlbedo()
    self.getMu()
 
  def getSellers(self):
    #proportion of diffuse radiation
    self.t.propDif=0
    for df in [0, 1]:
      alb = ['BSA', 'WSA'][df]
      for wv in ['vis', 'nir']:
        print('running', alb, wv, '@', datetime.datetime.now(), end='\r')
        self.t.propDif = df
        k = 'Sellers_%s_%s'%(alb,wv)
        self.Albedods[k] = xr.apply_ufunc(self.sellers,
                                          self.Albedods.mu,
                                          self.LAIds.MODIS_LAI_eff,
                                          self.Snowds.MODIS_NDSI_Snow_Cover,
                                          self.LCds.PFT_LC,
                                          wv)
  
  def createSellers(self):
    # create Sellers instance
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1
  
  def sellersUfunc(self, mu, LAI, SC, pft, wv):
    check = np.isnan(mu)|np.isnan(LAI)|np.isnan(SC)|np.isnan(pft)
    if check:
      return np.nan
    else:
      pftName = self.lcDict[pft]
      #cosine of solar zenith:
      self.t.mu=mu
      #leaf area index
      self.t.lai=LAI
      #leaf reflectance & tranmisttance
      self.t.leaf_r = self.rtDict[pftName]['r'][wv]
      self.t.leaf_t = self.rtDict[pftName]['t'][wv]
      #soil reflectance
      self.t.soil_r=self.getGroundAlbedo(SC, wv)
      #do the radiative transfer calculation:
      Iup, _, _, _ = self.t.getFluxes()
      return Iup[0]

  def getMODISdirs(self):
    self.LCdir = os.path.join(self.rootDir, 'MCD12Q1')
    self.LAIdir = os.path.join(self.rootDir, 'MCD15A3H')
    self.Snowdir = os.path.join(self.rootDir, 'MOD10A1')
    self.Albedodir = os.path.join(self.rootDir, 'MCD43A3')

  def getMODISfiles(self):
    self.LAIfile = glob.glob(os.path.join(self.LAIdir,
                                          '*A%s*.hdf' % self.doy))[0]
    self.Snowfile = glob.glob(os.path.join(self.Snowdir,
                                           '*A%s*.hdf' % self.doy))[0]
    self.Albedofile = glob.glob(os.path.join(self.Albedodir,
                                             '*A%s*.hdf' % self.doy))[0]
  
  def getDOYs(self):
    LAIfiles = sorted(glob.glob(os.path.join(self.LAIdir, '*.hdf*')))
    Snowfiles = sorted(glob.glob(os.path.join(self.Snowdir, '*.hdf*')))
    Albedofiles = sorted(glob.glob(os.path.join(self.Albedodir, '*.hdf*')))
    LAIdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in LAIfiles]
    Snowdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Snowfiles]
    Albedodoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Albedofiles]
    self.doyList = [d for d in LAIdoy if (d in Snowdoy) & (d in Albedodoy)]
    print(len(self.doyList), 'dates from', min(self.doyList), 'to',
              max(self.doyList), end='\r')
   
  def getLandCover(self):
    print('reading land cover @', datetime.datetime.now(), end='\r')
    LCfile = sorted(glob.glob(os.path.join(self.LCdir,'*.hdf')))[-1]
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("4326"),
                                 always_xy=True)
    LandCover = rx.open_rasterio(LCfile).sel(band=1)
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}
    LandCover = LandCover.rename({'LC_Type1':'IGBP_LC','LC_Type5':'PFT_LC'})
    X, Y = np.meshgrid(LandCover.x, LandCover.y)
    lon, lat = trans.transform(X, Y)
    LandCover['lon'] = (('y', 'x'), lon)
    LandCover['lat'] = (('y', 'x'), lat)
    LandCover = LandCover.drop(['band', 'spatial_ref'])
    LandCover.attrs.clear()
    LandCover = LandCover.drop([k for k in list(LandCover.keys())
                                if k not in ['IGBP_LC', 'PFT_LC', 'ForestMask',
                                             'lon', 'lat']])
    LandCover['IGBP_LC'] = (('y','x'), LandCover.IGBP_LC.values.astype(np.int8))
    LandCover['PFT_LC'] = (('y','x'), LandCover.PFT_LC.values.astype(np.int8))
    self.LCds = LandCover
    self.getClumping()
    nanMask = np.isnan(self.LCds.PFT_LC.values)==False
    self.pftList = np.sort(np.unique(self.LCds.PFT_LC.values[nanMask])\
                                                     .astype(int))
    self.pftList = self.pftList[self.pftList > 0]
    self.pftList = np.array([k for k in self.pftList
                             if (self.lcDict[k] in self.rtDict.keys())])
    self.LCds['PFT_LC'] = self.LCds.PFT_LC.where(np.isin(self.LCds.PFT_LC,
                                                         self.pftList))

  def getClumping(self):
    print('retrieving he et al (2012) clumping index @', datetime.datetime.now(), end='\r')
    Clumping = rx.open_rasterio(os.path.join(self.rootDir,
                                '../global_clumping_index.tif')).sel(band=1)
    LandFile2006 = sorted(glob.glob(os.path.join(self.LCdir, '*.hdf')))[0]
    LC2006 = rx.open_rasterio(LandFile2006).sel(band=1)
    ## Select Land Cover tile clumping factors
    LC2006['Clumping_Index'] = (('y', 'x'), Clumping.sel(y=LC2006.y, x=LC2006.x,
                                                     method='nearest').values)
    LC2006['Clumping_Index'] = LC2006.Clumping_Index\
                                     .where(LC2006.Clumping_Index!=255) / 100
    ## Reduce to places with same land cover type now  
    noChange = self.LCds['PFT_LC']==LC2006['LC_Type5']
    LC2006 = LC2006.where(noChange)
    self.LCds['omega'] = LC2006.Clumping_Index.sel(x=self.LCds.x,
                                                   y=self.LCds.y,
                                                   method='nearest')

  def getMu(self):
    getNoon = np.vectorize(lambda lon: datetime.timedelta(hours=12.+(lon/15.)))
    print('getting noon @', datetime.datetime.now(), end='\r')
    utcNoon = self.date + getNoon(self.LCds.lon.values)
    print('getting zenith @', datetime.datetime.now(), end='\r')
    mask = np.isnan(self.LCds.PFT_LC.values)==False
    lat = self.LCds.lat.where(mask).values
    lon = self.LCds.lon.where(mask).values
    zen = pvlib.solarposition.get_solarposition(utcNoon.flatten(),
                                                lat.flatten(),
                                                lon.flatten(),
                                                0).zenith
    self.Albedods['zen'] = (('date', 'y', 'x'), zen.values.reshape(self.Albedods.MODIS_BSA_vis.shape))
    print('calculating mu @', datetime.datetime.now(), end='\r')
    self.Albedods['mu'] = np.cos(np.deg2rad(self.Albedods.zen))
  
  def getSummerLAI(self):
    summerDOYs = [doy for doy in self.doyList if np.isin(datetime.datetime.strptime(doy, '%Y%j').month, [6,7,8])]
    LAI = np.full((len(summerDOYs), self.LCds.y.shape[0], self.LCds.x.shape[0]), np.nan)
    checkRTMethodUsed = np.vectorize(lambda i: bin(int(i))[-3:-1]=='00')
    for i in range(len(summerDOYs)):
      self.doy = summerDOYs[i]
      print('Loading LAI', self.doy, '@', datetime.datetime.now(), end='\r')
      self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                   .replace(tzinfo=datetime.timezone.utc)
      self.getMODISfiles()
      dsi = rx.open_rasterio(self.LAIfile)
      LAIQC = checkRTMethodUsed(dsi.FparLai_QC.values)
      LAIi = (dsi.Lai_500m.values - dsi.Lai_500m.add_offset) * dsi.Lai_500m.scale_factor
      LAIi[LAIQC==False] = np.nan
      LAIi[LAIi>10.] = np.nan
      LAI[i,:,:] = LAIi
    ds = xr.Dataset()
    ds.coords['x'] = self.LCds.x
    ds.coords['y'] = self.LCds.y
    ds['MODIS_LAI'] = (('y', 'x'), np.nanmean(LAI, axis=0))
    ds['MODIS_LAI_Sigma'] = (('y', 'x'), np.nanstd(LAI, axis=0))
    ds['MODIS_LAI'] = ds.MODIS_LAI.where(ds.MODIS_LAI<=10.)
    ds['MODIS_LAI_eff'] = ds.MODIS_LAI * self.LCds.omega
    ds.attrs.clear()
    self.LAIds = ds
 
  def getLAI(self):
    print('reading LAI @', datetime.datetime.now(), end='\r')
    LAI = rx.open_rasterio(self.LAIfile).sel(band=1)
    LAI = LAI.rename({'Lai_500m':'MODIS_LAI',
                      'FparLai_QC':'MODIS_LAI_QC',
                      'LaiStdDev_500m':'MODIS_LAI_Sigma'})
    LAI = LAI.drop(['Fpar_500m', 'FparExtra_QC', 'FparStdDev_500m'])
    LAI.coords['date'] = self.date
    LAI = LAI.expand_dims(dim={'date':1})
    checkRTMethodUsed = np.vectorize(lambda i: bin(i)[-3:-1]=='00')
    LAI['MODIS_LAI_QC'] = (('date', 'y', 'x'),
                           checkRTMethodUsed(LAI.MODIS_LAI_QC))
    LAI['MODIS_LAI'] = (LAI.MODIS_LAI-LAI.MODIS_LAI.add_offset) * \
                       LAI.MODIS_LAI.scale_factor
    LAI['MODIS_LAI_Sigma'] = (LAI.MODIS_LAI_Sigma - \
                              LAI.MODIS_LAI_Sigma.add_offset) * \
                              LAI.MODIS_LAI_Sigma
    LAI['MODIS_LAI'] = LAI.MODIS_LAI.where(LAI.MODIS_LAI<=10.)
    LAI['MODIS_LAI_Sigma'] = LAI.MODIS_LAI_Sigma.where(LAI.MODIS_LAI_Sigma<=10.)
    LAI['MODIS_LAI_eff'] = LAI.MODIS_LAI * self.LCds.omega
    ## Add DataArrays to list
    LAI.attrs.clear()
    self.LAIds = LAI

  def getSnow(self):
    print('reading snow @', datetime.datetime.now(), end='\r')
    snow = rx.open_rasterio(self.Snowfile).sel(band=1)
    snow = snow.rename({'NDSI_Snow_Cover':'MODIS_NDSI_Snow_Cover',
                        'Snow_Albedo_Daily_Tile':'MODIS_Snow_Albedo',
                        'NDSI_Snow_Cover_Basic_QA':'MODIS_Snow_Cover_QC'})
    snow = snow.drop(['NDSI_Snow_Cover_Algorithm_Flags_QA', 'NDSI',
                      'orbit_pnt', 'granule_pnt'])
    snow.coords['date'] = self.date
    snow = snow.expand_dims(dim={'date':1})
    snow['MODIS_Snow_Cover_QC'] = snow.MODIS_Snow_Cover_QC <= 2
    snow['MODIS_NDSI_Snow_Cover'] = snow['MODIS_NDSI_Snow_Cover']\
                             .where(snow['MODIS_NDSI_Snow_Cover'] <= 100) / 100.
    snow['MODIS_Snow_Albedo'] = snow['MODIS_Snow_Albedo']\
		              .where(snow['MODIS_Snow_Albedo'] <= 100) / 100.
    snow.attrs.clear()
    self.Snowds = snow
 
  def getAlbedo(self):
    print('reading albedo @', datetime.datetime.now(), end='\r') 
    Albedo = rx.open_rasterio(self.Albedofile).sel(band=1)
    Albedo = Albedo.rename({'Albedo_BSA_shortwave': 'MODIS_BSA_shortwave',
                            'Albedo_BSA_nir': 'MODIS_BSA_nir',
                            'Albedo_BSA_vis': 'MODIS_BSA_vis',
                            'Albedo_WSA_shortwave': 'MODIS_WSA_shortwave',
                            'Albedo_WSA_nir': 'MODIS_WSA_nir',
                            'Albedo_WSA_vis': 'MODIS_WSA_vis',
                            'BRDF_Albedo_Band_Mandatory_Quality_shortwave':
                            'MODIS_BRDF_shortwave_QC',
                            'BRDF_Albedo_Band_Mandatory_Quality_vis':
                            'MODIS_BRDF_VIS_QC',
                            'BRDF_Albedo_Band_Mandatory_Quality_nir':
                            'MODIS_BRDF_NIR_QC'})
    Albedo = Albedo.drop(['BRDF_Albedo_Band_Mandatory_Quality_Band1',
                          'Albedo_BSA_Band1', 'Albedo_BSA_Band2',
                          'Albedo_BSA_Band3', 'Albedo_BSA_Band4',
                          'Albedo_BSA_Band5', 'Albedo_BSA_Band6',
                          'Albedo_BSA_Band7',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band2',
                          'Albedo_WSA_Band1', 'Albedo_WSA_Band2',
                          'Albedo_WSA_Band3', 'Albedo_WSA_Band4',
                          'Albedo_WSA_Band5', 'Albedo_WSA_Band6',
                          'Albedo_WSA_Band7',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band3',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band4',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band5',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band6',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band7'])
    Albedo.coords['date'] = self.date
    Albedo = Albedo.expand_dims(dim={'date':1})
    # Check Full BRDF inversion used
    Albedo['MODIS_BRDF_shortwave_QC'] = Albedo.MODIS_BRDF_shortwave_QC==0
    Albedo['MODIS_BRDF_VIS_QC'] = Albedo.MODIS_BRDF_VIS_QC==0
    Albedo['MODIS_BRDF_NIR_QC'] = Albedo.MODIS_BRDF_NIR_QC==0
    for k in list(Albedo.keys()):
      if k[-2:] != 'QC':
        Albedo[k] = (Albedo[k]-Albedo[k].add_offset)*Albedo[k].scale_factor
        Albedo[k] = Albedo[k].where(Albedo[k] <= 1.0)
    Albedo.attrs.clear()
    self.Albedods = Albedo

  def getRTdict(self):
    ## Extract initial estimates of leaf reflectance and transmittance for PFT
    ## Based on Majasalmi & Bright (2019)
    rtDict ={'Broadleaf Crop':             {'r': {'vis': 0.08, 'nir': 0.42},
                                            't': {'vis': 0.05, 'nir': 0.40}},
             'Cereal Crop':                {'r': {'vis': 0.08, 'nir': 0.42},
                                            't': {'vis': 0.05, 'nir': 0.40}},
             'Deciduous Broadleaf Trees':  {'r': {'vis': 0.09, 'nir': 0.40},
                                            't': {'vis': 0.05, 'nir': 0.42}},
             'Deciduous Needleleaf Trees': {'r': {'vis': 0.08, 'nir': 0.39},
                                            't': {'vis': 0.06, 'nir': 0.42}},
             'Evergreen Broadleaf Trees':  {'r': {'vis': 0.11, 'nir': 0.46},
                                            't': {'vis': 0.06, 'nir': 0.33}},
             'Evergreen Needleleaf Trees': {'r': {'vis': 0.08, 'nir': 0.41},
                                            't': {'vis': 0.06, 'nir': 0.33}},
             'Grass':                      {'r': {'vis': 0.08, 'nir': 0.42},
                                            't': {'vis': 0.04, 'nir': 0.40}},
             'Shrub':                      {'r': {'vis': 0.08, 'nir': 0.42},
                                            't': {'vis': 0.05, 'nir': 0.40}}}
    self.rtDict = rtDict
 
  def getGroundAlbedo(self, snowCover, wv):
    snowAlbedo = self.albDict['sa'][wv]
    soilAlbedo = self.albDict['ga'][wv]
    return (snowCover*snowAlbedo) + ((1-snowCover)*soilAlbedo)

  def getAlbDict(self):
    self.albDict = {'sa': {'vis':0.6, 'nir':0.6},
                    'ga': {'vis':0.15, 'nir':0.83}}
 
  def saveOutput(self):
    outfile = os.path.join(self.rootDir, 'nc/%s.%s.nc' % ('sodankyla', self.doy))
    #self.LAIds['date'] = self.LAIds.date.astype(np.datetime64)
    self.Snowds['date'] = self.Snowds.date.astype(np.datetime64)
    self.Albedods['date'] = self.Albedods.date.astype(np.datetime64)
    lcVars = ['PFT_LC', 'lon', 'lat', 'omega']
    snowVars = ['MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover']
    laiVars = ['MODIS_LAI', 'MODIS_LAI_Sigma', 'MODIS_LAI_eff']
    albedoVars = ['zen', 'mu',
                  'MODIS_BSA_vis', 'MODIS_BSA_nir',
                  'MODIS_WSA_vis', 'MODIS_WSA_nir',
                  'Sellers_BSA_vis', 'Sellers_BSA_nir',
                  'Sellers_WSA_vis', 'Sellers_WSA_nir']
    LCds = (self.LCds[lcVars]*100).astype(np.int16)
    LAIds = (self.LAIds[laiVars]*100).astype(np.int16)
    Snowds = (self.Snowds[snowVars]*100).astype(np.int16)
    Albedods = (self.Albedods[albedoVars]*100).astype(np.int16)
    ds = xr.merge([LCds, LAIds, Snowds, Albedods]).to_netcdf(outfile)

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  rtmod = RTMOD(rootDir)
  rtmod.main()
