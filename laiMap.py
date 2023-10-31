import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rx
import xarray as xr
from pyproj import CRS, Proj, Transformer
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap

from multiprocessing import Pool
from functools import partial
from pySellersTwoStream.sellersTwoStream import twoStream


class RTMOD(object):
  def __init__(self, rootDir, ALSfile, lidar, threshold):
    """
    Initialise by loading data with time-invariate inputs
    """
    self.rootDir = rootDir
    self.ALSfile = ALSfile
    self.lidar = lidar
    self.alsThreshold = threshold
    self.nCPUs = 12
    self.k = 0.5 # As in JULES
    self.crownRatio = 12.9/2.7 # for boreal & montane (Jucker et al 2022)
    self.wv = ['vis', 'nir', 'shortwave', 'vis', 'nir', 'shortwave']
    self.df = ['BSA', 'BSA', 'BSA', 'WSA', 'WSA', 'WSA']
    self.createSellers()
    self.sellers = np.vectorize(self.sellersUfunc)
    self.getRTdict()
    self.getAlbDict()
    self.getMODISdirs()
    self.getDOYs()
    self.getLandCover()
    self.getSummerLAI()

  def main(self):
    """
    Run Sellers and SPARTACUS for each day in doyList
    """
    with Pool(self.nCPUs) as pool:
      pool.map(self.runDOY, self.doyList)
    #for doy in self.doyList:
    #  print(doy, end='\r')
    #  self.runDOY(doy)

  def runDOY(self, doy):
    """
    Run Sellers and SPARTACUS for a given day
    """
    self.getMODIS(doy)
    self.mergeMODIS()
    self.createSPARTACUSinput()
    if self.out.column.shape[0]>0:
      self.getMu()
      if self.out.column.shape[0]>0:
        self.getSellers()
        self.runSPARTACUS()

  def getMODISdirs(self):
    """
    Identify directories containing MODIS data
    """
    self.LCdir = os.path.join(self.rootDir, 'MCD12Q1')
    self.LAIdir = os.path.join(self.rootDir, 'MCD15A3H')
    self.Snowdir = os.path.join(self.rootDir, 'MOD10A1')
    self.Albedodir = os.path.join(self.rootDir, 'MCD43A3')

  def getMODIS(self, doy):
    """
    Read MODIS data for given day
    """
    self.doy = doy
    self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                 .replace(tzinfo=datetime.timezone.utc)
    self.getMODISfiles()
    self.getThisLAI()
    self.getSnow()
    self.getAlbedo()
  
  def getSellers(self):
    """
    Run Sellers for all pixels in out dataset
    """
    #proportion of diffuse radiation
    for alb, wv in zip(self.df, self.wv):
      if alb=='BSA':
        self.t.propDif = 0.
      else:
          self.t.propDif = 1.
      print('running', alb, wv, '@', datetime.datetime.now(), end='\r')
      k = 'Sellers_%s_%s'%(alb,wv)
      self.out[k] = xr.apply_ufunc(self.sellers,
                                   self.out.cos_solar_zenith_angle,
                                   self.out.MODIS_LAI_eff,
                                   self.out.ground_sw_albedo,
                                   self.out.PFT_LC,
                                   wv)
  
  def createSellers(self):
    """
    Create single layer Sellers instance
    """
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1

  def sellersUfunc(self, mu, LAI, SA, pft, wv):
    """
    Run Sellers for given inputs
    mu: cosine of solar zenith angle
    LAI: effective LAI
    SA: subcanopy albedo
    pft: Plant Functional Type
    wv: waveband
    """
    check = np.isnan(mu)|np.isnan(LAI)|(LAI<0.01)|np.isnan(SA)|np.isnan(pft)|(SA==0)
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
      self.t.soil_r=SA
      #do the radiative transfer calculation:
      Iup, _, _, _ = self.t.getFluxes()
      return Iup[0]

  def getMODISfiles(self):
    """
    Get lists of MODIS files
    """
    self.LAIfile = glob.glob(os.path.join(self.LAIdir,
                                          '*A%s*.hdf' % self.doy))[0]
    self.Snowfile = glob.glob(os.path.join(self.Snowdir,
                                           '*A%s*.hdf' % self.doy))[0]
    self.Albedofile = glob.glob(os.path.join(self.Albedodir,
                                             '*A%s*.hdf' % self.doy))[0]

  def getDOYs(self):
    """
    Idenitify dates with comprehensive MODIS data
    """
    LAIfiles = sorted(glob.glob(os.path.join(self.LAIdir, '*.hdf*')))
    Snowfiles = sorted(glob.glob(os.path.join(self.Snowdir, '*.hdf*')))
    Albedofiles = sorted(glob.glob(os.path.join(self.Albedodir, '*.hdf*')))
    LAIdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in LAIfiles]
    Snowdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Snowfiles]
    Albedodoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Albedofiles]
    self.doyList = [d for d in LAIdoy if (d in Snowdoy) & (d in Albedodoy)]
    #self.doyList = [d for d in self.doyList if d[:4] in ['2020', '2021', '2022']]
    print(len(self.doyList), 'dates from', min(self.doyList), 'to',
              max(self.doyList), end='\r')

  def getLandCover(self):
    """
    Read MODIS land cover classification (MCD12Q1)
    """
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
    self.getALS()
    nanMask = np.isnan(self.LCds.PFT_LC.values)==False
    self.pftList = np.sort(np.unique(self.LCds.PFT_LC.values[nanMask])\
                                                     .astype(int))
    self.pftList = self.pftList[self.pftList > 0]
    self.pftList = np.array([k for k in self.pftList
                             if (self.lcDict[k] in self.rtDict.keys())])
    self.LCds['PFT_LC'] = self.LCds.PFT_LC.where(np.isin(self.LCds.PFT_LC,
                                                         self.pftList))

  def getClumping(self):
    """
    Retrieve MODIS-derived clumping index
    """
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
    """
    Calculate solar position
    """
    getNoon = np.vectorize(lambda lon: datetime.timedelta(hours=12.+(lon/15.)))
    print('getting noon @', datetime.datetime.now(), end='\r')
    utcNoon = self.date + getNoon(self.out.lon.values)
    print('getting zenith @', datetime.datetime.now(), end='\r')
    lat = self.out.lat.values
    lon = self.out.lon.values
    zen = pvlib.solarposition.get_solarposition(utcNoon.flatten(),
                                                lat.flatten(),
                                                lon.flatten(),
                                                0).zenith
    self.out['zen'] = (('column'), zen)
    print('calculating mu @', datetime.datetime.now(), end='\r')
    self.out['cos_solar_zenith_angle'] = np.cos(np.deg2rad(self.out.zen))
    self.out['cos_solar_zenith_angle'] = self.out.cos_solar_zenith_angle.where((self.out.cos_solar_zenith_angle>1.e-3))
    self.out = self.out.dropna(dim='column')

  def getSummerLAI(self):
    """
    Calculate average summer LAI
    """
    summerDOYs = [doy for doy in self.doyList if np.isin(datetime.datetime.strptime(doy, '%Y%j').month, [6,7,8])]
    LAI = np.full((len(summerDOYs), self.LCds.y.shape[0], self.LCds.x.shape[0]), np.nan)
    checkRTMethodUsed = np.vectorize(lambda i: bin(int(i))[-3:-1]=='00')
    for i in range(len(summerDOYs)):
      self.doy = summerDOYs[i]
      print('Loading LAI', self.doy, '@', datetime.datetime.now(), end='\r')
      self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                   .replace(tzinfo=datetime.timezone.utc)
      self.getMODISfiles()
      dsi = rx.open_rasterio(self.LAIfile).sel(y=self.ySlice, x=self.xSlice)
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
    self.LAIAllds = ds

  def getThisSummerLAI(self):
    """
    Calculate average summer LAI for each year
    """
    years = np.sort(np.unique([datetime.datetime.strptime(doy, '%Y%j').year for doy in self.doyList]))
    LAI = {}
    LAIstd = {}
    for year in years:
      summerDOYs = [doy for doy in self.doyList if np.isin(datetime.datetime.strptime(doy, '%Y%j').month, [6,7,8])&(datetime.datetime.strptime(doy, '%Y%j').year==year)]
      LAIyear = np.full((len(summerDOYs), self.LCds.y.shape[0], self.LCds.x.shape[0]), np.nan)
      checkRTMethodUsed = np.vectorize(lambda i: bin(int(i))[-3:-1]=='00')
      for i in range(len(summerDOYs)):
        self.doy = summerDOYs[i]
        print('Loading LAI', self.doy, '@', datetime.datetime.now(), end='\r')
        self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                     .replace(tzinfo=datetime.timezone.utc)
        self.getMODISfiles()
        dsi = rx.open_rasterio(self.LAIfile).sel(y=self.ySlice, x=self.xSlice)
        LAIQC = checkRTMethodUsed(dsi.FparLai_QC.values)
        LAIi = (dsi.Lai_500m.values - dsi.Lai_500m.add_offset) * dsi.Lai_500m.scale_factor
        LAIi[LAIQC==False] = np.nan
        LAIi[LAIi>10.] = np.nan
        LAIyear[i,:,:] = LAIi
      LAI[year] = np.nanmean(LAIyear, axis=0)
      LAIstd[year] = np.nanstd(LAIyear, axis=0)
    ds = xr.Dataset()
    ds.coords['x'] = self.LCds.x
    ds.coords['y'] = self.LCds.y
    ds.coords['year'] = years
    ds['MODIS_LAI'] = (('year', 'y', 'x'), [LAI[y] for y in years])
    ds['MODIS_LAI_Sigma'] = (('year', 'y', 'x'), [LAIstd[y] for y in years])
    ds['MODIS_LAI'] = ds.MODIS_LAI.where(ds.MODIS_LAI<=10.)
    ds['MODIS_LAI_eff'] = ds.MODIS_LAI * self.LCds.omega
    ds.attrs.clear()
    self.LAIyeards = ds
  
  def getThisLAI(self):
    self.LAIds = self.LAIyeards.sel(year=int(self.doy[:4])).drop('year')

  def getLAI(self):
    """
    Read MODIS LAI (MCD15A3H)
    """
    print('reading LAI @', datetime.datetime.now(), end='\r')
    LAI = rx.open_rasterio(self.LAIfile).sel(band=1, y=self.ySlice, x=self.xSlice)
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
    """
    Read MODIS Snow Cover (MOD10A1)
    """
    print('reading snow @', datetime.datetime.now(), end='\r')
    snow = rx.open_rasterio(self.Snowfile).sel(band=1, y=self.ySlice, x=self.xSlice)
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
    """
    Read MODIS albedo (MCD43A3)
    """
    print('reading albedo @', datetime.datetime.now(), end='\r')
    Albedo = rx.open_rasterio(self.Albedofile).sel(band=1, y=self.ySlice, x=self.xSlice)
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
    """
    Extract initial estimates of leaf reflectance and transmittance for PFT
    Based on Majasalmi & Bright (2019)
    """
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
    ## Calculate weight average for broadband
    for k in rtDict.keys():
      for k2 in rtDict[k].keys():
        rtDict[k][k2]['shortwave'] = np.average([rtDict[k][k2]['vis'],
                                                 rtDict[k][k2]['nir']],
                                                 weights=[(0.7-0.3), (5.0-0.7)])
    self.rtDict = rtDict

  def getGroundAlbedo(self, snowCover, wv):
    """
    Estimate subcanopy albedo based on snow cover
    """
    snowAlbedo = self.albDict['sa'][wv]
    soilAlbedo = self.albDict['ga'][wv]
    return (snowCover*snowAlbedo) + ((1-snowCover)*soilAlbedo)
  
  def loadSodankylaAlbedo(self):
    """
    Get Sodankyla albedo per https://litdb.fmi.fi/ioa0008_data.php. Interpolate rolling monthly average to give daily estimate.
    Wavelength: 285nm - 2800nm
    """
    df = pd.read_csv('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/SodankylaAlbedoTimeseries_IOA0008_2012-01-01_2023-08-31.txt', header=1, names=['station', 'datetime', 'downwelling', 'upwelling'])
    df['downwelling'] = df.downwelling.where(df.downwelling>0.)
    df['upwelling'] = df.upwelling.where(df.upwelling>0.)
    df['albedo'] = df.upwelling/df.downwelling
    df['albedo'] = df.albedo.where((df.albedo>0.)&(df.albedo<1.))
    df['datetime'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S+00') for d in df.datetime]
    df['date'] = [datetime.datetime(d.year,d.month,d.day) for d in df.datetime]
    self.sodankyla = pd.DataFrame()
    self.sodankyla['albedo'] = df.groupby('date').albedo.mean().rolling(window=30, center=True, min_periods=5).mean().interpolate('cubicspline')
    self.sodankyla['doy'] = [d.strftime('%Y%j') for d in self.sodankyla.index]
    self.sodankyla.set_index('doy', inplace=True)
    
  def getAlbDict(self):
      """
      Get subcanopy albedo estimates for snow-covered and snow-free surfaces
      """
      self.albDict = {'sa': {'vis':0.6, 'nir':0.6},
                      'ga': {'vis':0.15, 'nir':0.83}}
      for k in self.albDict.keys():
        self.albDict[k]['shortwave'] = np.average([self.albDict[k]['vis'],
                                                   self.albDict[k]['nir']], 
                                                   weights=[(0.7-0.3), (5.0-0.7)])
  def mergeMODIS(self):
    """
    Combine MODIS dataset into single dataset
    """
    if 'date' in self.LAIds.coords:
      self.LAIds.coords['date'] = self.LAIds.date.astype(np.datetime64)
    self.Snowds.coords['date'] = self.Snowds.date.astype(np.datetime64)
    self.Albedods.coords['date'] = self.Albedods.date.astype(np.datetime64)
    lcVars = ['PFT_LC', 'lon', 'lat', 'omega', 'chm', 'cv']
    snowVars = ['MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover']
    laiVars = ['MODIS_LAI', 'MODIS_LAI_Sigma', 'MODIS_LAI_eff']
    albedoVars = [#'zen', 'cos_solar_zenith_angle',
                  'MODIS_BSA_vis', 'MODIS_BSA_nir', 'MODIS_BSA_shortwave',
                  'MODIS_WSA_vis', 'MODIS_WSA_nir', 'MODIS_WSA_shortwave']#,
                  #'Sellers_BSA_vis', 'Sellers_BSA_nir', 'Sellers_BSA_shortwave',
                  #'Sellers_WSA_vis', 'Sellers_WSA_nir', 'Sellers_WSA_shortwave']
    self.ds = xr.merge([self.LCds[lcVars],
                        self.LAIds[laiVars],
                        self.Snowds[snowVars],
                        self.Albedods[albedoVars]]).isel(date=0).copy()

  def getALS(self):
    """
    Get ALS-derived canopy height and cover
    """
    print('Retrieving ALS canopy height and cover @', datetime.datetime.now(), end='\r')
    albLon, albLat = 26.63319, 67.36198 # Location of Sodankyla IOA albedo measurement
    self.aoiLon = np.array([albLon-.25, albLon-.25, albLon+.25, albLon+.25])
    self.aoiLat = np.array([albLat-.25, albLat+.25, albLat+.25, albLat-.25])
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(CRS.from_epsg("4326"), MODProj.crs, always_xy=True)
    aoiX, aoiY = trans.transform(self.aoiLon, self.aoiLat)
    if self.lidar=='als.sodankyla':
      cv = rx.open_rasterio(self.ALSfile.replace('als','finland.canopy_cover.modis')).sel(band=1)
      chm = rx.open_rasterio(self.ALSfile.replace('als','finland.ndsm.modis.%s' % self.alsThreshold)).sel(band=1)
      self.LCds['chm'] = (('y', 'x'), chm.where((chm>0.)&(chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (cv.where((cv>1.e-3)&(cv<1.)).values.astype(np.float64)*100.))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='als.finland':
      cv = rx.open_rasterio(self.ALSfile.replace('als','finland.canopy_cover.modis')).sel(band=1)
      chm = rx.open_rasterio(self.ALSfile.replace('als','finland.ndsm.modis.%s' % self.alsThreshold)).sel(band=1)
      self.LCds['chm'] = (('y', 'x'), chm.where((chm>0.)&(chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (cv.where((cv>1.e-3)&(cv<1.)).values.astype(np.float64)*100.))
      self.xSlice = slice(-1e12, 1e12)
      self.ySlice = slice(1e12, -1e12)
    elif self.lidar=='icesat2.reclassified':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='icesat2.atl08':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.atl08.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='icesat2.atl08.finland':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.atl08.finland.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(-1e12, 1e12)
      self.ySlice = slice(1e12, -1e12)
    self.LCds = self.LCds.sel(y=self.ySlice, x=self.xSlice)

  def getLCdict(self, LandCover):
    """
    Get list of MODIS Plant Functional Types
    """
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}

  def getExtinction(self):
    """
    Calculate vegetation extinction coefficient
    """
    self.ds['omega'] = self.ds.omega.where((self.ds.omega > 0) & (self.ds.omega < 1))
    self.ds['veg_extinction'] = (('layer', 'y', 'x'), [(self.k*self.ds.MODIS_LAI)/self.ds.chm])

  def getFSD(self):
    """
    Define variability in LAI
    """
    self.ds['veg_fsd'] = (('layer', 'y', 'x'), [self.ds.MODIS_LAI_Sigma.where(self.ds.MODIS_LAI_Sigma>0)])

  def getSA(self, Nsw):
    """
    Retrieve subcanopy albedo
    """
    #self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [self.ds.MODIS_Snow_Albedo.where((self.ds.MODIS_Snow_Albedo>0.)&(self.ds.MODIS_Snow_Albedo<1.))]*Nsw)
    #gGA = np.vectorize(self.getGroundAlbedo)
    #self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [gGA(self.ds.MODIS_NDSI_Snow_Cover, wv) for wv in self.wv])
    self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), np.full((Nsw, self.ds.y.shape[0], self.ds.x.shape[0]), self.sodankyla.loc[self.doy].albedo))
    self.ds['ground_sw_albedo_direct'] = self.ds.ground_sw_albedo.copy()
  
  def getSSA(self):
    """
    Get vegetation single scattering albedo
    """
    ssa_vis = np.full(self.ds.PFT_LC.shape, np.nan)
    ssa_nir = np.full(self.ds.PFT_LC.shape, np.nan)
    ssa_short = np.full(self.ds.PFT_LC.shape, np.nan)
    PFTarr = np.unique(self.ds.PFT_LC)
    for PFT in PFTarr:
      if PFT in self.lcDict.keys():
        if self.lcDict[PFT] in self.rtDict.keys():
          mask = self.ds.PFT_LC==PFT
          ssa_vis[mask] = self.rtDict[self.lcDict[PFT]]['r']['vis'] + self.rtDict[self.lcDict[PFT]]['t']['vis']
          ssa_nir[mask] = self.rtDict[self.lcDict[PFT]]['r']['nir'] +  self.rtDict[self.lcDict[PFT]]['t']['nir']
          ssa_short[mask] = self.rtDict[self.lcDict[PFT]]['r']['shortwave'] +  self.rtDict[self.lcDict[PFT]]['t']['shortwave']
    self.ds['veg_sw_ssa'] = (('layer', 'sw', 'y', 'x'), [[ssa_vis, ssa_nir, ssa_short, ssa_vis, ssa_nir, ssa_short]])

  def createSPARTACUSinput(self):
    """
    Create dataset in format required for SPARTACUS runs
    """
    self.ds['height'] = (('layer_int','y','x'), [np.full(self.ds.chm.shape,0.),
                                                 self.ds.chm])
    self.ds['veg_fraction'] = (('layer', 'y', 'x'), [self.ds.cv])
    self.ds['veg_fraction'] = self.ds.veg_fraction.where(self.ds.veg_fraction>0.01)
    self.ds['veg_scale'] = (('layer', 'y', 'x'), [self.ds.chm / self.crownRatio])
    ## Default values
    Ny, Nx = self.ds.y.shape[0], self.ds.x.shape[0]
    Nsw = 6
    self.ds['surface_type'] = (('y', 'x'), np.full((Ny, Nx), int(1)))
    self.ds['nlayer'] = (('y', 'x'), np.full((Ny, Nx), int(1)))
    self.ds['veg_contact_fraction'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 0.))
    self.ds['building_fraction']  = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 0.))
    self.ds['building_scale'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 0.))
    self.ds['clear_air_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['veg_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['veg_air_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['air_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['ground_temperature'] = (('y', 'x'), np.full((Ny, Nx), 273.))
    self.ds['roof_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['wall_temperature'] = (('y', 'x', 'layer'), np.full((Ny, Nx, 1), 273.))
    self.ds['ground_lw_emissivity'] = (('y', 'x', 'lw'), np.full((Ny, Nx, 1), 1.))
    self.ds['veg_lw_ssa'] = (('y', 'x', 'layer', 'lw'), np.full((Ny, Nx, 1, 1), 1.))
    self.ds['roof_sw_albedo'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), 1.))
    self.ds['roof_sw_albedo_direct'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), 1.))
    self.ds['roof_lw_emissivity'] = (('y', 'x', 'lw'), np.full((Ny, Nx, 1), 1.))
    self.ds['wall_sw_albedo'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), 1.))
    self.ds['wall_sw_albedo_direct'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), 1.))
    self.ds['wall_lw_emissivity'] = (('y', 'x', 'lw'), np.full((Ny, Nx, 1), 1.))
    self.ds['sky_temperature'] = (('y', 'x'), np.full((Ny, Nx), 273.))
    self.ds['top_flux_dn_sw'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), 1.))
    self.ds['top_flux_dn_direct_sw'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), ([1.]*(Nsw//2))+([0.]*(Nsw//2))))
    self.getExtinction()
    self.getFSD()
    self.getSA(Nsw)
    self.getSSA()
    self.out = self.ds.stack(ind=('y', 'x'), create_index=False).copy()
    varList = list(self.out.variables.keys())
    for v in varList:
      Nd = len(self.out[v].dims)
      if Nd > 1:
        dims = sorted(self.out[v].dims)
        if Nd == 2:
          self.out[v] = self.out[v].transpose(dims[0], dims[1])
        elif Nd == 3:
          self.out[v] = self.out[v].transpose(dims[0], dims[1], dims[2])
        else:
          raise Exception('Too many dims on %s ' % v)
      else:
        pass
    self.spartVars = ['surface_type',
                      'nlayer',
                      'height',
                      'veg_fraction',
                      'veg_scale',
                      'veg_extinction',
                      'veg_fsd',
                      'veg_contact_fraction',
                      'building_fraction',
                      'building_scale',
                      'clear_air_temperature',
                      'veg_temperature',
                      'veg_air_temperature',
                      'air_temperature',
                      'ground_temperature',
                      'roof_temperature',
                      'wall_temperature',
                      'ground_sw_albedo',
                      'ground_sw_albedo_direct',
                      'ground_lw_emissivity',
                      'veg_sw_ssa',
                      'veg_lw_ssa',
                      'roof_sw_albedo',
                      'roof_sw_albedo_direct',
                      'roof_lw_emissivity',
                      'wall_sw_albedo',
                      'wall_sw_albedo_direct',
                      'wall_lw_emissivity',
                      'sky_temperature',
                      'top_flux_dn_sw',
                      'top_flux_dn_direct_sw']
    self.out['column'] = np.arange(self.out.ind.shape[0])
    self.spartVars.append('column')
    # Crop to 0.5° from albedo measurement
    if 'finland' not in self.lidar:
      self.out = self.out.where((self.out.lon>=self.aoiLon.min())&(self.out.lon<=self.aoiLon.max())&\
                                (self.out.lat>=self.aoiLat.min())&(self.out.lat<=self.aoiLat.max()))
    self.out = self.out.sel(ind=self.out[self.spartVars].column).dropna(dim='column')
    self.out['veg_fraction'] = self.out.veg_fraction / 100.
    self.out = self.out.transpose('column', 'layer', 'layer_int', 'lw', 'sw')

  def runSPARTACUS(self):
    """
    Save SPARTACUS input files and run
    """
    ## Remove any pixels with persistent missing data 
    self.out = self.out.dropna(dim='column')
    if self.out.column.shape[0]>0:
      for i in range(self.out.sw.shape[0]):
        inFile = os.path.join(self.rootDir, 'spartacusIn/sodankyla.%s.%s_%s.nc' % (self.doy, self.df[i], self.wv[i]))
        self.out.isel(sw=[i]).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')
        outFile = inFile.replace('spartacusIn', 'spartacusOut')
        command = ['spartacus_surface', 'config.nam', inFile, outFile]
        r=os.system(' '.join(command))

def readSPARTACUS(outFile):
  ds = xr.open_dataset(outFile.replace('spartacusOut', 'spartacusIn')).isel(layer=-1, layer_int=-1)
  labLoaded = []
  for df in ['BSA', 'WSA']:
    for wv in ['nir', 'vis', 'shortwave']:
      lab = '%s_%s' % (df, wv)
      if (os.path.exists(outFile.replace('BSA_nir', lab))):
        out = xr.open_dataset(outFile.replace('BSA_nir', lab)).isel(layer=-1, layer_interface=-1)
        ds['SPARTACUS_%s' % lab] = out.flux_up_layer_top_sw
        labLoaded.append(lab)
  ds['mask'] = (ds.MODIS_NDSI_Snow_Cover>0.) &\
               (ds.PFT_LC==1) &\
               (ds.omega>0.) &\
               (ds.MODIS_LAI>0.) &\
               (ds.zen<90.) &\
               (ds.MODIS_Snow_Albedo>0.)
  for lab in labLoaded:
      ds['mask'] = ds.mask &\
                   (ds['MODIS_%s' % lab] > 0.)&(ds['MODIS_%s' % lab]<1.)&\
                   (ds['Sellers_%s' % lab] > 0.)&(ds['Sellers_%s' % lab]<1.)&\
                   (ds['SPARTACUS_%s' % lab] > 0.)&(ds['SPARTACUS_%s' % lab]<1.)
  return ds.where(ds.mask).to_dataframe().reset_index().dropna()

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/als.tif'
  import argparse
  args = argparse.ArgumentParser()
  args.add_argument('-l', '--lidar', dest='lidar', type=str, default='als.sodankyla')
  args.add_argument('-t', '--threshold', dest='threshold', type=str, default='2m')
  args = args.parse_args()
  lidar = args.lidar
  threshold = args.threshold
  rtmod = RTMOD(rootDir, ALSfile, lidar, threshold)
  border = gpd.read_file('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ne_10m_admin_0_countries.shp')
  border = border[border.ADMIN=='Finland'].to_crs(CRS.from_epsg("32635"))
  als = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/finland.ndsm.modis.2m.tif').sel(band=1)
  als = als.where((als>2)&(als<50))
  alsMask = np.isnan(als)==False
  idxMask = np.argwhere(alsMask.values)
  maxIDX = np.full(als.y.shape[0], -1, dtype=int)
  minIDX = np.full(als.y.shape[0], -1, dtype=int)
  for i in range(als.y.shape[0]):
    if i in idxMask[:,0]:
      minIDX[i] = np.min(idxMask[idxMask[:,0]==i][:,1])
      maxIDX[i] = np.max(idxMask[idxMask[:,0]==i][:,1])
    elif i != 0:
      iBack = i
      iFor = i
      southMin, southMax, sDist = 0,0,1e12
      northMin, northMax, nDist = 0,0,1e12
      while iBack > 0:
        iBack -= 1
        if iBack in idxMask[:,0]:
          northMin = np.min(idxMask[idxMask[:,0]==iBack][:,1])
          northMax = np.min(idxMask[idxMask[:,0]==iBack][:,1])
          nDist = i-iBack
          iBack = 0
      while iFor < als.y.shape[0]:
        iFor += 1
        if iFor in idxMask[:,0]:
          southMin = np.min(idxMask[idxMask[:,0]==iFor][:,1])
          southMax = np.max(idxMask[idxMask[:,0]==iFor][:,1])
          sDist = iFor-i
          iFor = als.y.shape[0]
      minIDX[i] = int(np.round(np.average([northMin, southMin], weights=[1./nDist, 1./sDist])))
      maxIDX[i] = int(np.round(np.average([northMax, southMax], weights=[1./nDist, 1./sDist])))
    else:
      pass
  
  iceMask = np.full(alsMask.shape, False)
  for i in range(als.y.shape[0]):
    if (minIDX[i]>-1) & (maxIDX[i]>-1):
      iceMask[i][minIDX[i]:maxIDX[i]+1] = True
  
  import matplotlib as mpl
  mpl.rcParams['axes.facecolor'] = 'k'
  fig = plt.figure(figsize=(6,8))
  ax = fig.add_subplot(1,1,1)
  MODProj = Proj('+proj=sinu +R=6371007.181')
  trans = Transformer.from_proj(MODProj, CRS.from_epsg("32635").to_proj4(),
                                always_xy=True)
  X, Y = np.meshgrid(rtmod.LCds.x, rtmod.LCds.y)
  E, N = trans.transform(X, Y)
  chm = rtmod.LCds.where((rtmod.LAIAllds.MODIS_LAI_eff>1.)&(rtmod.LAIAllds.MODIS_LAI_eff<3.)).chm.values
  #chm = als.where((rtmod.LAIAllds.MODIS_LAI_eff.values>1.)&(rtmod.LAIAllds.MODIS_LAI_eff.values<3.)).values
  c = ax.pcolormesh(E, N, chm-np.nanmean(chm), vmin=-15, vmax=15, cmap='PiYG', zorder=1)
  border.boundary.plot(ax=ax, fc='none', ec='w', zorder=5)
  ax.set_xlim(E[iceMask].min(), (E[iceMask].max()+500))
  ax.set_ylim(N[iceMask].min(), (N[iceMask].max()+500))
  ax.set_xlabel('Easting (km)')
  ax.set_ylabel('Northing (km)')
  tickSep = 100000
  Eticks = np.arange(int(E[iceMask].min()/tickSep)*tickSep, tickSep+int(E[iceMask].max()/tickSep)*tickSep, tickSep)
  Nticks =np.arange(int(N[iceMask].min()/tickSep)*tickSep, tickSep+int(N[iceMask].max()/tickSep)*tickSep, tickSep)
  ax.set_xticks(Eticks, (Eticks/1000).astype(int))
  ax.set_yticks(Nticks, (Nticks/1000).astype(int))
  ax.fill_between(E[:,0], N[:,0], np.full(len(N), np.max(N)), fc='k', alpha=0.5, zorder=10)
  ax.set_aspect('equal')
  fig.colorbar(c, ax=ax, label='Canopy height deviation from mean (m)')
  fig.tight_layout()
  fig.savefig('final_plots/LAI1to3.icesat2.deviation.withBorder.pdf')
