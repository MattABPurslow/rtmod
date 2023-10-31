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
    self.getMODISdirs()
    self.getDOYs()
    self.getLandCover()
  
  def main(self):
    for doy in self.doyList:
      self.runDOY(doy)
      self.logStats()
  
  def logStats(self):
    outfile = os.path.join(self.rootDir, 'pkl/%s.%s.pkl' % ('sodankyla', self.doy))
    self.LAIds['date'] = self.LAIds.date.astype(np.datetime64)
    self.Snowds['date'] = self.Snowds.date.astype(np.datetime64)
    self.Albedods['date'] = self.Albedods.date.astype(np.datetime64)
    lcVars = ['PFT_LC', 'lon', 'lat', 'omega']
    snowVars = ['MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover']
    laiVars = ['MODIS_LAI', 'MODIS_LAI_Sigma']
    albedoVars = ['MODIS_BSA_vis', 'MODIS_BSA_nir',
                  'MODIS_WSA_vis', 'MODIS_WSA_nir']
    LCds = self.LCds[lcVars]
    LAIds = self.LAIds[laiVars]
    Snowds = self.Snowds[snowVars]
    Albedods = self.Albedods[albedoVars]
    df = xr.merge([LCds, LAIds, Snowds, Albedods]).to_dataframe()
    df['year'] = self.doy[:-3]
    df['doy'] = self.doy[-3:]
    df.to_pickle(outfile)
  
  def runDOY(self, doy):
    print(doy)
    self.getMODIS(doy)
 
  def getMODIS(self, doy):
    self.doy = doy
    self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                 .replace(tzinfo=datetime.timezone.utc)
    self.getMODISfiles()
    self.getLAI()
    self.getSnow()
    self.getAlbedo()
 
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
              max(self.doyList))
   
  def getLandCover(self):
    print('reading land cover @', datetime.datetime.now())
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

  def getClumping(self):
    print('retrieving he et al (2012) clumping index @', datetime.datetime.now())
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

  def getLAI(self):
    print('reading LAI @', datetime.datetime.now())
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
    print('reading snow @', datetime.datetime.now())
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
    print('reading albedo @', datetime.datetime.now()) 
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

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  rtmod = RTMOD(rootDir)
  rtmod.main()
  del rtmod
  pklList = np.sort(glob.glob(os.path.join(rootDir, 'pkl/*.pkl')))
  df = pd.DataFrame()
  for pkl in pklList:
    df = pd.concat([df, pd.read_pickle(pkl)])

  getDate = np.vectorize(lambda doy: datetime.datetime.strptime(doy, '%Y%j'))
  dateList = getDate([p.split('.')[-2] for p in pklList])
  monthList = np.array([d.month for d in dateList])
  months = np.arange(1, 12.1, 1).astype(int)
  meanLAI = np.full(months.shape, np.nan)
  meanSC = np.full(months.shape, np.nan)
  for m in months:
    df = pd.DataFrame()
    for pkl in pklList[monthList==m]:
      df = pd.concat([df, pd.read_pickle(pkl)])
    df = df[df.PFT_LC==1]
    meanLAI[m-1] = df.MODIS_LAI.mean()
    meanSC[m-1] = df.MODIS_NDSI_Snow_Cover.mean()
