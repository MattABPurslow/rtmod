import os, pdb
import glob
import datetime
import numpy as np
import rioxarray as rx
import xarray as xr
from pysolar.solar import get_altitude
from multiprocessing import Pool
from functools import partial
from pyproj import Proj, CRS
from pyproj import Transformer
import matplotlib.pyplot as plt
from progress.bar import Bar
from matplotlib.cm import viridis as cmap

def getLonLat(LandFile, lonMin, lonMax, latMin, latMax):
  ## Define MODIS coordinates
  MODProj = Proj('+proj=sinu +R=6371007.181')
  LandCover = rx.open_rasterio(LandFile).sel(band=1)
  trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("4326"), always_xy=True)
  x, y = LandCover.x, LandCover.y
  X, Y = np.meshgrid(x, y)
  lon, lat = trans.transform(X, Y)
  siteMask = (lon>=lonMin)&(lon<=lonMax)&(lat>=latMin)&(lat<=latMax)
  j, i = np.argwhere(siteMask).T
  xSlice, ySlice = slice(i.min(),i.max()), slice(j.min(),j.max())
  return lon, lat, i.min(), i.max(), j.min(), j.max()

def getZen(lon, lat, date):
  now = date + datetime.timedelta(hours=12. + (lon/15.))
  return float(90) - get_altitude(lat, lon, now)
  
getZen = np.vectorize(getZen)

def getLandCover(LandFile, lon, lat, xSlice, ySlice):
  ## Get Forest Mask
  LandCover = rx.open_rasterio(LandFile).sel(band=1)
  LandCover = LandCover.rename({'LC_Type1':'IGBP_LC','LC_Type5':'PFT_LC'})
  LandCover['lon'] = (('y', 'x'), lon)
  LandCover['lat'] = (('y', 'x'), lat)
  LandCover = LandCover.isel(y=ySlice, x=xSlice)
  LandCover['ForestMask'] = LandCover.IGBP_LC <= 5
  LandCover = LandCover.drop(['band', 'spatial_ref'])
  LandCover.attrs.clear()
  LandCover = LandCover.drop([k for k in list(LandCover.keys())
                              if k not in ['IGBP_LC', 'PFT_LC', 'ForestMask',
                                           'lon', 'lat']])
  LandCover['IGBP_LC'] = (('y', 'x'), LandCover.IGBP_LC.values.astype(np.int8))
  LandCover['PFT_LC'] = (('y', 'x'), LandCover.PFT_LC.values.astype(np.int8))
  return LandCover

checkRTMethodUsed = np.vectorize(lambda i: bin(i)[-3:-1]=='00')

def getLAI(LAIDir, ySlice, xSlice, startDate, endDate):
  ## Get list of LAI files
  LAIFiles = np.sort(glob.glob(LAIDir+'/*.hdf'))
  ## Create empty list to hold loaded rasters
  LAIList = []
  ## Loop over LAI files
  bar = Bar('Reading LAI Files', max=len(LAIFiles))
  for LAIFile in LAIFiles:
    ## Load LAI and add date
    ## Get date
    date = datetime.datetime.strptime(os.path.split(LAIFile)[-1].split('.')[1], 'A%Y%j').replace(tzinfo=datetime.timezone.utc)
    if (date >= startDate) & (date <= endDate):
      LAI = rx.open_rasterio(LAIFile).sel(band=1)
      LAI = LAI.rename({'Lai_500m':'MODIS_LAI',
                        'FparLai_QC':'MODIS_LAI_QC',
                        'LaiStdDev_500m':'MODIS_LAI_Sigma'})
      LAI = LAI.drop(['Fpar_500m', 'FparExtra_QC', 'FparStdDev_500m'])
      LAI.coords['date'] = date
      LAI = LAI.expand_dims(dim={'date':1})
      LAI = LAI.isel(y=ySlice, x=xSlice)
      LAI['MODIS_LAI_QC'] = (('date', 'y', 'x'),
                             checkRTMethodUsed(LAI.MODIS_LAI_QC))
      LAI['MODIS_LAI'] = (LAI.MODIS_LAI-LAI.MODIS_LAI.add_offset) * \
                         LAI.MODIS_LAI.scale_factor
      LAI['MODIS_LAI_Sigma'] = (LAI.MODIS_LAI_Sigma - \
                                LAI.MODIS_LAI_Sigma.add_offset) * \
                                LAI.MODIS_LAI_Sigma
      LAI['MODIS_LAI'] = LAI.MODIS_LAI.where(LAI.MODIS_LAI<=10.)
      LAI['MODIS_LAI_Sigma'] = LAI.MODIS_LAI_Sigma.where(LAI.MODIS_LAI_Sigma<=10.)
      ## Add DataArrays to list
      LAI.attrs.clear()
      LAIList.append(LAI.drop(['band', 'spatial_ref']))
    bar.next()
  bar.finish()
  return xr.merge(LAIList)

def getSnow(SnowDir, ySlice, xSlice, dateArr):
  ## Get list of snow files
  SnowFiles = np.sort(glob.glob(SnowDir+'/*.hdf'))
  ## Create empty list to hold loaded rasters
  SnowList = []
  ## Loop over snow files
  bar = Bar('Reading Snow Cover Files', max=len(SnowFiles))
  for SnowFile in SnowFiles:
    ## Load snow and add date
    date = datetime.datetime.strptime(os.path.split(SnowFile)[-1].split('.')[1], 'A%Y%j').replace(tzinfo=datetime.timezone.utc)
    if (date >= startDate) & (date <= endDate) & (np.in1d(date, dateArr)):
      snow = rx.open_rasterio(SnowFile).sel(band=1)
      snow = snow.rename({'NDSI_Snow_Cover':'MODIS_NDSI_Snow_Cover',
                          'Snow_Albedo_Daily_Tile':'MODIS_Snow_Albedo',
                          'NDSI_Snow_Cover_Basic_QA':'MODIS_Snow_Cover_QC'})
      snow = snow.drop(['NDSI_Snow_Cover_Algorithm_Flags_QA', 'NDSI',
                        'orbit_pnt', 'granule_pnt'])
      date = datetime.datetime.strptime(os.path.split(SnowFile)[-1].split('.')[1], 'A%Y%j').replace(tzinfo=datetime.timezone.utc)
      snow.coords['date'] = date
      snow = snow.expand_dims(dim={'date':1})
      snow = snow.isel(y=ySlice, x=xSlice)
      snow['MODIS_Snow_Cover_QC'] = snow.MODIS_Snow_Cover_QC <= 2
      snow['MODIS_NDSI_Snow_Cover'] = snow['MODIS_NDSI_Snow_Cover'].where(snow['MODIS_NDSI_Snow_Cover'] <= 100) / 100.
      snow['MODIS_Snow_Albedo'] = snow['MODIS_Snow_Albedo'].where(snow['MODIS_Snow_Albedo'] <= 100) / 100.
      snow.attrs.clear()
      SnowList.append(snow.drop(['band', 'spatial_ref']))
    bar.next()
  bar.finish()
  return xr.merge(SnowList)

def getAlbedo(AlbedoDir, ySlice, xSlice, dateArr):
  ## Get list of Albedo files
  AlbedoFiles = np.sort(glob.glob(AlbedoDir+'/*.hdf'))
  ## Create empty list to hold loaded rasters
  AlbedoList = []
  ## Load each albedo file in turn
  bar = Bar('Reading Albedo Files', max=len(AlbedoFiles))
  for AlbedoFile in AlbedoFiles:
    ## Load albedo and add date
    date = datetime.datetime.strptime(os.path.split(AlbedoFile)[-1].split('.')[1], 'A%Y%j').replace(tzinfo=datetime.timezone.utc)
    if (date >= startDate) & (date <= endDate) & (np.in1d(date, dateArr)):
      Albedo = rx.open_rasterio(AlbedoFile).sel(band=1)
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
      Albedo.coords['date'] = date
      Albedo = Albedo.expand_dims(dim={'date':1})
      Albedo = Albedo.isel(y=ySlice, x=xSlice)
      # Check Full BRDF inversion used
      Albedo['MODIS_BRDF_shortwave_QC'] = Albedo.MODIS_BRDF_shortwave_QC==0
      Albedo['MODIS_BRDF_VIS_QC'] = Albedo.MODIS_BRDF_VIS_QC==0
      Albedo['MODIS_BRDF_NIR_QC'] = Albedo.MODIS_BRDF_NIR_QC==0
      for k in list(Albedo.keys()):
        if k[-2:] != 'QC':
          Albedo[k] = (Albedo[k]-Albedo[k].add_offset)*Albedo[k].scale_factor
          Albedo[k] = Albedo[k].where(Albedo[k] <= 1.0)
      Albedo.attrs.clear()
      AlbedoList.append(Albedo.drop(['band', 'spatial_ref']))
    bar.next()
  bar.finish()
  return xr.merge(AlbedoList)


def iterateExtraction(idx,
                      Nx, Ny, xArr, yArr, lon, lat, startDate, endDate,
                      LandFile, LAIDir, SnowDir, AlbedoDir, outFile):
  i, j = idx
  outFile = outFile.replace('.nc','.%s.%s.nc' \
            % (str(xArr[i]).zfill(len(str(xArr[-1]))), \
               str(yArr[j]).zfill(len(str(yArr[-1])))))
  if len(glob.glob(outFile))==0:
    xSlice = slice(xArr[i], xArr[i+1]-1)
    ySlice = slice(yArr[j], yArr[j+1]-1)
    LandCover = getLandCover(LandFile, lon, lat, xSlice, ySlice)
    LAIDS = getLAI(LAIDir, ySlice, xSlice, startDate, endDate)
    SnowDS = getSnow(SnowDir, ySlice, xSlice, LAIDS.date)
    AlbedoDS = getAlbedo(AlbedoDir, ySlice, xSlice, LAIDS.date)
    ds = xr.merge([LandCover, SnowDS, LAIDS, AlbedoDS])
    ds['date'] = ds.date.astype(np.datetime64)
    ds['zen'] = (('date', 'y', 'x'), [getZen(ds.lon, ds.lat,
                 datetime.datetime.utcfromtimestamp(int(d)/1e9)\
                 .replace(tzinfo=datetime.timezone.utc))\
                 for d in ds.date])
    ds['zen'] = ds.where(ds.zen<90.).zen
    ds.to_netcdf(outFile)

if __name__=='__main__':
  ## Define input files and directories
  site = 'sodankyla'
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/%s' % site
  outFile = os.path.join(rootDir, 'MODISnc/MODIS.%s.nc' % site)
  if site=='laret':
    sY, eY = 2021, 2022
  elif site=='sodankyla':
    sY, eY = 2018, 2019
  else:
    raise Exception('Unknown site')
  startDate = datetime.datetime(sY,9,1,tzinfo=datetime.timezone.utc)
  endDate = datetime.datetime(eY,6,30,tzinfo=datetime.timezone.utc)
  LandFile = sorted(glob.glob(os.path.join(rootDir, 'MCD12Q1/*.hdf')))[-1]
  SnowDir = os.path.join(rootDir,'MOD10A1')
  LAIDir = os.path.join(rootDir,'MCD15A3H')
  AlbedoDir = os.path.join(rootDir,'MCD43A3')
  ## Define lat/lon bands
  if site == 'laret':
    lonSite, latSite = 9.875, 46.845
  elif site == 'sodankyla':
    lonSite, latSite = 26.635, 67.365
  else:
    raise Exception('Unknown site')
  ## Get x, y bins for tiles
  lonRange = latRange = 5.0
  lonStep = latStep = 0.05
  if site=='laret':
    lonMin, lonMax = (lonSite - lonRange/2.), (lonSite + lonRange/2.)
    latMin, latMax = (latSite - latRange/2.), (latSite + latRange/2.)
  elif site=='sodankyla':
    lonMin, lonMax = lonSite-1, (lonSite + lonRange) - 1
    latMin, latMax = (latSite - latRange) + 1, latSite + 1
  else:
    raise Exception('Unknown site')

  #lon, lat, _, _, _, _ = getLonLat(LandFile, -180, 180., -90., 90.):
  #lonMin, lonMax, latMin, latMax = lon.min(), lon.max(), lat.min(), lat.max()
  lon, lat, xMin, xMax, yMin, yMax = getLonLat(LandFile, lonMin, lonMax,
                                                         latMin, latMax)
  #lonRange, latRange = lonMax-lonMin, latMax-latMin
  xArr = np.linspace(xMin, xMax+1, int(lonRange/lonStep)+1).astype(int)
  yArr = np.linspace(yMin, yMax+1, int(latRange/latStep)+1).astype(int)
  Nx, Ny = len(xArr)-1, len(yArr)-1
  idx = np.array([[[i,j] for j in range(Ny)]
                         for i in range(Nx)]).reshape(Nx*Ny, 2)
  #iterateExtraction(idx[0], Nx, Ny, xArr, yArr, lon, lat, startDate, endDate,
  #                  LandFile, LAIDir, SnowDir, AlbedoDir, outFile)
  with Pool(8) as pool:
    func = partial(iterateExtraction,
                   Nx=Nx, Ny=Ny, xArr=xArr, yArr=yArr, lon=lon, lat=lat,
                   startDate=startDate, endDate=endDate, LandFile=LandFile,
                   LAIDir=LAIDir, SnowDir=SnowDir, AlbedoDir=AlbedoDir,
                   outFile=outFile)
    pool.map(func, idx)
