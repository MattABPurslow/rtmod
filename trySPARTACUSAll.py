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

class RTMOD(object):
  def __init__(self, rootDir, ALSfile):
    self.rootDir = rootDir
    self.ALSfile = ALSfile
    self.k = 0.5 # As in JULES
    self.crownRatio = 12.9/2.7 # for boreal & montane (Jucker et al 2022)
    self.wv = ['vis', 'nir', 'vis', 'nir']
    self.df = ['BSA', 'BSA', 'WSA', 'WSA']
    self.nCPUs = 8
    self.getRTdict()
    self.getAlbDict()
    self.getMODISgrid()
    self.getALS()
    self.getDOYs()
  
  def main(self):
    #with Pool(self.nCPUs) as pool:
    #  pool.map(self.runDOY, self.doyList)
    for doy in self.doyList:
      try:
        print(doy, end='\r')
        self.runDOY(doy)
      except:
        pass
  
  def runDOY(self, doy):
    self.getMODIS(doy)
    self.createSPARTACUSinput()
    self.runSPARTACUS()
  
  def getDOYs(self):
    self.doyList = np.sort([os.path.split(d)[-1].split('.')[-2]\
                            for d in glob.glob(os.path.join(self.rootDir, 'nc/*.nc'))])
   
  def getMODISgrid(self):
    lcFile = sorted(glob.glob(os.path.join(self.rootDir, 'MCD12Q1/*.hdf')))[-1]
    lc = rx.open_rasterio(lcFile)
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("4326"),
                                 always_xy=True)
    X, Y = np.meshgrid(lc.x, lc.y)
    self.getLCdict(lc)
    lon, lat = trans.transform(X, Y)
    lc['lon'] = (('y', 'x'), lon)
    lc['lat'] = (('y', 'x'), lat)
    self.grid = lc[['lon', 'lat']].drop_vars('spatial_ref')
    self.grid.attrs.clear()
  
  def getMODIS(self, doy):
    self.doy = doy
    self.ds = xr.open_dataset(os.path.join(self.rootDir, 'nc/sodankyla.%s.nc' % doy))
    self.ds = self.ds.sel(y=self.grid.y, x=self.grid.x).isel(date=0)
    for v in self.ds.variables:
      if v not in self.ds.coords:
        self.ds[v] = self.ds[v].astype(np.float64)/100.
    self.ds['chm'] = self.grid.chm.astype(np.float64)
    self.ds['cv'] = self.grid.cv.astype(np.float64)
    self.ds = self.ds.drop(['band', 'spatial_ref'])
  
  def getALS(self):
    #als = xr.open_dataset(self.ALSfile)
    #self.grid = self.grid.sel(x=slice(als.x.min(), als.x.max()), y=slice(als.y.max(), als.y.min()))
    cv = rx.open_rasterio(self.ALSfile.replace('als','canopy_cover_modis')).sel(band=1)
    chm = rx.open_rasterio(self.ALSfile.replace('als','ndsm_modis')).sel(band=1)
    #self.grid['chm'] = als.chm
    #self.grid['cv'] = als.cv
    self.grid['chm'] = (('y', 'x'), chm.where((chm>=0.)&(chm<50.)).values)
    self.grid['cv'] = (('y', 'x'), (cv.where((cv>0.)&(cv<1.)).values*100.))
    
  def getRTdict(self):
    ## Extract initial estimates of leaf reflectance and transmittance for PFT
    rtDict = {'Broadleaf Crop':             {'r': {'vis': 0.08, 'nir': 0.42}, 
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

  def getLCdict(self, LandCover):
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}

  def getExtinction(self):
    self.ds['omega'] = self.ds.omega.where((self.ds.omega > 0) & (self.ds.omega < 1))
    self.ds['veg_extinction'] = (('layer', 'y', 'x'), [self.k*self.ds.MODIS_LAI*self.ds.omega])
  
  def getFSD(self):
    self.ds['veg_fsd'] = (('layer', 'y', 'x'), [self.ds.MODIS_LAI_Sigma])
 
  def getGroundAlbedo(self, snowCover, wv):
    snowAlbedo = self.albDict['sa'][wv]
    soilAlbedo = self.albDict['ga'][wv]
    if snowCover>0.5:
      return snowAlbedo
    else:
      return soilAlbedo
    #return (snowCover*snowAlbedo) + ((1-snowCover)*soilAlbedo)

  def getAlbDict(self):
    self.albDict = {'sa': {'vis':0.6, 'nir':0.6},
                    'ga': {'vis':0.15, 'nir':0.83}}
   
  def getSA(self, Nsw):
    #elf.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [self.ds.MODIS_Snow_Albedo]*Nsw)
    gGA = np.vectorize(self.getGroundAlbedo)
    self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [gGA(self.ds.MODIS_NDSI_Snow_Cover, wv) for wv in self.wv])
    self.ds['ground_sw_albedo_direct'] = self.ds.ground_sw_albedo.copy()
  
  def getSSA(self):
    ssa_vis = np.full(self.ds.PFT_LC.shape, np.nan)
    ssa_nir = np.full(self.ds.PFT_LC.shape, np.nan)
    PFTarr = np.unique(self.ds.PFT_LC)
    for PFT in PFTarr:
      if PFT in self.lcDict.keys():
        if self.lcDict[PFT] in self.rtDict.keys():
          mask = self.ds.PFT_LC==PFT
          ssa_vis[mask] = self.rtDict[self.lcDict[PFT]]['r']['vis'] + self.rtDict[self.lcDict[PFT]]['t']['vis']
          ssa_nir[mask] = self.rtDict[self.lcDict[PFT]]['r']['nir'] +  self.rtDict[self.lcDict[PFT]]['t']['nir']
    self.ds['veg_sw_ssa'] = (('layer', 'sw', 'y', 'x'), [[ssa_vis, ssa_nir, ssa_vis, ssa_nir]])

  def createSPARTACUSinput(self):
    self.ds['mu'] = self.ds.mu.where((self.ds.mu>0))
    self.ds = self.ds.rename({'mu': 'cos_solar_zenith_angle'})
    self.ds['height'] = (('layer_int','y','x'), [np.full(self.ds.chm.shape,0.), 
                                                 self.ds.chm])
    self.ds['veg_fraction'] = (('layer', 'y', 'x'), [self.ds.cv])
    S = (self.ds.chm/self.crownRatio)
    self.ds['veg_scale'] = ((4.*self.ds.veg_fraction)/S).where(S>0)
    ## Default values
    Ny, Nx = self.ds.y.shape[0], self.ds.x.shape[0]
    Nsw = 4
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
    self.ds['top_flux_dn_direct_sw'] = (('y', 'x', 'sw'), np.full((Ny, Nx, Nsw), [0.,0.,1.,1.]))
    self.getExtinction()
    self.getFSD()
    self.getSA(Nsw)
    self.getSSA()

  def runSPARTACUS(self):
    self.out = self.ds.stack(col=('y', 'x'), create_index=False)
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
    self.spartVars = ['cos_solar_zenith_angle',
                      'surface_type',
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
    self.out['ind'] = np.arange(self.out.col.shape[0])
    self.spartVars.append('ind')
    self.out = self.out.sel(col=self.out[self.spartVars].ind).dropna(dim='ind')
    self.out['veg_fraction'] = self.out.veg_fraction / 100.
    self.wv = ['vis', 'nir', 'vis', 'nir']
    self.df = ['BSA', 'BSA', 'WSA', 'WSA']
    for i in range(self.out.sw.shape[0]):
      inFile = os.path.join(self.rootDir,
               'spartacusIn/sodankyla.%s.%s_%s.nc'\
               % (self.doy, self.df[i], self.wv[i]))
      self.out.isel(sw=[i]).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  #ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.alsmodisgrid.nc'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/als.tif'
  rtmod = RTMOD(rootDir, ALSfile)
  rtmod.main()
