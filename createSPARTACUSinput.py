import os, glob, pdb
import numpy as np
import xarray as xr
import rioxarray as rx
import matplotlib.pyplot as plt
from pyproj import CRS, Proj, Transformer

class SPARTACUS(object):
  def __init__(self, MODISdir, ALSfile):
    self.MODISdir = MODISdir
    self.ALSfile = ALSfile

  def getMODIS(self):
    print('Reading MODIS tiles')
    self.ds = xr.Dataset()
    for MODISfile in sorted(glob.glob(os.path.join(self.MODISdir, '*.nc'))):
      self.ds = xr.merge([self.ds, xr.open_dataset(MODISfile)])
    varNames = ['zen', 'MODIS_LAI', 'IGBP_LC', 'PFT_LC', 'lon', 'lat',
                'MODIS_Snow_Albedo', 'MODIS_BSA_vis', 'MODIS_WSA_vis',
                'MODIS_BSA_nir', 'MODIS_WSA_nir']
    self.ds['mu'] = np.cos(np.deg2rad(self.ds.zen))

  def getALS(self):
    als = xr.open_dataset(self.ALSfile)
    chm = np.full(self.ds.lon.shape, np.nan)
    cv = np.full(self.ds.lon.shape, np.nan)
    xArr = als.x.sel(x=self.ds.x, method='nearest')
    yArr = als.y.sel(y=self.ds.y, method='nearest')
    xArr = np.append(xArr, xArr[-1]+(xArr[-1]-xArr[-2]))
    yArr = np.append(yArr, yArr[-1]+(yArr[-1]-yArr[-2]))
    pdb.set_trace()
    for i in range(len(yArr)-1):
      for j in range(len(xArr)-1):
        alsij = als.sel(x=slice(xArr[j],xArr[j+1]), y=slice(yArr[i],yArr[i+1]))
        chm[i,j] = alsij.chm.mean(skipna=True).values
        cv[i,j] = alsij.cv.mean(skipna=True).values
    self.ds['chm'] = (('y', 'x'), chm/100.)
    self.ds['cv'] = (('y', 'x'), cv/100.)
 
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

  def getSPARTACUSinput(self):
    mu = ds.mu.values.flatten()
    chm = np.random.rand(5)
    cv = np.random.rand(5)
    Lab = np.random.rand(5)
    omega = np.random.rand(5)
    omegaSD = np.random.rand(5)/10
    LAI = ds.MODIS_LAI.values.flatten()
    LAISD = ds.MODIS_LAI_sigma.values.flatten()
    SA_vis = np.random.rand(5)
    SA_nir = np.random.rand(5)
    r_vis = 0.07
    t_vis = 0.10
    r_nir = 0.42
    t_nir = 0.40
    diffFrac = 0
    
    ds = xr.Dataset()
    ds['cos_solar_zenith_angle']  = (('col'), np.cos(zenRad))
    ds['surface_type'] = (('col'), np.full(ds.col.shape, int(1)))
    ds['nlayer'] = (('col'), np.full(ds.col.shape, int(1)))
    ds['height'] = (('col', 'layer_int'), np.array([chm]).T)
    ds['veg_fraction'] = (('col', 'layer'), np.array([cv]).T)
    ds['veg_scale'] = (('col', 'layer'), np.array([Lab]).T)
    ds['veg_extinction'] = (('col', 'layer'), np.array([omega * LAI]).T)
    ds['veg_fsd'] = (('col', 'layer'), np.array([omegaSD *  LAISD]).T)
    ds['veg_contact_fraction'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 0.))
    ds['building_fraction']  = (('col', 'layer'), np.full(ds.veg_fsd.shape, 0.))
    ds['building_scale'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 0.))
    ds['clear_air_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['veg_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['veg_air_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['air_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['ground_temperature'] = (('col'), np.full(ds.col.shape, 273.))
    ds['roof_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['wall_temperature'] = (('col', 'layer'), np.full(ds.veg_fsd.shape, 273.))
    ds['ground_sw_albedo'] = (('col', 'sw'), np.array([SA_vis, SA_nir]).T)
    ds['ground_sw_albedo_direct'] = (('col', 'sw'), np.array([SA_vis, SA_nir]).T)
    ds['ground_lw_emissivity'] = (('col', 'lw'), np.full((ds.col.shape[0], 1), 1.))
    ds['veg_sw_ssa'] = (('col', 'layer', 'sw'), np.array([[np.array([r_vis+t_vis, r_nir+t_nir])]]*ds.col.shape[0]))
    ds['veg_lw_ssa'] = (('col', 'layer', 'lw'), np.full((ds.col.shape[0],1,1), 1.))
    ds['roof_sw_albedo'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.))
    ds['roof_sw_albedo_direct'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.))
    ds['roof_lw_emissivity'] = (('col', 'lw'), np.full(ds.ground_lw_emissivity.shape, 1.))
    ds['wall_sw_albedo'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.))
    ds['wall_sw_albedo_direct'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.))
    ds['wall_lw_emissivity'] = (('col', 'lw'), np.full(ds.ground_lw_emissivity.shape, 1.))
    ds['sky_temperature'] = (('col'), np.full(ds.col.shape, 273.))
    ds['top_flux_dn_sw'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.))
    ds['top_flux_dn_direct_sw'] = (('col', 'sw'), np.full(ds.ground_sw_albedo.shape, 1.-diffFrac))
    return ds    

if __name__=='__main__':
  MODISdir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MODISnc/'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.als.modisgrid.nc'
  spart = SPARTACUS(MODISdir, ALSfile)
  spart.getMODIS()
  #spart.getALS()
  #spart.getClumping()
