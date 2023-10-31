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
    """
    Initialise by loading data with time-invariate inputs
    """
    self.rootDir = rootDir
    self.k = 0.5 # As in JULES
    self.crownRatio = 12.9/2.7 # for boreal & montane (Jucker et al 2022)
    self.wv = ['vis', 'nir', 'shortwave', 'vis', 'nir', 'shortwave']
    self.df = ['BSA', 'BSA', 'BSA', 'WSA', 'WSA', 'WSA']
    self.createSellers()
    self.sellers = np.vectorize(self.sellersUfunc)
    self.getRTdict()
    self.getAlbDict()
    self.getOut()

  def main(self):
    """
    Run Sellers and SPARTACUS for each day in doyList
    """
    self.getSellers()
    self.runSPARTACUS()
  
  def getOut(self):
    column = 0
    k = 0.5
    #for solar_zenith_angle in np.arange(0., 90., 10.):
    for lai in np.arange(0.5, 5.1, 0.5):
        for veg_fsd in np.arange(0.25, 1., 0.25):
          for veg_fraction in np.arange(0.1, 1., 0.2):
            for height in np.arange(5., 46., 10.):
              for veg_scale in [0.1, 0.2, 0.5, 1., 2., 5., 10.]:
                for ground_sw_albedo in np.arange(0.25, 1., 0.5):
                  for ground_sw_albedo_direct in np.arange(0.25, 1., 0.5):
                    for veg_sw_r, veg_sw_t in zip([0.1, 0.4], [0.1, 0.4]):
                      if (veg_sw_r+veg_sw_t)<=1:
                        for top_flux_dn_direct_sw in np.arange(0., 1.01, 0.5):
                          print(column, end='\r')
                          self.ds = xr.Dataset()
                          ## Default values
                          #self.ds.coords['column'] = [column]
                          zen = np.arange(0., 90., 10.)
                          Nc = len(zen)
                          self.ds.coords['column'] = np.arange(column, column+Nc, 1)
                          self.ds['solar_zenith_angle'] = ('column', zen)
                          self.ds['cos_solar_zenith_angle'] = np.cos(np.deg2rad(self.ds.solar_zenith_angle))
                          self.ds.coords['layer'] = [0]
                          self.ds.coords['layer_int'] = [0,1]
                          self.ds.coords['surface_type'] = int(1)
                          self.ds.coords['nlayer'] = int(1)
                          self.ds['veg_contact_fraction'] = (('column','layer'), [[0.]]*Nc)
                          self.ds['building_fraction']  = (('column','layer'), [[0.]]*Nc)
                          self.ds['building_scale'] = (('column','layer'), [[0.]]*Nc)
                          self.ds['clear_air_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds['veg_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds['veg_air_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds['air_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds.coords['ground_temperature'] = ('column', [273.]*Nc)
                          self.ds['roof_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds['wall_temperature'] = (('column','layer'), [[273.]]*Nc)
                          self.ds['ground_lw_emissivity'] = (('column','lw'), [[1.]]*Nc)
                          self.ds['veg_lw_ssa'] = (('column','layer', 'lw'), [[[1.]]]*Nc)
                          self.ds['roof_sw_albedo'] = (('column','sw'), [[1.]]*Nc)
                          self.ds['roof_sw_albedo_direct'] = (('column','sw'), [[1.]]*Nc)
                          self.ds['roof_lw_emissivity'] = (('column','lw'), [[1.]]*Nc)
                          self.ds['wall_sw_albedo'] = (('column','sw'), [[1.]]*Nc)
                          self.ds['wall_sw_albedo_direct'] = (('column','sw'), [[1.]]*Nc)
                          self.ds['wall_lw_emissivity'] = (('column','lw'), [[1.]]*Nc)
                          self.ds['sky_temperature'] = ('column',[273.]*Nc)
                          self.ds['top_flux_dn_sw'] = (('column','sw'), [[1.]]*Nc)
                          self.ds['lai'] = (('column','layer'), [[lai]]*Nc)
                          self.ds['veg_fsd'] = (('column','layer'), [[veg_fsd]]*Nc)
                          self.ds['k'] = (('column','layer'), [[k]]*Nc)
                          self.ds['veg_extinction'] = self.ds.k * self.ds.lai
                          self.ds['veg_fraction'] = (('column','layer'), [[veg_fraction]]*Nc)
                          self.ds['height'] = (('column','layer_int'), [[0., height]]*Nc)
                          self.ds['veg_scale'] = (('column','layer'), [[veg_scale]]*Nc)
                          self.ds['ground_sw_albedo'] = (('column','sw'), [[ground_sw_albedo]]*Nc)
                          self.ds['ground_sw_albedo_direct'] = (('column','sw'), [[ground_sw_albedo_direct]]*Nc)
                          self.ds['veg_sw_r'] = (('column','sw'),[[veg_sw_r]]*Nc)
                          self.ds['veg_sw_t'] = (('column','sw'),[[veg_sw_t]]*Nc)
                          self.ds['veg_sw_ssa'] = self.ds.veg_sw_r + self.ds.veg_sw_t
                          self.ds['top_flux_dn_direct_sw'] = (('column', 'sw'), [[top_flux_dn_direct_sw]]*Nc)
                          if column>0:
                            self.out = xr.concat([self.out, self.ds], dim='column')
                          else:
                            self.out = self.ds.copy()
                          column += Nc
    inFile = os.path.join(self.rootDir, 'sensitivityIn.nc')
    self.out.to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')

  def getSellers(self):
    """
    Run Sellers for all pixels in out dataset
    """
    #proportion of diffuse radiation
    k = 'Sellers_Albedo'
    self.out[k] = xr.apply_ufunc(self.sellers,
                                 self.out.cos_solar_zenith_angle,
                                 self.out.lai,
                                 self.out.ground_sw_albedo,
                                 self.out.veg_sw_r,
                                 self.out.veg_sw_t)
  
  def createSellers(self):
    """
    Create single layer Sellers instance
    """
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1

  def sellersUfunc(self, mu, LAI, SC, r, t):
    """
    Run Sellers for given inputs
    mu: cosine of solar zenith angle
    LAI: effective LAI
    SC: subcanopy albedo
    pft: Plant Functional Type
    wv: waveband
    """
    #cosine of solar zenith:
    self.t.mu=mu
    #leaf area index
    self.t.lai=LAI
    #leaf reflectance & tranmisttance
    self.t.leaf_r = r
    self.t.leaf_t = t
    #soil reflectance
    self.t.soil_r=SC#self.getGroundAlbedo(SC, wv)
    #do the radiative transfer calculation:
    Iup, _, _, _ = self.t.getFluxes()
    return Iup[0]

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
  def getExtinction(self):
    """
    Calculate vegetation extinction coefficient
    """
    self.ds['omega'] = self.ds.omega.where((self.ds.omega > 0) & (self.ds.omega < 1))
    self.ds['veg_extinction'] = (('layer', 'y', 'x'), [self.k*self.ds.MODIS_LAI_eff])

  def getFSD(self):
    """
    Define variability in LAI
    """
    self.ds['veg_fsd'] = (('layer', 'y', 'x'), [self.ds.MODIS_LAI_Sigma.where(self.ds.MODIS_LAI_Sigma>0)])

  def getSA(self, Nsw):
    """
    Retrieve subcanopy albedo
    """
    self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [self.ds.MODIS_Snow_Albedo.where((self.ds.MODIS_Snow_Albedo>0.)&(self.ds.MODIS_Snow_Albedo<1.))]*Nsw)
    #gGA = np.vectorize(self.getGroundAlbedo)
    #self.ds['ground_sw_albedo'] = (('sw', 'y', 'x'), [gGA(self.ds.MODIS_NDSI_Snow_Cover, wv) for wv in self.wv])
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
    del self.ds
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
    self.spartVars = [#'cos_solar_zenith_angle',
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
    self.out['column'] = np.arange(self.out.ind.shape[0])
    self.spartVars.append('column')
    self.out = self.out.sel(ind=self.out[self.spartVars].column).dropna(dim='column')
    self.out['veg_fraction'] = self.out.veg_fraction / 100.
    self.out = self.out.transpose('column', 'layer', 'layer_int', 'lw', 'sw')

  def runSPARTACUS(self):
    """
    Save SPARTACUS input files and run
    """
    ## Remove any pixels with persistent missing data 
    self.out = self.out.dropna(dim='column')
    inFile = os.path.join(self.rootDir, 'sensitivityIn.nc')
    self.out.isel(sw=[i]).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')
    outFile = inFile.replace('In', 'Out')
    command = ['spartacus_surface', 'config.nam', inFile, outFile]
    r=os.system(' '.join(command))

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  rtmod = RTMOD(rootDir)
  rtmod.main()

"""
start=220
step = 1
for c in range(start, self.out.isel(sw=[i]).column.shape[0], step):
  inFile = os.path.join(self.rootDir, 'spartacusIn/sodankyla.%s.%s_%s.nc' % (self.doy, self.df[i], self.wv[i]))
  if step>1:
    self.out.isel(sw=[i], column=slice(c, c+step-1)).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')
  else:
    self.out.isel(sw=[i], column=[c]).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')
  outFile = inFile.replace('spartacusIn', 'spartacusOut')
  command = ['spartacus_surface', 'config.nam', inFile, outFile]
  r=os.system(' '.join(command))
  if r!=0:
    print(c)
    pdb.set_trace()
"""

