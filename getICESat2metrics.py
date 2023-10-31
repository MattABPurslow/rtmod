import glob, os, pdb
import numpy as np
import h5py
import pandas as pd
import rioxarray as rx
import xarray as xr
from scipy.interpolate import interp1d
from pyproj import CRS, Proj, Transformer
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap

class atl08(object):
  def __init__(self, atl08File, gtx):
    self.atl08File = atl08File
    self.gtx = gtx
    self.ρvρg = 0.9311037715925602
    self.getATL08()
  
  def cv(self):
    self.data['cv'] = 100.*(1./(1.+(self.ρvρg*(self.data.Eg/self.data.Ev))))

  def getATL08(self):
    ''' Load additional information from ATL08 data product '''
    h5 = h5py.File(self.atl08File, 'r')
    if self.gtx in list(h5):
      if 'land_segments' in list(h5[self.gtx]):
        self.getMasks(h5['/'.join([self.gtx,'land_segments'])])
        self.data['ch'] = h5['/'.join([self.gtx,'land_segments/canopy/h_canopy'])][:]
        self.data['Eg'] = h5['/'.join([self.gtx,'land_segments/terrain/photon_rate_te'])][:]
        self.data['Ev'] = h5['/'.join([self.gtx,'land_segments/canopy/photon_rate_can_nr'])][:]
        self.data = self.data.loc[self.data.ch < 50.]
        self.clean()
        if self.data.shape[0]>0:
          self.cv()
        h5.close()
      else:
        self.data = pd.DataFrame()
    else:
      self.data = pd.DataFrame()
  
  def getMasks(self, ls):
    ''' Load ATL08 masks '''
    ## Get ATL08 masks and flags
    self.data = pd.DataFrame({'seg_id_beg': ls['segment_id_beg'][:],
                              'lon': ls['longitude'][:],
                              'lat': ls['latitude'][:],
                              'watermask': ls['segment_watermask'][:],
                              'mswmask': ls['msw_flag'][:],
                              'cloudmask': ls['layer_flag'][:],
                              'cloud_flag': ls['cloud_flag_atm'][:],
                              'cloud_fold': ls['cloud_fold_flag'][:],
                              'landcover': ls['segment_landcover'][:],
                              'snowcover': ls['segment_snowcover'][:],
                              'night': ls['night_flag'][:],
                              'sat': ls['sat_flag'][:]})
  
  def clean(self):
    self.data['clean'] = (self.data.watermask==0).astype(bool) & \
                         (self.data.snowcover==1).astype(bool) & \
                         (self.data.cloudmask==1).astype(bool) & \
                         np.isin(self.data.mswmask, [-1, 0]).astype(bool) & \
                         (self.data.sat==0).astype(bool)
    self.data = self.data.loc[self.data.clean]

if __name__=='__main__':
  finland = True
  perfect = False
  atl08Dir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ATL08'
  if perfect:
    met = pd.read_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/icesat2.pkl')
  elif finland:
    atl08List = np.loadtxt('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ATL08/finland.atl08.list', dtype=str)
    met = pd.DataFrame()
    cnt = 1
    for atl08File in atl08List:
      os.system('mv %s .' % atl08File)
      for gtx in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
        print(atl08File, gtx, cnt, len(atl08List), end='\r');
        track = atl08(os.path.split(atl08File)[-1], gtx)
        met = pd.concat([met, track.data])
      cnt += 1
      os.system('mv %s %s' % (os.path.split(atl08File)[-1], atl08File))
  else:
    met = pd.DataFrame()
    cnt = 1
    for atl08File in np.sort(glob.glob(os.path.join(atl08Dir, '*.h5'))):
      os.system('mv %s .' % atl08File)
      for gtx in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
        print(atl08File, gtx, cnt, 135, end='\r');
        track = atl08(os.path.split(atl08File)[-1], gtx)
        met = pd.concat([met, track.data])
      cnt += 1
      os.system('mv %s %s' % (os.path.split(atl08File)[-1], atl08File))

  lc = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MCD12Q1/MCD12Q1.A2021001.h19v02.061.2022216103417.hdf')
  
  MODProj = Proj('+proj=sinu +R=6371007.181')
  trans = Transformer.from_crs(CRS.from_epsg("4326"), MODProj.crs, always_xy=True)
  met['x'], met['y'] = trans.transform(met.lon, met.lat)
  met = met.loc[(met.x >= lc.x.values.min())&(met.x <= lc.x.values.max())]
  met = met.loc[(met.y >= lc.y.values.min())&(met.y <= lc.y.values.max())]
  met['MODx'] = lc.x.values[[np.argmax(x < lc.x.values)-1 for x in met.x]]
  met['MODy'] = lc.y.values[[np.argmax(y > lc.y.values)-1 for y in met.y]]
  group = met.groupby(['MODy', 'MODx']).mean()
  
  ds = xr.Dataset()
  ds.coords['x'] = lc.x.values
  ds.coords['y'] = lc.y.values
  cv = np.full((ds.y.shape[0], ds.x.shape[0]), np.nan)
  cv_ALS = np.full((ds.y.shape[0], ds.x.shape[0]), np.nan)
  chm = np.full((ds.y.shape[0], ds.x.shape[0]), np.nan)
  for idx in group.index:
    i = int(np.argwhere((ds.y.astype(int).values == int(idx[0]))))
    j = int(np.argwhere((ds.x.astype(int).values == int(idx[1]))))
    cv[i,j] = group.loc[idx].cv
    if perfect:
      cv_ALS[i,j] = group.loc[idx].cv_ALS
    chm[i,j] = group.loc[idx].ch
  
  ds['cv'] = (('y', 'x'), cv)
  if perfect:
    ds['cv_ALS'] = (('y', 'x'), cv_ALS)
  ds['chm'] = (('y', 'x'), chm)
  """ 
  fig, ax = plt.subplots(1,1)
  ax.pcolormesh(ds.x, ds.y, ds.chm)
  ax.scatter(met.x, met.y,s=1, c='k')
  ax.set_xlim(met.x.min(), met.x.max())
  ax.set_ylim(met.y.min(), met.y.max())
  fig.show()
  """
  if perfect:
    ds.to_netcdf('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/icesat2.reclassified.nc', engine='scipy', format='NETCDF3_CLASSIC')
  elif finland:
    ds.to_netcdf('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/icesat2.atl08.finland.nc', engine='scipy', format='NETCDF3_CLASSIC')
  else:
    ds.to_netcdf('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/icesat2.atl08.nc', engine='scipy', format='NETCDF3_CLASSIC')
