import glob, os, pdb
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

def readSPARTACUS(outFile, mask):
  ds = xr.open_dataset(outFile)
  ds['ALS_mask'] = (('y', 'x'), mask.values)
  ds = ds.where(ds.ALS_mask)
  #ds = ds.sel(x=slice(1112183, 1306773), y=slice(7521651, 7397019))
  for v in ds.variables:
     if v not in ds.coords:
       ds[v] = ds[v].astype(float) / 100.
  ds['mask'] = (ds.MODIS_NDSI_Snow_Cover>0.5)&(ds.PFT_LC==1)&(ds.omega>0.)&(ds.MODIS_LAI>0.)&(ds.zen>0.)&(ds.zen<90.)&(ds.MODIS_Snow_Albedo>0.)&(ds.MODIS_BSA_nir>0.)#&(ds.lon>20.)&(ds.lon<32.)&(ds.lat>60.)&(ds.lat<70.)
  return ds.where(ds.mask).to_dataframe().dropna()

def plotRelationship(df, x, y):
  print('plotting', y, 'against', x, '@', datetime.datetime.now())
  fig, ax = plt.subplots(1,1, figsize=(9,9))
  dfi = df[[x,y]].where(df[[x,y]]>-1).dropna()
  sns.histplot(dfi, x=x, y=y, ax=ax, binwidth=(0.1, 0.01))
  #plt.scatter(dfi[x], dfi[y], s=1, alpha=0.05)
  dfi['xBin'] = np.round(dfi[x])
  group = dfi[['xBin',y]].groupby('xBin').mean()
  ax.plot(group.index, group, label='Mean', c='r')
  res = linregress(dfi[x], dfi[y])
  xArr = np.array([dfi[x].min(), dfi[x].max()])
  ax.plot(xArr, res.slope*xArr+res.intercept, label='lstsq', c='k')
  ax.legend(loc='best', title='R=%.3f' % res.rvalue,
            edgecolor='none', facecolor='none')
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  fig.tight_layout()
  #fig.show()
  fig.savefig('plots/%s.%s.finland.test.png' % (x, y))
  plt.close()

if __name__=='__main__':
  outDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/nc'
  outList = sorted(glob.glob(os.path.join(outDir, '*.nc')))
  
  chm = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/ndsm_modis.tif')
  cv = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/canopy_cover_modis.tif')
  mask = ((cv >=0) & (cv <= 1) & (chm >= 0) & (chm <50)).sel(band=1).drop(['band', 'spatial_ref'])

  df = pd.DataFrame()
  cnt = 1
  for outFile in outList:
    print(cnt, '/', len(outList), end='\r'); cnt+=1
    df = pd.concat([df, readSPARTACUS(outFile,mask)], ignore_index=True)
  
  for wv in ['vis', 'nir']:
    for sa in ['BSA', 'WSA']:
      for rt in ['Sellers']:
        lab = '%s_%s' % (sa, wv)
        df['%s_%s_diff' % (rt, lab)] = df['%s_%s' % (rt, lab)] -\
                                       df['MODIS_%s' % lab]
  
  plotRelationship(df, 'zen', 'Sellers_BSA_nir_diff')
  """
  indVar = [v for v in df.columns if 'nir_diff' not in v]
  dfpos = df.where(df.Sellers_BSA_nir_diff > 0).dropna()
  dfneg = df.where(df.Sellers_BSA_nir_diff < -.25).dropna()
  dfmid = df.where((df.Sellers_BSA_nir_diff >= -.25)&(df.Sellers_BSA_nir_diff <= 0)).dropna()
  
  for v in indVar:
    if df[v].max() != df[v].min():
      fig, ax = plt.subplots(1,1,figsize=(9,9))
      if v!='date':
        b = np.linspace(df[v].min(), df[v].max(), 100)
      else:
        b = np.unique(df[v].values)
      bplot = b[:-1] + ((b[1]-b[0])/2.)
      pos, _ = np.histogram(dfpos[v], bins=b)
      mid, _ = np.histogram(dfmid[v], bins=b)
      neg, _ = np.histogram(dfneg[v], bins=b)
      ax.plot(bplot, pos, label='upper')
      ax.plot(bplot, mid, label='middle')
      ax.plot(bplot, neg, label='lower')
      ax.legend(loc='best')
      ax.set_xlabel(v)
      ax.set_ylabel('Frequency')
      fig.savefig('clusterHists/%s.png' % v)
      plt.close()
  """
