import glob, os, pdb
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

def readSPARTACUS(outFile):
  out = xr.open_dataset(outFile).isel(layer=-1, layer_interface=-1).rename({'column':'ind'})
  wv = os.path.split(outFile)[-1].split('.')[-2]
  ds = xr.open_dataset(outFile.replace('spartacusOut', 'spartacusIn')).isel(layer=-1, layer_int=-1)
  ds['SPARTACUS_%s' % wv] = out.flux_up_layer_top_sw
  return ds.to_dataframe()

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
  fig.savefig('plots/juneend.%s.%s.pdf' % (x, y))
  plt.close()

if __name__=='__main__':
  outDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacusOut'
  outList = sorted(glob.glob(os.path.join(outDir, '*.nc')))
  """
  cnt = 1
  for outFile in outList:
    print(cnt, '/', len(outList), end='\r'); cnt+=1
    df = readSPARTACUS(outFile)
    df.to_csv(outFile.replace('.nc', '.csv'))
  """
  df = pd.DataFrame()
  cnt = 1
  for outFile in outList:
    print(cnt, '/', len(outList), end='\r'); cnt+=1
    if outFile[-10:-3]=='BSA_nir':
      df = pd.concat([df, pd.read_csv(outFile.replace('.nc', '.csv'), index_col=0)], axis=0)
 
  sa = 'BSA'
  wv = 'nir'
  for rt in ['Sellers', 'SPARTACUS']:
    lab = '%s_%s' % (sa, wv)
    df['%s_%s_diff' % (rt, lab)] = df['%s_%s' % (rt, lab)] -\
                                   df['MODIS_%s' % lab]

  df.to_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacus.out.pkl')
  """
  cols = ['PFT_LC', 'lon', 'lat', 'omega', 'MODIS_LAI',
         'MODIS_LAI_eff', 'MODIS_Snow_Albedo',
         'MODIS_NDSI_Snow_Cover', 'date',
         'chm', 'cv', 'height', 'veg_fraction', 'veg_scale']
  
  for col in cols:
    df.where(df.MODIS_NDSI_Snow_Cover<0.01).plot.scatter('zen', 'Sellers_BSA_nir_diff', c=col, s=1)
    plt.title('No Snow')
    plt.savefig('scatter/nosnow.%s.png' % col)
    plt.close()
  
  for col in cols:
    df.where(df.MODIS_NDSI_Snow_Cover>0.9).plot.scatter('zen', 'Sellers_BSA_nir_diff', c=col, s=1)
    plt.title('Snow')
    plt.savefig('scatter/snow.%s.png' % col)
    plt.close()

  """
  #plotRelationship(df, 'zen', 'Sellers_BSA_nir_diff')
  #plotRelationship(df, 'zen', 'SPARTACUS_BSA_nir_diff')
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
