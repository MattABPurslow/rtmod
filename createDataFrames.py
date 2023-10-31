import glob, os, pdb
import datetime
import numpy as np
import pandas as pd
import xarray as xr

def readSPARTACUS(outFile):
  out = xr.open_dataset(outFile).isel(layer=-1, layer_interface=-1).rename({'column':'ind'})
  wv = os.path.split(outFile)[-1].split('.')[-2]
  ds = xr.open_dataset(outFile.replace('spartacusOut', 'spartacusIn')).isel(layer=-1, layer_int=-1)
  ds['SPARTACUS_%s' % wv] = out.flux_up_layer_top_sw
  ds['Sellers_%s_diff' % wv] = ds['Sellers_%s' % wv] - ds['MODIS_%s' % wv]
  ds['SPARTACUS_%s_diff' % wv] = ds['SPARTACUS_%s' % wv] - ds['MODIS_%s' % wv]
  ds.to_dataframe().to_pickle(outFile.replace('spartacusOut', 'spartacusPickle').replace('.nc', '.pkl'))

if __name__=='__main__':
  outDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacusOut'
  outList = sorted(glob.glob(os.path.join(outDir, '*.nc')))
  
  cnt = 1
  for outFile in outList:
    print(cnt, '/', len(outList), end='\r'); cnt+=1
    readSPARTACUS(outFile)
