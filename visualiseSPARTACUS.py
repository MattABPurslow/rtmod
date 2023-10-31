import glob, os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def readSPARTACUS(outFile):
  out = xr.open_dataset(outFile).isel(layer=-1).rename({'column':'col'})
  wv = os.path.split(outFile)[-1].split('.')[-2]
  ds = xr.open_dataset(outFile.replace('spartacusOut', 'spartacusIn'))
  ds['SPARTACUS_%s' % wv] = out.flux_up_layer_top_sw
  return ds.to_dataframe()

outDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacusOut'
outList = sorted(glob.glob(os.path.join(outDir, '*')))
df = pd.DataFrame()
for outFile in outList:
  print(outFile, end='\r')
  df = pd.concat([df, readSPARTACUS(outFile)], sort=False)

df.to_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/sodankyla.spartacus.pkl')
