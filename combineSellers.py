import glob, os
import pandas as pd
import xarray as xr

ncDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ncFinal'
ncList = sorted(glob.glob(os.path.join(ncDir, '*.nc')))

df = pd.DataFrame()
for nc in ncList:
  print(nc)
  df = pd.concat([df, xr.open_dataset(nc).to_dataframe().dropna()])

df.to_pickle(os.path.join(ncDir, '../sodankyla.final.pkl'))
