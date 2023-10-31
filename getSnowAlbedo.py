import glob, os, datetime
import numpy as np
import pandas as pd
import rioxarray as rx

logFile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/sapickle/evergreen.needleleaf.snow.albedo.csv'
lcList = np.sort(glob.glob('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MCD12Q1/*.hdf'))
lcYear = [lcFile.split('.')[1][1:5] for lcFile in lcList]
snowList = np.sort(glob.glob('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MOD10A1/*.hdf'))

pd.DataFrame({'date':[], 'sc':[], 'sa':[]}).to_csv(logFile, index=False)
for snowFile in snowList:
  date = snowFile.split('.')[1][1:]
  print(date, end='\r')
  snow = rx.open_rasterio(snowFile).isel(band=0).drop(['spatial_ref', 'band'])
  snow.coords['date'] = datetime.datetime.strptime(date, '%Y%j')
  snow = snow.expand_dims({'date':1})
  lcFile = lcList[np.int32(date[:4])==np.int32(lcYear)][0]
  lc = rx.open_rasterio(lcFile).isel(band=0).drop(['spatial_ref', 'band'])
  snow['PFT'] = lc.LC_Type5.astype(int)
  df = pd.DataFrame({'date': np.full(2400*2400, date),
                     'pft': snow.PFT.values.flatten().astype(int),
                     'sc': snow.NDSI_Snow_Cover.values.flatten().astype(int),
                     'sa': snow.Snow_Albedo_Daily_Tile.values.flatten().astype(int)})
  df = df.where((df.pft==1)&(df.sc<=100)&(df.sa<=100)).dropna()
  df['sc'] = df.sc.astype(int)
  df['sa'] = df.sa.astype(int)
  df[['date', 'sc', 'sa']].to_csv(logFile, mode='a', index=False, header=False)
