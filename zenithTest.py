import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset('test.in.nc')

def testZenith(ds, mu):
  ds['cos_solar_zenith_angle'] = (('ind'), np.full(ds.cos_solar_zenith_angle.shape, mu))
  ds.to_netcdf('in.nc', engine='scipy', format='NETCDF3_CLASSIC')
  os.system(' '.join(['spartacus_surface', 'config.nam', 'in.nc', 'out.nc']))
  ds = xr.open_dataset('out.nc')
  return float(ds.flux_up_layer_top_sw.mean()), float(ds.flux_up_layer_top_sw.std())


muArr = np.arange(0.01, 1., 0.01)
meanArr = np.full(muArr.shape, np.nan)
stdArr = np.full(muArr.shape, np.nan)

for i in range(muArr.shape[0]):
  try:
    meanArr[i], stdArr[i] = testZenith(ds, muArr[i])
  except:
    pass

plt.plot(np.rad2deg(np.arccos(muArr)), meanArr)
plt.xlabel('Solar zenith angle (°)')
plt.ylabel('Mean diffuse VIS albedo')
plt.savefig('zenithTest/zen.SPARTACUS_WSA_vis.pdf')
