import pdb
import numpy as np
import xarray as xr
import rioxarray as rx
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['savefig.dpi'] = 600
import matplotlib.pyplot as plt
from pyproj import CRS, Proj, Transformer
MODProj = Proj('+proj=sinu +R=6371007.181')
trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("32635"),
                             always_xy=True)

xMin, xMax = 1119500., 1157600.
yMin, yMax = 7476700., 7506400.

als = xr.Dataset()
als['chm'] = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/finland.ndsm.modis.5m.tif')
als['cv'] = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/finland.canopy_cover.modis.tif')*100.
als = als.where((als.chm >=0)&(als.chm<=30)&(als.cv>=0)&(als.cv<=100)).isel(band=0)
#als = als.sel(y=slice(yMax,yMin), x=slice(xMin, xMax))
X, Y = np.meshgrid(als.x, als.y)
lon, lat = trans.transform(X, Y)
als['lon'], als['lat'] = (('y', 'x'), lon), (('y', 'x'), lat)

alsMask = (np.isnan(als.chm)|np.isnan(als.cv))==False
idxMask = np.argwhere(alsMask.values)
maxIDX = np.full(als.y.shape[0], -1, dtype=int)
minIDX = np.full(als.y.shape[0], -1, dtype=int)
for i in range(als.y.shape[0]):
  if i in idxMask[:,0]:
    minIDX[i] = np.min(idxMask[idxMask[:,0]==i][:,1])
    maxIDX[i] = np.max(idxMask[idxMask[:,0]==i][:,1])
  elif i != 0:
    iBack = i
    iFor = i
    southMin, southMax, sDist = 0,0,1e12
    northMin, northMax, nDist = 0,0,1e12
    while iBack > 0:
      iBack -= 1
      if iBack in idxMask[:,0]:
        northMin = np.min(idxMask[idxMask[:,0]==iBack][:,1])
        northMax = np.min(idxMask[idxMask[:,0]==iBack][:,1])
        nDist = i-iBack
        iBack = 0
    while iFor < als.y.shape[0]:
      iFor += 1
      if iFor in idxMask[:,0]:
        southMin = np.min(idxMask[idxMask[:,0]==iFor][:,1])
        southMax = np.max(idxMask[idxMask[:,0]==iFor][:,1])
        sDist = iFor-i
        iFor = als.y.shape[0]
    minIDX[i] = int(np.round(np.average([northMin, southMin], weights=[1./nDist, 1./sDist])))
    maxIDX[i] = int(np.round(np.average([northMax, southMax], weights=[1./nDist, 1./sDist])))
  else:
    pass

iceMask = np.full(alsMask.shape, False)
for i in range(als.y.shape[0]):
  if (minIDX[i]>-1) & (maxIDX[i]>-1):
    iceMask[i][minIDX[i]:maxIDX[i]+1] = True

#als5m = xr.Dataset()
#als5m['chm'] = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/ndsm_5m.tif')
#als5m['cv'] = rx.open_rasterio('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/canopy_cover_5m.tif')
#als5m = als5m.where((als5m.chm >=0)&(als5m.chm<=50)&(als5m.cv>=0)&(als5m.cv<=100)).isel(band=0)

atl08 = xr.open_dataset('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/icesat2.atl08.finland.nc')
#atl08 = atl08.sel(y=slice(yMax,yMin), x=slice(xMin, xMax))
atl08 = atl08.where(iceMask)
X, Y = np.meshgrid(atl08.x, atl08.y)
lon, lat = trans.transform(X, Y)
atl08['lon'], atl08['lat'] = (('y', 'x'), lon), (('y', 'x'), lat)

diff = xr.Dataset()
diff.coords['x'] = als.x
diff.coords['y'] = als.y
diff['chm'] = (('y', 'x'), atl08.sel(y=als.y, x=als.x, method='nearest').chm.values - als.chm.values)
diff['cv'] = (('y', 'x'), atl08.sel(y=als.y, x=als.x, method='nearest').cv.values - als.cv.values)
X, Y = np.meshgrid(diff.x, diff.y)
lon, lat = trans.transform(X, Y)
diff['lon'], diff['lat'] = (('y', 'x'), lon), (('y', 'x'), lat)


"""
##
## Canopy height maps
##
mpl.rcParams['axes.facecolor'] = 'k'
fig, ax = plt.subplots(3,1, figsize=(3,8))
ax[0].set_ylabel('Northing (km)')
ax[0].set_xlabel('Easting (km)')
ax[0].text(-0.35, 0.5, 'ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0].transAxes)
c = ax[0].pcolormesh(als.lon/1000, als.lat/1000, als.chm, vmin=0, vmax=30, cmap='Greens',rasterized=True)
plt.colorbar(c, ax=ax[0], label='Canopy height (m)')
ax[1].text(-0.35, 0.5, 'ATL08', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0].transAxes)
c = ax[1].pcolormesh(atl08.lon/1000, atl08.lat/1000, atl08.chm, vmin=0, vmax=30, cmap='Greens',rasterized=True)
ax[1].set_xlabel('Easting (km)')
ax[1].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[1], label='Canopy height (m)')
ax[2].text(-0.35, 0.5, 'ATL08 - ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0].transAxes)
c = ax[2].pcolormesh(diff.lon/1000, diff.lat/1000, diff.chm, vmin=-10, vmax=10, cmap='PiYG',rasterized=True)
ax[2].set_xlabel('Easting (km)')
ax[2].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[2], label='Canopy height difference (m)')
for i in range(3):
    ax[i].set_aspect('equal')
    #lonMin, lonMax= 466000, 498000
    #latMin, latMax= 7456000, 7490000
    #ax[i].set_xlim(lonMin, lonMax)
    #ax[i].set_ylim(latMin, latMax)
    ax[i].set_xlim(atl08.lon.where(iceMask).min()/1000, (atl08.lon.where(iceMask).max()+500)/1000)
    ax[i].set_ylim(atl08.lat.where(iceMask).min()/1000, (atl08.lat.where(iceMask).max()+500)/1000)
    ax[i].ticklabel_format(style='plain')

fig.tight_layout()
#fig.show()
fig.savefig('final_plots/ATL08vALS.height.finland.pdf')

##
## Canopy cover maps
##
mpl.rcParams['axes.facecolor'] = 'k'
fig, ax = plt.subplots(3,1,figsize=(3,8))
c = ax[0].pcolormesh(als.lon/1000, als.lat/1000, als.cv, vmin=0, vmax=100, cmap='Greens',rasterized=True)
plt.colorbar(c, ax=ax[0], label='Canopy cover (%)')
ax[0].text(-0.35, 0.5, 'ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0].transAxes)
ax[0].set_ylabel('Northing (km)')
ax[0].set_xlabel('Easting (km)')
c = ax[1].pcolormesh(atl08.lon/1000, atl08.lat/1000, atl08.cv, vmin=0, vmax=100, cmap='Greens',rasterized=True)
ax[1].text(-0.35, 0.5, 'ATL08', ha='center', va='center', rotation=90, fontsize='large', transform=ax[1].transAxes)
ax[1].set_xlabel('Easting (km)')
ax[1].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[1], label='Canopy cover (%)')
ax[2].text(-0.35, 0.5, 'ATL08 - ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0].transAxes)
c = ax[2].pcolormesh(diff.lon/1000, diff.lat/1000, diff.cv, vmin=-25, vmax=25, cmap='PiYG',rasterized=True)
ax[2].set_xlabel('Easting (km)')
ax[2].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[2], label='Canopy cover difference (%)')
for i in range(3):
    ax[i].set_aspect('equal')
    #lonMin, lonMax= 466000, 498000
    #latMin, latMax= 7456000, 7490000
    #ax[i].set_xlim(lonMin, lonMax)
    #ax[i].set_ylim(latMin, latMax)
    ax[i].set_xlim(atl08.lon.where(iceMask).min()/1000, (atl08.lon.where(iceMask).max()+500)/1000)
    ax[i].set_ylim(atl08.lat.where(iceMask).min()/1000, (atl08.lat.where(iceMask).max()+500)/1000)
    ax[i].set_aspect('equal')
    ax[i].ticklabel_format(style='plain')

fig.tight_layout()
#fig.show()
fig.savefig('final_plots/ATL08vALS.cover.finland.pdf')
"""

## Combined maps
mpl.rcParams['axes.facecolor'] = 'k'
fig, ax = plt.subplots(3,2, figsize=(6,8), sharex=True, sharey=True)
ax[0,0].set_ylabel('Northing (km)')
#ax[0,0].set_xlabel('Easting (km)')
ax[0,0].text(-0.7, 0.5, 'ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0,0].transAxes)
c = ax[0,0].pcolormesh(als.lon/1000, als.lat/1000, als.chm, vmin=0, vmax=30, cmap='Greens',rasterized=True)
plt.colorbar(c, ax=ax[0,0], label='Canopy height (m)')
ax[1,0].text(-0.7, 0.5, 'ATL08', ha='center', va='center', rotation=90, fontsize='large', transform=ax[1,0].transAxes)
c = ax[1,0].pcolormesh(atl08.lon/1000, atl08.lat/1000, atl08.chm, vmin=0, vmax=30, cmap='Greens',rasterized=True)
#ax[1,0].set_xlabel('Easting (km)')
ax[1,0].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[1,0], label='Canopy height (m)')
ax[2,0].text(-0.7, 0.5, 'ATL08 - ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[2,0].transAxes)
c = ax[2,0].pcolormesh(diff.lon/1000, diff.lat/1000, diff.chm, vmin=-10, vmax=10, cmap='PiYG',rasterized=True)
ax[2,0].set_xlabel('Easting (km)')
ax[2,0].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[2,0], label='Canopy height difference (m)')
c = ax[0,1].pcolormesh(als.lon/1000, als.lat/1000, als.cv, vmin=0, vmax=100, cmap='Greens',rasterized=True)
plt.colorbar(c, ax=ax[0,1], label='Canopy cover (%)')
#ax[0,1].set_ylabel('Northing (km)')
#ax[0,1].set_xlabel('Easting (km)')
c = ax[1,1].pcolormesh(atl08.lon/1000, atl08.lat/1000, atl08.cv, vmin=0, vmax=100, cmap='Greens',rasterized=True)
#ax[1,1].set_xlabel('Easting (km)')
#ax[1,1].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[1,1], label='Canopy cover (%)')
c = ax[2,1].pcolormesh(diff.lon/1000, diff.lat/1000, diff.cv, vmin=-25, vmax=25, cmap='PiYG',rasterized=True)
ax[2,1].set_xlabel('Easting (km)')
#ax[2,1].set_ylabel('Northing (km)')
plt.colorbar(c, ax=ax[2,1], label='Canopy cover difference (%)')
for i in range(3):
  for j in range(2):
    ax[i,j].set_aspect('equal')
    #lonMin, lonMax= 466000, 498000
    #latMin, latMax= 7456000, 7490000
    #ax[i].set_xlim(lonMin, lonMax)
    #ax[i].set_ylim(latMin, latMax)
    ax[i,j].set_xlim(atl08.lon.where(iceMask).min()/1000, (atl08.lon.where(iceMask).max()+500)/1000)
    ax[i,j].set_ylim(atl08.lat.where(iceMask).min()/1000, (atl08.lat.where(iceMask).max()+500)/1000)
    ax[i,j].ticklabel_format(style='plain')

fig.tight_layout()
fig.savefig('final_plots/ATL08vALS.combined.finland.pdf')

##
##Height and cover distributions
##
mpl.rcParams['axes.facecolor'] = 'w'
fig, ax = plt.subplots(3,2, figsize=(6,4))
ax[0,0].text(-0.4, 0.5, 'ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[0,0].transAxes)
ax[0,0].set_xlabel('Canopy height (m)')
ax[0,0].set_xlim(0,30)
ax[0,0].hist(als.chm.values.flatten(), bins=np.arange(0, 31, 1), color='#208020')
ax[1,0].text(-0.4, 0.5, 'ATL08', ha='center', va='center', rotation=90, fontsize='large', transform=ax[1,0].transAxes)
ax[1,0].set_xlabel('Canopy height (m)')
ax[1,0].set_xlim(0,30)
ax[1,0].hist(atl08.chm.values.flatten(), bins=np.arange(0,31,1), color='#208020')
ax[2,0].text(-0.4, 0.5, 'ATL08 - ALS', ha='center', va='center', rotation=90, fontsize='large', transform=ax[2,0].transAxes)
ax[2,0].set_xlabel('Canopy height difference (m)')
ax[2,0].set_xlim(-15,15)
ax[2,0].hist(diff.chm.values.flatten(), bins=np.arange(-15,15.1, 1), color='#802080')
ax[0,1].set_xlabel('Canopy cover (%)')
ax[0,1].set_xlim(0,100)
ax[0,1].hist(als.cv.values.flatten(), bins=np.arange(0, 101, 5), color='#208020')
ax[1,1].set_xlabel('Canopy cover (%)')
ax[1,1].set_xlim(0,100)
ax[1,1].hist(atl08.cv.values.flatten(), bins=np.arange(0, 101,5), color='#208020')
ax[2,1].set_xlabel('Canopy cover difference (%)')
ax[2,1].set_xlim(-50,50)
ax[2,1].hist(diff.cv.values.flatten(), bins=np.arange(-50,51,5), color='#802080')
for i in range(3):
  ax[i,0].text(-0.3, 0.5, 'Number of\npixels', ha='center', va='center', rotation=90, transform=ax[i,0].transAxes)
  yMax = 1e4*(1+np.round(np.max([ax[i,0].get_ylim(), ax[i,1].get_ylim()])/1e4))
  ax[i,0].set_ylim(0, yMax); ax[i,1].set_ylim(0, yMax)
  ax[i,1].set_yticklabels([])

fig.tight_layout()
fig.show()
fig.savefig('final_plots/ATL08vALS.distributions.finland.pdf')

mpl.rcParams['axes.facecolor'] = 'w'
fig, ax = plt.subplots(1,1, figsize=(6,8))
ax.pcolormesh(als.lon/1000, als.lat/1000, als.where())
