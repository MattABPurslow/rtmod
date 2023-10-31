import numpy as np
import rioxarray as rx
import xarray as xr
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.facecolor'] = 'k'
import matplotlib.pyplot as plt

laiFile = '/exports/csce/datastore/geos/users/s1503751/MODIS/global/lai.merged.tif'
chmFile = '/exports/csce/datastore/geos/users/s1503751/MODIS/global/chm.merged.tif'

lai = rx.open_rasterio(laiFile).isel(band=0).rename('lai')
chm = rx.open_rasterio(chmFile).isel(band=0).rename('chm') 

fig, ax = plt.subplots(1,1,figsize=(6,8/3.))
lai.plot.pcolormesh(ax=ax, vmin=0, vmax=5, cmap='Greens', cbar_kwargs={'label': 'Leaf Area Index (m²m⁻²)'}, rasterized=True)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
fig.tight_layout()
fig.show()
fig.savefig('global.figure.pdf', dpi=600)

chm.plot.pcolormesh(ax=ax, vmin=0, vmax=50, cmap='Greens', cbar_kwargs={'label': 'Canopy height (m)'}, rasterized=True)
chm.where((lai>=1)&(lai<=3)).plot.pcolormesh(ax=ax, vmin=0, vmax=50, cmap='Greens', cbar_kwargs={'label': 'Canopy height (m)'}, rasterized=True)
fig.tight_layout()
fig.savefig('global.figure.pdf', dpi=600)
