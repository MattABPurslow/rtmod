import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

df = pd.read_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/rt.als.finland.2m.pkl')
df = df.groupby(['x','y']).mean()
df['latBin'] = np.round(df.lat, decimals=1)

mean = df.groupby('latBin').mean().sort_index()
q25 = df.groupby('latBin').quantile(0.25).sort_index()
q75 = df.groupby('latBin').quantile(0.75).sort_index()

fig, ax = plt.subplots(2,1,figsize=(6,4), sharex=True)
ax[0].fill_between(q25.index, q25.MODIS_LAI_eff, q75.MODIS_LAI_eff, color=cmap(0.5), alpha=0.5, edgecolor='none')
ax[0].plot(mean.index, mean.MODIS_LAI_eff, color=cmap(0.5))
ax[0].set_ylabel('MODIS Effective LAI (m²m⁻²)')#, color=cmap(0.5))
ax[0].set_ylim(0, 4)
ax[0].set_xlim(60,df.latBin.max())
#plt.setp(ax[0].get_yticklabels(), color=cmap(0.5))

ax[1].fill_between(q25.index, q25.height, q75.height, color=cmap(0.2), alpha=0.5, edgecolor='none')
ax[1].plot(mean.index, mean.height, color=cmap(0.2))
ax[1].set_ylabel('Canopy height (m)')#, color=cmap(0.2))
#plt.setp(ax[1].get_yticklabels(), color=cmap(0.2))
ax[1].set_ylim(0, 30)
ax[1].set_xlim(60,df.latBin.max())

ax[1].set_xlabel('Latitude (°)')
fig.tight_layout()
fig.savefig('LatitudeGradient.pdf')


