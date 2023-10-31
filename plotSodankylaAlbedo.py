import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

df = pd.read_csv('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/SodankylaAlbedoTimeseries_IOA0008_2012-01-01_2023-08-31.txt', header=1, names=['station', 'datetime', 'downwelling', 'upwelling'])
df['downwelling'] = df.downwelling.where(df.downwelling>0.)
df['upwelling'] = df.upwelling.where(df.upwelling>0.)
df['albedo'] = df.upwelling/df.downwelling
df['albedo'] = df.albedo.where((df.albedo>0.)&(df.albedo<1.))
df['datetime'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S+00') for d in df.datetime]
df['date'] = [datetime.datetime(d.year,d.month,d.day) for d in df.datetime]
df.set_index('datetime', inplace=True)

fig, ax = plt.subplots(1,1,figsize=(9,4))
df.albedo.plot(color=cmap(.2), alpha=.1, label='10 minutes', lw=1)
df.groupby('date').albedo.mean().plot(ax=ax, c=cmap(.2), alpha=.5, label='1 day', lw=1)
df.groupby('date').albedo.mean().rolling(window=30, center=True, min_periods=5).mean().interpolate('linear').plot(ax=ax, c=cmap(.2), alpha=1., label='30 days', lw=1)
ax.set_xlabel('Date')
ax.set_ylabel('Subcanopy albedo')
ax.set_ylim(0,1)
ax.set_xlim(datetime.datetime(2012,1,1), datetime.datetime(2023,12,31))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.tick_params(axis='x', which='minor', length=2)
ax.tick_params(axis='x', which='major', length=5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='left')#, va='center')
fig.legend(loc='upper center', edgecolor='none', facecolor='none', ncol=3)
fig.tight_layout(rect=(0,0,1,.95))
fig.show()
