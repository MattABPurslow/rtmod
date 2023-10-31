import glob, os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

df = pd.read_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/sodankyla.final.pkl')

for wv in ['vis', 'nir']:
  for a in ['BSA', 'WSA']:
    for rt in ['Sellers', 'SPARTACUS']:
      df['%s_%s_%s_diff' % (rt, a, wv)] = df['%s_%s_%s' % (rt, a, wv)] - \
                                          df['MODIS_%s_%s' % (a, wv)]

df = df.where((df.veg_fraction>0) & (df.zen>0) & (df.zen<=90)).dropna()

corr = df.corr()
plt.figure(figsize=(9,9))
cm = plt.pcolormesh(corr.columns, corr.columns, corr.values, vmin=-1, vmax=1, cmap='bwr')
plt.colorbar(cm, label="Pearson's R")
plt.gca().set_xticklabels(corr.columns, rotation=-90, va="top")
plt.tight_layout()
#plt.show()
plt.savefig('plots/corr.pdf')
plt.close()

def plotRelationship(df, x, y):
  print('plotting', y, 'against', x, '@', datetime.datetime.now())
  fig, ax = plt.subplots(1,1, figsize=(9,9))
  sns.histplot(df, x=x, y=y, ax=ax)
  res = linregress(df[x], df[y])
  xArr = np.array([df[x].min(), df[x].max()])
  ax.plot(xArr, res.slope*xArr+res.intercept, label='lstsq', c='k')
  ax.legend(loc='best', title='R=%.3f' % res.rvalue,
            edgecolor='none', facecolor='none')
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  fig.tight_layout()
  fig.show()
  fig.savefig('plots/%s.%s.pdf' % (x, y))
  plt.close()

depVars = ['BSA_vis_diff', 'BSA_nir_diff', 'WSA_vis_diff', 'WSA_nir_diff']
indVars = sorted([c for c in df.columns if c not in depVars])[::-1]
modVars = [c for c in indVars if (c[:5]=='MODIS')&((c[-3:]=='nir')|(c[-3:]=='vis'))]
selVars = [c for c in indVars if (c[:5]=='Selle')&((c[-3:]=='nir')|(c[-3:]=='vis'))]

for x in indVars:
  if df[x].max()!=df[x].min():
    for y in depVars:
      plotRelationship(df, x, y)

for x in modVars:
  if df[x].max()!=df[x].min():
    for y in selVars:
      plotRelationship(df, x, y)
