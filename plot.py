import glob, os, pdb
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

from matplotlib.colors import Normalize
from matplotlib.cm import viridis as cmap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

def readSPARTACUS(outFile):
  ds = xr.open_dataset(outFile.replace('spartacusOut', 'spartacusIn')).isel(layer=-1, layer_int=-1)
  labLoaded = []
  for df in ['BSA', 'WSA']:
    for wv in ['nir', 'vis', 'shortwave']:
      lab = '%s_%s' % (df, wv)
      if (os.path.exists(outFile.replace('BSA_nir', lab))):
        out = xr.open_dataset(outFile.replace('BSA_nir', lab)).isel(layer=-1, layer_interface=-1)
        ds['SPARTACUS_%s' % lab] = out.flux_up_layer_top_sw
        labLoaded.append(lab)
  ds['mask'] = (ds.MODIS_NDSI_Snow_Cover>0.) &\
               (ds.PFT_LC==1) &\
               (ds.omega>0.) &\
               (ds.MODIS_LAI>0.) &\
               (ds.zen<90.) &\
               (ds.MODIS_Snow_Albedo>0.)
  for lab in labLoaded:
      ds['mask'] = ds.mask &\
                   (ds['MODIS_%s' % lab] > 0.)&(ds['MODIS_%s' % lab]<1.)&\
                   (ds['Sellers_%s' % lab] > 0.)&(ds['Sellers_%s' % lab]<1.)&\
                   (ds['SPARTACUS_%s' % lab] > 0.)&(ds['SPARTACUS_%s' % lab]<1.)
  return ds.where(ds.mask).to_dataframe().reset_index().dropna()
  #return ds[['zen', 'MODIS_%s' % wv, 'SPARTACUS_%s' % wv, 'Sellers_%s' % wv]].where(ds.mask).to_dataframe().dropna()

def plotRelationship(df, x, y):
  print('plotting', y, 'against', x, '@', datetime.datetime.now())
  fig, ax = plt.subplots(1,1, figsize=(9,9))
  dfi = df[[x,y]].where(df[[x,y]]>-1).dropna()
  sns.histplot(dfi, x=x, y=y, ax=ax, binwidth=(0.1, 0.01))
  #plt.scatter(dfi[x], dfi[y], s=1, alpha=0.05)
  dfi['xBin'] = np.round(dfi[x])
  group = dfi[['xBin',y]].groupby('xBin').mean()
  ax.plot(group.index, group, label='Mean', c='r')
  if dfi[x].min()<dfi[x].max():
    res = linregress(dfi[x], dfi[y])
    xArr = np.array([dfi[x].min(), dfi[x].max()])
    ax.plot(xArr, res.slope*xArr+res.intercept, label='lstsq', c='k')
    ax.legend(loc='best', title='R=%.3f' % res.rvalue,
              edgecolor='none', facecolor='none')
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  fig.tight_layout()
  #fig.show()
  fig.savefig('final_plots/icesat2.Sodankyla_Albedo.%s.%s.beforeMay.pdf' % (x, y))
  plt.close()

def plotRelationshipReduced(df, x, y, reducer):
  print('plotting', y, 'against', x, '@', datetime.datetime.now())
  fig, ax = plt.subplots(1,1, figsize=(9,9))
  dfi = df[[x,y]].where(df[[x,y]]>-1).dropna()
  sns.histplot(dfi, x=x, y=y, ax=ax, binwidth=(0.1, 0.01))
  #plt.scatter(dfi[x], dfi[y], s=1, alpha=0.05)
  dfi['xBin'] = np.round(dfi[x])
  group = dfi[['xBin',y]].groupby('xBin').mean()
  ax.plot(group.index, group, label='Mean', c='r')
  if dfi[x].min()<dfi[x].max():
          res = linregress(dfi[x], dfi[y])
          xArr = np.array([dfi[x].min(), dfi[x].max()])
          ax.plot(xArr, res.slope*xArr+res.intercept, label='lstsq', c='k')
          ax.legend(loc='best', title='R=%.3f' % res.rvalue,
                    edgecolor='none', facecolor='none')
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  fig.tight_layout()
  #fig.show()
  fig.savefig('final_plots/%s.%s.%s.pdf' % (x, y, reducer))
  plt.close()

if __name__=='__main__':
  outDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacusOut'
  outList = sorted(glob.glob(os.path.join(outDir, '*BSA_nir.nc')))
  df = pd.DataFrame()
  cnt = 1
  for outFile in outList:
    print(cnt, '/', len(outList), end='\r'); cnt+=1
    df = pd.concat([df, readSPARTACUS(outFile)], ignore_index=True)
  df.to_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/rt.icesat2.atl08.pkl')
  """
  df = pd.read_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/rt.als.pkl')
  df['month'] = [d.month for d in df.date]
  df = df.loc[(df.month<5)]
  #df = df.loc[(df.MODIS_NDSI_Snow_Cover>0.75)]
  #df = df.loc[(df.month<5)&(df.MODIS_NDSI_Snow_Cover>0.75)]
  for wv in ['vis', 'nir', 'shortwave']:
    for sa in ['BSA', 'WSA']:
      for rt in ['Sellers', 'SPARTACUS']:
        lab = '%s_%s' % (sa, wv)
        df['%s_%s_diff' % (rt, lab)] = df['%s_%s' % (rt, lab)] -\
                                       df['MODIS_%s' % lab]

  ##
  ## MODIS Snow Albedo as function of LAI & canopy cover / height
  cvStep = 0.1
  laiStep = 0.5
  df['veg_fraction_bin'] = np.round(df.veg_fraction/cvStep)*cvStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'veg_fraction_bin'])
  cvArr = np.arange(df.veg_fraction_bin.min(), df.veg_fraction_bin.max(), cvStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
  fig, ax = plt.subplots(1,1,figsize=(16,9))
  albStr = 'MODIS_Snow_Albedo'
  mArr = np.full(laiArr.shape[0], np.nan)
  cArr = np.full(laiArr.shape[0], np.nan)
  pArr = np.full(laiArr.shape[0], np.nan)
  rArr = np.full(laiArr.shape[0], np.nan)
  for i in range(len(laiArr)):
    dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
    result = linregress(dfi['veg_fraction'], dfi[albStr])
    mArr[i] = result.slope
    cArr[i] = result.intercept
    pArr[i] = result.pvalue
    rArr[i] = result.rvalue
  ax.plot(laiArr, mArr, c='g')
  ax.set_ylim(-0.05, 0.05)
  ax.axhline(0, lw=0.5, c='k')
  ax2 = ax.twinx()
  ax2.plot(laiArr, rArr, c='r')
  ax2.set_ylim(-1,1)
  ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
  ax.set_ylabel('Gradient', color='g')
  plt.setp(ax.get_yticklabels(), color='g')
  ax2.set_ylabel('R-value', color='r')
  plt.setp(ax2.get_yticklabels(), color='r')
  fig.tight_layout()
  fig.savefig('final_plots/modis_snow_albedo.coverdependence.pdf')
  plt.close()
  chmStep = 5.
  laiStep = 0.5
  df['height_bin'] = np.round(df.height/chmStep)*chmStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'height_bin'])
  chmArr = np.arange(df.height_bin.min(), df.height_bin.max(), chmStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
  fig, ax = plt.subplots(1,1,figsize=(16,9))
  albStr = 'MODIS_Snow_Albedo'
  mArr = np.full(laiArr.shape[0], np.nan)
  cArr = np.full(laiArr.shape[0], np.nan)
  pArr = np.full(laiArr.shape[0], np.nan)
  rArr = np.full(laiArr.shape[0], np.nan)
  for i in range(len(laiArr)):
    dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
    result = linregress(dfi['height'], dfi[albStr])
    mArr[i] = result.slope
    cArr[i] = result.intercept
    pArr[i] = result.pvalue
    rArr[i] = result.rvalue
  ax.plot(laiArr, mArr, c='g')
  ax.set_ylim(-0.05, 0.05)
  ax.axhline(0, lw=0.5, c='k')
  ax2 = ax.twinx()
  ax2.plot(laiArr, rArr, c='r')
  ax2.set_ylim(-1,1)
  ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
  ax.set_ylabel('Gradient (m⁻¹)', color='g')
  plt.setp(ax.get_yticklabels(), color='g')
  ax2.set_ylabel('R-value', color='r')
  plt.setp(ax2.get_yticklabels(), color='r')
  fig.tight_layout()
  fig.savefig('final_plots/modis_snow_albedo.heightdependence.pdf')
  plt.close()

  ##
  ## Linear regression of albedo against height for LAI bins
  ##
  chmStep = 5.
  laiStep = 0.5
  df['height_bin'] = np.round(df.height/chmStep)*chmStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'height_bin'])
  chmArr = np.arange(df.height_bin.min(), df.height_bin.max(), chmStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  wv = ['vis', 'nir', 'shortwave']
  sa = ['BSA', 'WSA']
  fig = plt.figure(figsize=(16,9))
  gs = GridSpec(2,13)
  ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,12,4)] for i in range(2)]
  for si in range(len(sa)):
   for wj in range(len(wv)):
    albStr = 'MODIS_%s_%s' % (sa[si], wv[wj])
    mArr = np.full(laiArr.shape[0], np.nan)
    cArr = np.full(laiArr.shape[0], np.nan)
    pArr = np.full(laiArr.shape[0], np.nan)
    rArr = np.full(laiArr.shape[0], np.nan)
    for i in range(len(laiArr)):
      dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
      result = linregress(dfi['height'], dfi[albStr])
      mArr[i] = result.slope
      cArr[i] = result.intercept
      pArr[i] = result.pvalue
      rArr[i] = result.rvalue
    ax[si][wj].plot(laiArr, mArr, c='g')
    ax[si][wj].set_ylim(-0.05, 0.05)
    ax[si][wj].axhline(0, lw=0.5, c='k')
    ax2 = ax[si][wj].twinx()
    ax2.plot(laiArr, rArr, c='r')
    ax2.set_ylim(-1,1)
    if si==0:
      ax[si][wj].set_title(wv[wj])
    if si==len(sa)-1:
      ax[si][wj].set_xlabel('MODIS Effective LAI (m²m⁻²)')
    if wj==0:
      ax[si][wj].set_ylabel('%s\nGradient (m⁻¹)' % sa[si], color='g')
      plt.setp(ax[si][wj].get_yticklabels(), color='g')
    if wj!=0:
      plt.setp(ax[si][wj].get_yticklabels(), visible=False)
    if wj==len(wv)-1:
      ax2.set_ylabel('R-value', color='r')
      plt.setp(ax2.get_yticklabels(), color='r')
    else:
      plt.setp(ax2.get_yticklabels(), visible=False)
  fig.tight_layout()
  fig.savefig('final_plots/modis.heightdependence.lsm_snow_albedo.pdf')
  plt.close()
  ##
  ## MODIS Snow Albedo as function of LAI & canopy cover / height
  cvStep = 0.1
  laiStep = 0.5
  df['veg_fraction_bin'] = np.round(df.veg_fraction/cvStep)*cvStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'veg_fraction_bin'])
  cvArr = np.arange(df.veg_fraction_bin.min(), df.veg_fraction_bin.max(), cvStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
  wv = ['vis', 'nir', 'shortwave']
  sa = ['BSA', 'WSA']
  fig, ax = plt.subplots(1,1,figsize=(16,9))
  albStr = 'MODIS_NDSI_Snow_Cover'
  alb = group[albStr].mean()
  for i in range(len(laiArr)):
    for j in range(len(cvArr)):
      if (laiArr[i], cvArr[j]) in alb.index:
        albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
  c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                    np.append(cvArr, cvArr[-1]+cvStep)*100,
                    albArr.T, vmin=0, vmax=1)
  ax.set_ylabel('Canopy cover (%)')
  ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, ax=ax, label='Mean MODIS snow cover')
  fig.tight_layout()
  fig.savefig('final_plots/coverVefflai.MODIS_NDSI_Snow_Cover.mean.pdf')
  plt.close()
  chmStep = 5.
  laiStep = 0.5
  df['height_bin'] = np.round(df.height/chmStep)*chmStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'height_bin'])
  chmArr = np.arange(df.height_bin.min(), df.height_bin.max(), chmStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
  fig, ax = plt.subplots(1,1,figsize=(16,9))
  albStr = 'MODIS_NDSI_Snow_Cover'
  alb = group[albStr].mean()
  for i in range(len(laiArr)):
    for j in range(len(chmArr)):
      if (laiArr[i], chmArr[j]) in alb.index:
        albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
  c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                    np.append(chmArr, chmArr[-1]+chmStep),
                    albArr.T, vmin=0, vmax=1)
  ax.set_ylabel('Canopy height (m)')
  ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, ax=ax, label='Mean MODIS snow cover')
  fig.tight_layout()
  fig.savefig('final_plots/heightVefflai.MODIS_NDSI_Snow_Cover.mean.pdf')
  plt.close()
  ##
  ## MODIS albedo as function of LAI & canopy cover
  ##
  cvStep = 0.1
  laiStep = 0.5
  df['veg_fraction_bin'] = np.round(df.veg_fraction/cvStep)*cvStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'veg_fraction_bin'])
  cvArr = np.arange(df.veg_fraction_bin.min(), df.veg_fraction_bin.max(), cvStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  wv = ['vis', 'nir', 'shortwave']
  sa = ['BSA', 'WSA']
  fig = plt.figure(figsize=(16,9))
  gs = GridSpec(2,13)
  ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,12,4)] for i in range(2)]
  for si in range(len(sa)):
   for wj in range(len(wv)):
    albStr = 'SPARTACUS_%s_%s' % (sa[si], wv[wj])
    alb = group[albStr].mean()
    albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
    for i in range(len(laiArr)):
      for j in range(len(cvArr)):
        if (laiArr[i], cvArr[j]) in alb.index:
          albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
    c = ax[si][wj].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                              np.append(cvArr, cvArr[-1]+cvStep)*100,
                              albArr.T, vmin=0, vmax=1)
    if wj == 0:
      ax[si][wj].set_ylabel('%s\nCanopy cover (%s)' % (sa[si], '%'))
    else:
      plt.setp(ax[si][wj].get_yticklabels(), visible=False)
    if si == 0:
      ax[si][wj].set_title('%s' % wv[wj])
      plt.setp(ax[si][wj].get_xticklabels(), visible=False)
    if si == 1:
      ax[si][wj].set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='Mean SPARTACUS albedo')
  fig.tight_layout()
  fig.savefig('final_plots/coverVefflai.LSM_SPARTACUS.mean.pdf')
  plt.close()
  #fig.show()
  fig = plt.figure(figsize=(16,9))
  gs = GridSpec(2,13)
  ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,12,4)] for i in range(2)]
  for si in range(len(sa)):
   for wj in range(len(wv)):
    albStr = 'SPARTACUS_%s_%s' % (sa[si], wv[wj])
    alb = group[albStr].count()
    albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
    for i in range(len(laiArr)):
      for j in range(len(cvArr)):
        if (laiArr[i], cvArr[j]) in alb.index:
          albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
    c = ax[si][wj].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                              np.append(cvArr, cvArr[-1]+cvStep)*100., 
                              albArr.T, vmin=0, vmax=100)
    if wj == 0:
      ax[si][wj].set_ylabel('%s\nCanopy cover (%s)' % (sa[si], '%'))
    else:
      plt.setp(ax[si][wj].get_yticklabels(), visible=False)
    if si == 0:
      ax[si][wj].set_title('%s' % wv[wj])
      plt.setp(ax[si][wj].get_xticklabels(), visible=False)
    if si == 1:
      ax[si][wj].set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='# pixels')
  fig.tight_layout()
  fig.savefig('final_plots/coverVefflai.LSM_SPARTACUS.count.pdf')
  plt.close() 
  #fig.show()
  ##
  ## MODIS albedo as function of LAI and canopy height
  ##
  chmStep = 5.
  laiStep = 0.5
  df['height_bin'] = np.round(df.height/chmStep)*chmStep
  df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
  group = df.groupby(['MODIS_LAI_bin', 'height_bin'])
  chmArr = np.arange(df.height_bin.min(), df.height_bin.max(), chmStep)
  laiArr = np.arange(df.MODIS_LAI_bin.min(), df.MODIS_LAI_bin.max(), laiStep)
  wv = ['vis', 'nir', 'shortwave']
  sa = ['BSA', 'WSA']
  fig = plt.figure(figsize=(16,9))
  gs = GridSpec(2,13)
  ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,12,4)] for i in range(2)]
  for si in range(len(sa)):
   for wj in range(len(wv)):
    albStr = 'MODIS_%s_%s' % (sa[si], wv[wj])
    alb = group[albStr].mean()
    albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
    for i in range(len(laiArr)):
      for j in range(len(chmArr)):
        if (laiArr[i], chmArr[j]) in alb.index:
          albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
    c = ax[si][wj].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                              np.append(chmArr, chmArr[-1]+chmStep),
                              albArr.T, vmin=0, vmax=1)
    if wj == 0:
      ax[si][wj].set_ylabel('%s\nCanopy height (%s)' % (sa[si], 'm'))
    else:
      plt.setp(ax[si][wj].get_yticklabels(), visible=False)
    if si == 0:
      ax[si][wj].set_title('%s' % wv[wj])
      plt.setp(ax[si][wj].get_xticklabels(), visible=False)
    if si == 1:
      ax[si][wj].set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='Mean MODIS albedo')
  fig.tight_layout()
  fig.savefig('final_plots/heightVefflai.icesat2.mean.pdf')
  plt.close()
  #fig.show()
  fig = plt.figure(figsize=(16,9))
  gs = GridSpec(2,13)
  ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,12,4)] for i in range(2)]
  for si in range(len(sa)):
   for wj in range(len(wv)):
    albStr = 'SPARTACUS_%s_%s' % (sa[si], wv[wj])
    alb = group[albStr].count()
    albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
    for i in range(len(laiArr)):
      for j in range(len(chmArr)):
        if (laiArr[i], chmArr[j]) in alb.index:
          albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
    c = ax[si][wj].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                              np.append(chmArr, chmArr[-1]+chmStep),
                              albArr.T, vmin=0, vmax=100)
    if wj == 0:
      ax[si][wj].set_ylabel('%s\nCanopy height (%s)' % (sa[si], 'm'))
    else:
      plt.setp(ax[si][wj].get_yticklabels(), visible=False)
    if si == 0:
      ax[si][wj].set_title('%s' % wv[wj])
      plt.setp(ax[si][wj].get_xticklabels(), visible=False)
    if si == 1:
      ax[si][wj].set_xlabel('MODIS Effective LAI (m²m⁻²)')
  fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='# pixels')
  fig.tight_layout()
  fig.savefig('final_plots/heightVefflai.LSM_SPARTACUS.count.pdf')
  plt.close()
  #fig.show()
  vList = ['veg_fraction', 'height', 'MODIS_LAI_eff', 'MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover', 'zen', 'MODIS_LAI']
  for v in vList:
    vMin, vMax = df[v].min(), df[v].max()
    if vMax > 20:
      vArr = np.arange(0, int(vMax)+5, 5)
      widths=[4]
    elif vMax > 1:
      vArr = np.arange(0, int(vMax)+1, 1)
      widths=[0.8]
    else:
      vArr = np.arange(0, 1.01, 0.1)
      widths=[0.08]
    fig, ax = plt.subplots(3,6,figsize=(16,9),sharex=True)
    for i in range(len(vArr)-1):
      dfi = df.loc[(df[v]>=vArr[i])&(df[v]<vArr[i+1])]
      r, c = 0, 0
      for wv in ['vis', 'nir', 'shortwave']:
        for sa in ['BSA', 'WSA']:
          for rt in ['Sellers', 'SPARTACUS', 'MODIS']:
            ax[r][c].boxplot(dfi['%s_%s_%s' % (rt, sa, wv)], vert=True, positions=[vArr[i]],
                             flierprops={'ms':1}, widths=widths)
            ax[r][c].axhline(0, lw=0.1, c='k')
            ax[r][c].set_ylabel('%s %s\n(%s - MODIS)' % (sa, wv, rt))
            ax[r][c].set_xlabel('%s' % v)
            ax[r][c].set_ylim(0,1)
            if vMax > 20:
              ax[r][c].set_xlim(vArr[0]-2.5, vArr[-1]+2.5)
            elif vMax > 1:
              ax[r][c].set_xlim(vMin-.5, vMax+.5)
            else:
              ax[r][c].set_xlim(vMin-0.05, vMax+0.05)
            r+=1
          r=0
          c += 1
    for r in range(2):
      for c in range(6):
        if vMax > 1:
          ax[r][c].set_xticks(vArr[:-1], ['%.1f' % (v+(vArr[1]-vArr[0])/2.) for v in vArr[:-1]], rotation=90, va='top', ha='center')
        else:
            ax[r][c].set_xticks(vArr[:-1], ['%.2f' % (v+(vArr[1]-vArr[0])/2.) for v in vArr[:-1]], rotation=90, va='top', ha='center')
    fig.tight_layout()
    fig.savefig('final_plots/boxplot.absolute.nirRTvalues.%s.png' % v)
    plt.close()
  from matplotlib.colors import Normalize
  from matplotlib.cm import viridis as cmap
  from matplotlib.cm import ScalarMappable
  from matplotlib.gridspec import GridSpec
  vList = ['veg_fraction', 'height', 'MODIS_LAI_eff', 'MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover', 'zen', 'MODIS_LAI']
  for v in vList:
    vMin, vMax = df[v].min(), df[v].max()
    vArr = np.linspace(vMin, vMax+1e-6, 10)
    bins = np.arange(-1, 1.01, 0.01)
    #fig, ax = plt.subplots(2,6,figsize=(16,5),sharex=True)
    fig = plt.figure(figsize=(16,5))
    gs = GridSpec(2,19)
    ax = [[fig.add_subplot(gs[i,j:j+3]) for j in range(0, 18, 3)] for i in range(2)]
    for i in range(len(vArr)-1):
      dfi = df.loc[(df[v]>=vArr[i])&(df[v]<vArr[i+1])]
      r, c = 0, 0
      for wv in ['vis', 'nir', 'shortwave']:
        for sa in ['BSA', 'WSA']:
          for rt in ['Sellers', 'SPARTACUS']:
            n, _ = np.histogram(dfi['%s_%s_%s_diff' % (rt, sa, wv)], bins=bins)
            n = 100.*(n/np.sum(n))
            ax[r][c].plot(bins[:-1]+0.05, n, c=cmap((np.mean([vArr[i], vArr[i+1]])-vMin)/(vMax-vMin)))
            ax[r][c].set_xlabel('%s %s\n(%s - MODIS)' % (sa, wv, rt))
            ax[r][c].set_ylabel('% pixels')
            r+=1
          r=0
          c += 1
    cax = fig.add_subplot(gs[:,-1])
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vMin, vmax=vMax), cmap=cmap), label=v, cax=cax)
    fig.tight_layout()
    fig.savefig('final_plots/biasDistribution.blackGround.%s.png' % v)

  for v in vList:
    vMin, vMax = df[v].min(), df[v].max()
    if vMax > 20:
      vArr = np.arange(0, int(vMax)+5, 5)
      widths=[4]
    elif vMax > 1:
      vArr = np.arange(0, int(vMax)+1, 1)
      widths=[0.8]
    else:
      vArr = np.arange(0, 1.01, 0.1)
      widths=[0.08]
    fig, ax = plt.subplots(2,6,figsize=(16,9),sharex=True)
    for i in range(len(vArr)-1):
      dfi = df.loc[(df[v]>=vArr[i])&(df[v]<vArr[i+1])]
      r, c = 0, 0
      for wv in ['vis', 'nir', 'shortwave']:
        for sa in ['BSA', 'WSA']:
          for rt in ['Sellers', 'SPARTACUS']:
            ax[r][c].boxplot(dfi['%s_%s_%s_diff' % (rt, sa, wv)], vert=True, positions=[vArr[i]],
                             flierprops={'ms':1}, widths=widths)
            ax[r][c].axhline(0, lw=0.1, c='k')
            ax[r][c].set_ylabel('%s %s\n(%s - MODIS)' % (sa, wv, rt))
            ax[r][c].set_xlabel('%s' % v)
            ax[r][c].set_ylim(-1,1)
            if vMax > 20:
              ax[r][c].set_xlim(vArr[0]-2.5, vArr[-1]+2.5)
            elif vMax > 1:
              ax[r][c].set_xlim(vMin-.5, vMax+.5)
            else:
              ax[r][c].set_xlim(vMin-0.05, vMax+0.05)
            r+=1
          r=0
          c += 1
    for r in range(2):
      for c in range(6):
        if vMax > 1:
          ax[r][c].set_xticks(vArr[:-1], ['%.1f' % (v+(vArr[1]-vArr[0])/2.) for v in vArr[:-1]],
                                rotation=90, va='top', ha='center')
        else:
            ax[r][c].set_xticks(vArr[:-1], ['%.2f' % (v+(vArr[1]-vArr[0])/2.) for v in vArr[:-1]],
                                rotation=90, va='top', ha='center')
    fig.tight_layout()
    fig.savefig('final_plots/boxplot.blackGround.%s.png' % v)
    plt.close()
  fig, ax = plt.subplots(2, 6, figsize=(16,9))
  c = 0
  for wv in ['vis', 'nir', 'shortwave']:
    for sa in ['BSA', 'WSA']:
      r = 0
      for rt in ['Sellers', 'SPARTACUS']:
        lab = '%s_%s' % (sa, wv)
        dfi = df[['MODIS_%s' % lab, '%s_%s' % (rt, lab)]].dropna()
        sns.histplot(dfi, x='MODIS_%s' % lab, y='%s_%s' % (rt, lab), ax=ax[r,c], binwidth=(0.01, 0.01))
        res = linregress(dfi['MODIS_%s' % lab], dfi['%s_%s' % (rt, lab)])
        ax[r,c].plot([0, 1], res.slope*np.array([0,1])+res.intercept, c='k')
        ax[r,c].set_xlabel('MODIS %s %s' % (sa, wv))
        ax[r,c].set_ylabel('%s %s %s' % (rt, sa, wv))
        ax[r,c].set_aspect('equal')
        ax[r,c].set_xlim(0,1)
        ax[r,c].set_ylim(0,1)
        ax[r,c].text(.95, .95, 'R² = %.2f\np = %.2f' % (res.rvalue**2, res.pvalue), va='top', ha='right')
        r += 1
      r = 0
      c += 1
  fig.tight_layout()
  fig.savefig('final_plots/MODELvMODIS.icesat2.sodankyla.beforeMay.SelleffSPARTtrue.PerMetreExtinction.pdf')
  plt.close()
  vList = ['veg_fraction', 'height', 'veg_scale', 'veg_extinction', 'MODIS_LAI_eff', 'MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover', 'zen', 'MODIS_LAI', 'ground_sw_albedo', 'date']
  for v in vList:
    fig, ax = plt.subplots(2, 6, figsize=(16,9), sharex=True)
    c = 0
    for wv in ['vis', 'nir', 'shortwave']:
      for sa in ['BSA', 'WSA']:
        r = 0
        for rt in ['Sellers', 'SPARTACUS']:
          lab = '%s_%s' % (sa, wv)
          dfi = df.sort_values(v)
          ax[r,c].scatter(dfi['MODIS_%s' % lab], dfi['%s_%s' % (rt, lab)], c=dfi[v], s=1)
          ax[r,c].set_xlabel('MODIS %s %s' % (sa, wv))
          ax[r,c].set_ylabel('%s %s %s' % (rt, sa, wv))
          ax[r,c].set_aspect('equal')
          ax[r,c].set_xlim(0,1)
          ax[r,c].set_ylim(0,1)
          r += 1
        r = 0
        c += 1
    fig.suptitle(v)
    fig.tight_layout()
    fig.savefig('final_plots/MODELvMODIS.blackGround.SCover0.75.spring.%s.png' % v)
    plt.close()
    #fig.show()
  
  fig, ax = plt.subplots(2, 6, figsize=(16,9), sharex=True)
  c = 0
  for wv in ['vis', 'nir', 'shortwave']:
    for sa in ['BSA', 'WSA']:
      r = 0
      for rt in ['Sellers', 'SPARTACUS']:
        lab = '%s_%s' % (sa, wv)
        dfi = df[['MODIS_Snow_Albedo', '%s_%s' % (rt, lab)]].dropna()
        sns.histplot(dfi, x='MODIS_Snow_Albedo', y='%s_%s' % (rt, lab), ax=ax[r,c], binwidth=(0.01, 0.01))
        res = linregress(dfi['MODIS_Snow_Albedo'], dfi['%s_%s' % (rt, lab)])
        ax[r,c].plot([0, 1], res.slope*np.array([0,1])+res.intercept, c='k')
        ax[r,c].set_xlabel('MODIS Snow Albedo')
        ax[r,c].set_ylabel('%s %s %s' % (rt, sa, wv))
        ax[r,c].set_aspect('equal')
        ax[r,c].set_xlim(0,1)
        ax[r,c].set_ylim(0,1)
        ax[r,c].text(.95, .95, 'R² = %.2f\np = %.2f' % (res.rvalue**2, res.pvalue), va='top', ha='right')
        r += 1
      r = 0
      c += 1
  fig.tight_layout()
  fig.show()
  fig.savefig('final_plots/MODELvMODISSnowAlbedo.LSM_SCweighted.SCover0.75.spring.pdf')
  plt.close()
  for wv in ['vis', 'nir']:
    for sa in ['BSA', 'WSA']:
      lab = '%s_%s' % (sa, wv)
      df['%s_diff' % lab] = df['SPARTACUS_%s' % lab] -\
                            df['Sellers_%s' % lab]
      plotRelationship(df, 'zen', '%s_diff' % lab)
      #plotRelationship(df, 'MODIS_%s' % lab, 'SPARTACUS_%s' % lab)

  for wv in ['vis', 'nir']:
    for sa in ['BSA', 'WSA']:
      for rt in ['Sellers', 'SPARTACUS']:
        lab = '%s_%s_%s' % (rt, sa, wv)
        ## LAI
        fig, ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.set_xlabel('LAI')
        ax.set_ylabel('δ bias / δθ', color='g'); ax.tick_params(axis='y', colors='g')
        ax2.set_ylabel('R²', color='r'); ax2.tick_params(axis='y', colors='r')
        for LAI in np.arange(0, 7.1, 0.5):
          mask = (df.MODIS_LAI > LAI) & (df.MODIS_LAI <= LAI+0.5)
          dfi = df.loc[mask, ['zen',lab]].dropna()
          if mask.sum() > 0:
            res = linregress(dfi['zen'], dfi[lab])
            m = ax.scatter(LAI+.25, res.slope, c='g')
            r = ax2.scatter(LAI+.25, res.rvalue**2, c='r')
        fig.tight_layout()
        fig.savefig('final_plots/LSM.%s.lai.gradient.pdf' % lab)
        ## Snow cover
        fig, ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.set_xlabel('Snow Cover')
        ax.set_ylabel('δ bias / δθ', color='g'); ax.tick_params(axis='y', colors='g')
        ax2.set_ylabel('R²', color='r'); ax2.tick_params(axis='y', colors='r')
        for sc in np.arange(0., 0.91, 0.1):
          mask = (df.MODIS_NDSI_Snow_Cover > sc) & (df.MODIS_NDSI_Snow_Cover <= sc+0.1)
          dfi = df.loc[mask, ['zen',lab]].dropna()
          if mask.sum()>0:
            res = linregress(dfi['zen'], dfi[lab])
            m = ax.scatter(sc+.05, res.slope, c='g', label='m')
            r = ax2.scatter(sc+.05, res.rvalue**2, c='r', label='R²')
        fig.tight_layout()
        fig.savefig('final_plots/LSM.%s.snowcover.gradient.pdf' % lab)
        ## Snow albedo
        fig, ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.set_xlabel('Snow Albedo')
        ax.set_ylabel('δ bias / δθ', color='g'); ax.tick_params(axis='y', colors='g')
        ax2.set_ylabel('R²', color='r'); ax2.tick_params(axis='y', colors='r')
        for sab in np.arange(0., 0.91, 0.1):
          mask = (df.MODIS_Snow_Albedo > sab) & (df.MODIS_Snow_Albedo <= sab+0.1)
          dfi = df.loc[mask, ['zen',lab]].dropna()
          if mask.sum()>0:
            res = linregress(dfi['zen'], dfi[lab])             
            m = ax.scatter(sab+.05, res.slope, c='g', label='m')
            r = ax2.scatter(sab+.05, res.rvalue**2, c='r', label='R²')
        fig.tight_layout()
        fig.savefig('final_plots/LSM.%s.snowalbedo.gradient.pdf' % lab)
        ## Canopy height
        fig, ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.set_xlabel('Canopy height')
        ax.set_ylabel('δ bias / δθ', color='g'); ax.tick_params(axis='y', colors='g')
        ax2.set_ylabel('R²', color='r'); ax2.tick_params(axis='y', colors='r')
        for h in np.arange(0., 30.1, 5.):
          mask = (df.chm > h) & (df.chm <= h+5.)
          dfi = df.loc[mask, ['zen',lab]].dropna()
          if mask.sum()>0:
            res = linregress(dfi['zen'], dfi[lab])
            m = ax.scatter(h+2.5, res.slope, c='g', label='m')
            r = ax2.scatter(h+2.5, res.rvalue**2, c='r', label='R²')
        fig.tight_layout()
        fig.savefig('final_plots/LSM.%s.canopyheight.gradient.pdf' % lab)
        ## Canopy cover
        fig, ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.set_xlabel('Canopy cover')
        ax.set_ylabel('δ bias / δθ', color='g'); ax.tick_params(axis='y', colors='g')
        ax2.set_ylabel('R²', color='r'); ax2.tick_params(axis='y', colors='r')
        for cv in np.arange(0., 100., 10.):
          mask = (df.cv > cv) & (df.cv <= cv+10.)
          dfi = df.loc[mask, ['zen',lab]].dropna()
          if mask.sum()>0:
            res = linregress(dfi['zen'], dfi[lab])
            m = ax.scatter(cv+5, res.slope, c='g', label='m')
            r = ax2.scatter(cv+5, res.rvalue**2, c='r', label='R²')
        fig.tight_layout()
        fig.savefig('final_plots/LSM.%s.canopycover.gradient.pdf' % lab)

  print('pixels above 75°', (df.zen>=75).sum())
  df = df.loc[(df.zen>=75).values]
  for wv in ['vis', 'nir']:
    for sa in ['BSA', 'WSA']:
      for rt in ['Sellers', 'SPARTACUS']:
        lab = '%s_%s' % (sa, wv)
        for LAI in np.arange(0, 7.1, 0.5):
          mask = (df.MODIS_LAI > LAI) & (df.MODIS_LAI <= LAI+0.5)
          if mask.sum() > 0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'LAI%01f-%01f' % (LAI, LAI+0.5))
        for sc in np.arange(0.5, 0.91, 0.1):
          mask = (df.MODIS_NDSI_Snow_Cover > sc) & (df.MODIS_NDSI_Snow_Cover <= sc+0.1)
          if mask.sum()>0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'NDSISC%01f-%01f' % (sc, sc+0.1))
        for om in np.arange(0., 0.91, 0.1):
          mask = (df.omega > om) & (df.omega <= om+0.1)
          if mask.sum()>0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'omega%01f-%01f' % (om, om+0.1))
        for sab in np.arange(0., 0.91, 0.1):
          mask = (df.MODIS_Snow_Albedo > sab) & (df.MODIS_Snow_Albedo <= sab+0.1)
          if mask.sum()>0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'SnowAlbedo%01f-%01f' % (sab, sab+0.1))
        for h in np.arange(0., 30.1, 5.):
          mask = (df.chm > h) & (df.chm <= h+5.)
          if mask.sum()>0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'CHM%d-%d' % (h, h+5.))
        for cv in np.arange(0., 100., 10.):
          mask = (df.cv > cv) & (df.cv <= cv+10.)
          if mask.sum()>0:
            plotRelationshipReduced(df.loc[mask], 'zen', '%s_%s_diff' % (rt, lab), 'cv%d-%d' % (cv, cv+5.))
  """
