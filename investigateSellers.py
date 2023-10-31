import glob, os, pdb
import numpy as np
import xarray as xr
from progress.bar import Bar
import matplotlib.pyplot as plt
from matplotlib.cm import viridis as cmap
from matplotlib.colors import Normalize

if __name__=='__main__':
  combinedDir = 'sodankyla/SellersCoarseNC'
  combinedFiles = np.sort(glob.glob(os.path.join(combinedDir, '*.nc')))
  x, y, date = np.array([os.path.split(cF)[-1].split('.')[-4:-1] for cF in combinedFiles]).T
  
  xArr = np.unique(x)
  yArr = np.unique(y)
  dateArr = np.unique(date)
  """
  ## Plot side-by-side albedo & LAI maps and difference
  bar = Bar('Mapping albedo', max=len(dateArr))
  for date in dateArr:
    combinedFilesDate = np.sort([cf for cf in combinedFiles if cf.split('.')[-2]==date])
    ds0 = xr.open_dataset(combinedFilesDate[0])
    wvArr = ['PAR']
    dfArr = [0]
    rArr = [0.1]
    tArr = [0.1]
    SAarr = [0.3]
    for wv in wvArr:
      for df in dfArr:
        for r in rArr:
          for t in tArr:
            for SA in SAarr:
                try:
                  ds = xr.Dataset()
                  for cf in combinedFilesDate:
                    dsi = xr.open_dataset(cf)
                    dsi.coords['wv'] = ['PAR', 'NIR']
                    dsi = dsi.sel(date=date, r=r, t=t, df=df, wv=wv)
                    dsi = dsi.sel(SA=dsi.MODIS_Snow_Albedo, method='nearest')
                    ds = xr.merge([ds, dsi])
                  ds.coords['wv'] = 'PAR'
                  lcMask = (ds.IGBP_LC != 13) & (ds.IGBP_LC != 17)
                  ## Plot Albedo
                  fig = plt.figure(figsize=(12,4))
                  ax = [plt.subplot2grid((1,100), (0,0), colspan=20),
                        plt.subplot2grid((1,100), (0,30), colspan=20),
                        plt.subplot2grid((1,100), (0,70), colspan=20)]
                  cax = [plt.subplot2grid((1,100), (0,52), colspan=2),
                         plt.subplot2grid((1,100), (0,92), colspan=2)]
                  ax[0].set_title('MODIS'); ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
                  ylgn = ax[0].pcolormesh(ds.lon, ds.lat,
                                          ds.where(lcMask).MODIS_Albedo,
                                          cmap='Greys_r', vmin=0, vmax=1)
                  ax[1].set_title('Sellers'); ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
                  ax[1].pcolormesh(ds.lon, ds.lat,
                                   ds.where(lcMask).Sellers_Albedo.sel(LAI=ds.Sellers_LAI_BestFit, method='nearest'),
                                   cmap='Greys_r', vmin=0, vmax=1)
                  ax[2].set_title('Difference'); ax[2].set_xlabel('Longitude'); ax[2].set_ylabel('Latitude')
                  bwr = ax[2].pcolormesh(ds.lon, ds.lat,
                                         ds.where(lcMask).Sellers_Albedo.sel(LAI=ds.Sellers_LAI_BestFit, method='nearest')\
                                         -ds.MODIS_Albedo, cmap='bwr', vmin=-.25, vmax=.25)
                  fig.colorbar(ylgn, cax=cax[0], label='Albedo')
                  fig.colorbar(bwr, cax=cax[1], label='Albedo Difference (Sellers - MODIS)')
                  fname = 'mapCompPlot/albedo.fitted.%s.%s.diffuseFrac.%d.SA.%.2f.r.%.1f.t.%.1f.png' % (date, wv, df, SA, r, t)
                  fig.savefig(fname)
                  plt.close()
                  ## Plot Sellers output using MODIS LAI
                  fig = plt.figure(figsize=(12,4))
                  ax = [plt.subplot2grid((1,100), (0,0), colspan=20),
                        plt.subplot2grid((1,100), (0,30), colspan=20),
                        plt.subplot2grid((1,100), (0,70), colspan=20)]
                  cax = [plt.subplot2grid((1,100), (0,52), colspan=2),
                         plt.subplot2grid((1,100), (0,92), colspan=2)]
                  ax[0].set_title('MODIS'); ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
                  ylgn = ax[0].pcolormesh(ds.lon, ds.lat,
                                          ds.where(lcMask).MODIS_Albedo,
                                          cmap='Greys_r', vmin=0, vmax=1)
                  ax[1].set_title('Sellers'); ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
                  ax[1].pcolormesh(ds.lon, ds.lat,
                                   ds.where(lcMask).Sellers_Albedo.sel(LAI=ds.MODIS_LAI, method='nearest'),
                                   cmap='Greys_r', vmin=0, vmax=1)
                  ax[2].set_title('Difference'); ax[2].set_xlabel('Longitude'); ax[2].set_ylabel('Latitude')
                  bwr = ax[2].pcolormesh(ds.lon, ds.lat,
                                         ds.where(lcMask).Sellers_Albedo.sel(LAI=ds.MODIS_LAI, method='nearest')\
                                         -ds.MODIS_Albedo, cmap='bwr', vmin=-.25, vmax=.25)
                  fig.colorbar(ylgn, cax=cax[0], label='Albedo')
                  fig.colorbar(bwr, cax=cax[1], label='Albedo Difference (Sellers - MODIS)')
                  fname = 'mapCompPlot/albedo.modis_lai.%s.%s.diffuseFrac.%d.SA.%.1f.r.%.1f.t.%.1f.png' % (date, wv, df, SA, r, t)
                  fig.savefig(fname)
                  plt.close()
                  ## Plot LAI 
                  fig = plt.figure(figsize=(15,4))
                  ax = [plt.subplot2grid((1,100), (0,0), colspan=20),
                        plt.subplot2grid((1,100), (0,30), colspan=20),
                        plt.subplot2grid((1,100), (0,70), colspan=20)]
                  cax = [plt.subplot2grid((1,100), (0,52), colspan=2),
                         plt.subplot2grid((1,100), (0,92), colspan=2)]
                  ax[0].set_title('MODIS'); ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
                  ylgn = ax[0].pcolormesh(ds.lon, ds.lat,
                                          ds.where(lcMask).MODIS_LAI,
                                          cmap='YlGn', vmin=0, vmax=ds.LAI.max())
                  ax[1].set_title('Sellers'); ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
                  ax[1].pcolormesh(ds.lon, ds.lat,
                                   ds.where(lcMask).Sellers_LAI_BestFit,
                                   cmap='YlGn', vmin=0, vmax=ds.LAI.max())
                  ax[2].set_title('Difference'); ax[2].set_xlabel('Longitude'); ax[2].set_ylabel('Latitude')
                  bwr = ax[2].pcolormesh(ds.lon, ds.lat,
                                         ds.where(lcMask).Sellers_LAI_BestFit-ds.MODIS_LAI,
                                         cmap='bwr', vmin=-5., vmax=5.)
                  fig.colorbar(ylgn, cax=cax[0], label='LAI')
                  fig.colorbar(bwr, cax=cax[1], label='LAI Difference (Sellers - MODIS)')
                  fname = 'mapCompPlot/lai.%s.%s.diffuseFrac.%d.SA.%.1f.r.%.1f.t.%.1f.png' % (date, wv, df, SA, r, t)
                  fig.savefig(fname)
                  plt.close()
                  bar.next()
                except:
                  pass
  bar.finish()
  
  ## Plot leaf r, t affect on Sellers albedo error as function of LAI difference
  dsAll = xr.Dataset()
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in combinedFiles:
    dsi = xr.open_dataset(cf)
    dsi = dsi.sel(SA=dsi.MODIS_Snow_Albedo, t=dsi.r, method='nearest')
    dsAll = xr.merge([dsAll, dsi])
    bar.next()
  bar.finish()
  for wv in dsAll.wv.values:
    for df in dsAll.df.values:
      ds = dsAll.sel(wv=wv, df=df)
      if df==1:
        dirdif = 'diffuse'
      else:
        dirdif = 'direct'
      for r in ds.r.values:
        plt.plot((ds.LAI - ds.MODIS_LAI).mean(dim=('date','y','x')),
                 (ds.Sellers_Albedo.sel(r=r) - ds.MODIS_Albedo).mean(dim=('date','y','x')),
                 color=cmap(r / ds.r.max()), label='%.2f' % r)
      plt.axhline(0, lw=.5, c='k')
      plt.axvline(0, c='r', ls=':')
      plt.legend(loc='best', edgecolor='none')
      plt.xlabel('LAI offset (Sellers - MODIS)')
      plt.ylabel('Albedo difference (Sellers - MODIS)')
      plt.tight_layout()
      plt.savefig('laiAvgPlot/leaf_r,t_dependence.%s.%s.pdf' % (wv, dirdif))
      plt.close()
  
  ## Plot leaf r, t affect on Sellers albedo error as function of LAI difference
  dsAll = xr.Dataset()
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in [cf for cf in combinedFiles if cf.split('.')[-2]=='20190411']:
    dsi = xr.open_dataset(cf)
    dsi = dsi.sel(SA=dsi.MODIS_Snow_Albedo, method='nearest')
    dsAll = xr.merge([dsAll, dsi])
    bar.next()
  bar.finish()
  for wv in dsAll.wv.values:
    for df in dsAll.df.values:
      ds = dsAll.sel(wv=wv, df=df)
      if df==1:
        dirdif = 'diffuse'
      else:
        dirdif = 'direct'
      fig, ax = plt.subplots(1,2)
      ax[0].set_title('Sellers')
      modAlb = ds.MODIS_Albedo.mean()
      selAlb = ds.sel(LAI=ds.Sellers_LAI_BestFit, method='nearest').Sellers_Albedo.mean(dim=('date', 'x', 'y'))
      sel = ax[0].pcolormesh(ds.r, ds.t, selAlb, cmap='Greys_r', vmin=0, vmax=1)
      ax[1].set_title('Difference')
      diff = ax[1].pcolormesh(ds.r, ds.t, selAlb-modAlb, cmap='bwr', vmin=-.25, vmax=.25)
      for i in range(2):
        ax[i].set_xlabel('Leaf reflectance')
        ax[i].set_ylabel('Leaf transmittance')
      fig.colorbar(sel, ax=ax[0], label='Mean albedo')
      fig.colorbar(diff, ax=ax[1], label='Albedo difference (Sellers - MODIS)')
      fig.tight_layout()
      fig.show(); pdb.set_trace()
      plt.savefig('laiAvgPlot/leaf_r,t_dependence.%s.%s.bestfitlai.pdf' % (wv, dirdif))
      plt.close()
  """
  ## Plot leaf r, t affect on Sellers albedo error as function of LAI difference
  dsAll = xr.Dataset()
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in [cf for cf in combinedFiles if cf.split('.')[-2]=='20190411']:
    dsi = xr.open_dataset(cf)
    dsi = dsi.sel(SA=dsi.MODIS_Snow_Albedo, method='nearest')
    dsAll = xr.merge([dsAll, dsi])
    bar.next()
  bar.finish()
  for wv in dsAll.wv.values:
    for df in dsAll.df.values:
      ds = dsAll.sel(wv=wv, df=df)
      if df==1:
        dirdif = 'diffuse'
      else:
        dirdif = 'direct'
      fig, ax = plt.subplots(1,2,figsize=(10,4))
      modLAI = ds.MODIS_LAI
      selLAI = ds.Sellers_LAI_BestFit.where(ds.r+ds.t < 1)
      ax[0].set_title('Sellers')
      sel = ax[0].pcolormesh(ds.r, ds.t, selLAI.mean(dim=('date','y','x'), skipna=True),
                             cmap='YlGn', vmin=0, vmax=3)
      ax[1].set_title('Difference')
      diff = ax[1].pcolormesh(ds.r, ds.t, (selLAI-modLAI).mean(dim=('date','y','x'), skipna=True),
                              cmap='bwr', vmin=-1.5, vmax=1.5)
      for i in range(2):
        ax[i].set_xlabel('Leaf reflectance')
        ax[i].set_ylabel('Leaf transmittance')
        ax[i].set_aspect('equal')
      cb = fig.colorbar(sel, ax=ax[0], label='Mean Best Fit LAI')
      cb.ax.axhline(modLAI.mean(skipna=True), color='r', ls=':')
      fig.colorbar(diff, ax=ax[1], label='LAI difference (Sellers Best Fit - MODIS)')
      fig.tight_layout()
      plt.savefig('laiAvgPlot/leaf_r,t_lai_dependence.%s.%s.png' % (wv, dirdif))
      plt.close()
  """
  ## Plot snow albedo effect on Sellers albedo error as function of LAI difference
  for wv in ['PAR', 'NIR']:
    for df in [0,1]:
      if wv == 'PAR':
        r = t = 0.1
      if wv == 'NIR':
        r = t = 0.3
      ds = xr.Dataset()
      bar = Bar('Reading files', max=len(combinedFiles))
      for cf in combinedFiles:
        dsi = xr.open_dataset(cf)
        dsi.coords['wv'] = ['PAR', 'NIR']
        dsi = dsi.sel(wv=wv, df=df, r=r, t=t)
        ds = xr.merge([ds, dsi])
        bar.next()
      bar.finish()
      if df==1:
        dirdif = 'diffuse'
      else:
        dirdif = 'direct'
      for SA in ds.SA.values:
        plt.plot((ds.LAI - ds.MODIS_LAI).mean(dim=('date','y','x')),
                 (ds.Sellers_Albedo.sel(SA=SA) - ds.MODIS_Albedo).mean(dim=('date','y','x')),
                 color=cmap(SA / ds.SA.max()), label='%.1f' % SA)
      plt.axhline(0, lw=.5, c='k')
      plt.legend(loc='best', edgecolor='none')
      plt.xlabel('LAI offset (Sellers - MODIS)')
      plt.ylabel('Albedo difference (Sellers - MODIS)')
      plt.tight_layout()
      plt.savefig('laiAvgPlot/snow_albedo_dependence.%s.%s.pdf' % (wv, dirdif))
      plt.close()

  ## Plot Albedo error against Sellers LAI for different snow albedos at each pixel
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in combinedFiles:
    ds = xr.open_dataset(cf)
    for d in ds.date:
      for x in ds.x:
        for y in ds.y:
          for df in ds.df:
            for r in ds.r:
              for t in ds.t:
                for wv in ds.wv:
                  fig, ax = plt.subplots(1,1)
                  for SA in ds.SA:
                    dsi = ds.sel(date=d, y=y, x=x, r=r, t=t, SA=SA, wv=wv, df=df)
                    ax.plot(dsi.LAI, dsi.Sellers_Albedo - dsi.MODIS_Albedo, c=cmap(SA/ds.SA.max()))
                  if df == 0:
                    ax.set_title('Direct %s, r=%.1f, t=%.1f' % (str(wv.values), r, t))
                  else:
                    ax.set_title('Diffuse %s, r=%.1f, t=%.1f')
                  ax.set_xlabel('Sellers LAI')
                  ax.set_ylabel('Albedo difference (Sellers - MODIS)')
                  ax.axvline(dsi.MODIS_LAI, c='r',ls=':', label='MODIS LAI')
                  ax.set_xlim(ds.LAI.min(), ds.LAI.max())
                  ax.set_ylim(-1, 1)
                  ax.axhline(0, lw=.5, c='k')
                  sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=ds.SA.min(), vmax=ds.SA.max()))
                  plt.colorbar(sm, label='Snow Albedo', ax=ax)
                  fname = 'SA.%s.x.%d.y.%d.df.%d.r.%.1f.t.%.1f.%s.png' % (str(d.values)[:10].replace('-',''), x, y, df, 
                                                                          dsi.r.values, dsi.t.values, str(wv.values))
                  fig.tight_layout()
                  fig.savefig(os.path.join('laiCompPlot',fname))
                  plt.close()
    bar.next()
  bar.finish()
  
  ## Plot Albedo error against Sellers LAI for different leaf reflectances at each pixel
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in combinedFiles:
    ds = xr.open_dataset(cf)
    ds.coords['wv'] = ['PAR', 'NIR']
    for d in ds.date:
      for x in ds.x:
        for y in ds.y:
          for df in ds.df:
            for SA in ds.SA:
              for wv in ds.wv:
                for t in ds.t:
                  fig, ax = plt.subplots(1,1)
                  for r in ds.r:
                    dsi = ds.sel(date=d, y=y, x=x, r=r, t=t, SA=SA, wv=wv, df=df)
                    ax.plot(dsi.LAI, dsi.Sellers_Albedo - dsi.MODIS_Albedo, c=cmap(r/ds.r.max()))
                  if df == 0:
                    ax.set_title('Direct %s, snow albedo=%.1f, t=%.1f' % (str(wv.values), SA, t))
                  else:
                    ax.set_title('Diffuse %s, snow albedo=%.1f, t=%.1f')
                  ax.set_xlabel('Sellers LAI')
                  ax.set_ylabel('Albedo difference (Sellers - MODIS)')
                  ax.axvline(dsi.MODIS_LAI, c='r',ls=':', label='MODIS LAI')
                  ax.set_xlim(ds.LAI.min(), ds.LAI.max())
                  ax.set_ylim(-1, 1)
                  ax.axhline(0, lw=.5, c='k')
                  sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=ds.r.min(), vmax=ds.r.max()))
                  plt.colorbar(sm, label='Leaf reflectance', ax=ax)
                  fname = 'leaf_r.%s.x.%d.y.%d.df.%d.SA.%.1f.t.%.1f.%s.png' % (str(d.values)[:10].replace('-',''), x, y, df, 
                                                                          dsi.SA.values, dsi.t.values, str(wv.values))
                  fig.tight_layout()
                  fig.savefig(os.path.join('laiCompPlot',fname))
                  plt.close()
    bar.next()
  bar.finish()
  
  ## Plot Albedo error against Sellers LAI for different leaf reflectances at each pixel
  bar = Bar('Reading files', max=len(combinedFiles))
  for cf in combinedFiles:
    ds = xr.open_dataset(cf)
    ds.coords['wv'] = ['PAR', 'NIR']
    for d in ds.date:
      for x in ds.x:
        for y in ds.y:
          for df in ds.df:
            for SA in ds.SA:
              for wv in ds.wv:
                for r in ds.r:
                  fig, ax = plt.subplots(1,1)
                  for t in ds.t:
                    dsi = ds.sel(date=d, y=y, x=x, r=r, t=t, SA=SA, wv=wv, df=df)
                    ax.plot(dsi.LAI, dsi.Sellers_Albedo - dsi.MODIS_Albedo, c=cmap(t/ds.t.max()))
                  if df == 0:
                    ax.set_title('Direct %s, snow albedo=%.1f, r=%.1f' % (str(wv.values), SA, r))
                  else:
                    ax.set_title('Diffuse %s, snow_albedo=%.1f, r=%.1f')
                  ax.set_xlabel('Sellers LAI')
                  ax.set_ylabel('Albedo difference (Sellers - MODIS)')
                  ax.axvline(dsi.MODIS_LAI, c='r',ls=':', label='MODIS LAI')
                  ax.set_xlim(ds.LAI.min(), ds.LAI.max())
                  ax.set_ylim(-1, 1)
                  ax.axhline(0, lw=.5, c='k')
                  sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=ds.r.min(), vmax=ds.r.max()))
                  plt.colorbar(sm, label='Leaf transmittance', ax=ax)
                  fname = 'leaf_t.%s.x.%d.y.%d.df.%d.SA.%.1f.r.%.1f.%s.png' % (str(d.values)[:10].replace('-',''), x, y, df, 
                                                                               dsi.SA.values, dsi.r.values, str(wv.values))
                  fig.tight_layout()
                  fig.savefig(os.path.join('laiCompPlot',fname))
                  plt.close()
    bar.next()
  bar.finish()
  """
