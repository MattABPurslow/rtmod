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

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'

from matplotlib.cm import gist_earth as cmap
color = {'MODIS': cmap(.5),
         'Sellers': cmap(.2),
         'SPARTACUS': cmap(.8)}
wvDict = {'vis':'Visible', 'nir':'NIR', 'shortwave':'Broadband'}
saDict = {'BSA': 'Black Sky Albedo', 'WSA': 'White Sky Albedo'}
wv = list(wvDict)
sa = list(saDict)

if __name__=='__main__':
  #for lidar in ['icesat2.atl08.finland', 'icesat2.atl08', 'icesat2.reclassified', 'als.sodankyla.0m', 'als.finland.0m', 'als.sodankyla.2m', 'als.finland.2m', 'als.sodankyla.5m', 'als.finland.5m']:
  #for lidar in ['icesat2.atl08.finland', 'als.finland.2m']:
  for lidar in ['icesat2.atl08', 'als.sodankyla.2m']:
    color = {'MODIS': cmap(.5),
                     'Sellers': cmap(.2),
                              'SPARTACUS': cmap(.8)}
    wvDict = {'vis':'Visible', 'nir':'NIR', 'shortwave':'Broadband'}
    saDict = {'BSA': 'Black Sky Albedo', 'WSA': 'White Sky Albedo'}
    wv = list(wvDict)
    sa = list(saDict)
    show = False
    df = pd.read_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/rt.%s.pkl' % lidar)
    lidar = '%s.beforeMay' % lidar
    df['month'] = [d.month for d in df.date]
    df['zenith_bin'] = np.round(df.zen, decimals=0)
    #df = df.loc[df.lat<65]
    df = df.loc[df.month < 5]
    #df = df.groupby(['x', 'y']).mean()
    #df = df.groupby(['zenith_bin']).mean()
    #df = df.groupby(['zenith_bin', 'x', 'y']).mean()
    for wvi in wv:
      for sai in sa:
        for rt in ['Sellers', 'SPARTACUS']:
          lab = '%s_%s' % (sai, wvi)
          df['%s_%s_diff' % (rt, lab)] = df['%s_%s' % (rt, lab)] -\
                                         df['MODIS_%s' % lab]
    ## Create height bins
    chmStep = 5.0
    cvStep = 0.05
    laiStep = 0.5
    df['height_bin'] = np.round(df.height/chmStep)*chmStep
    df['cv_bin'] = np.round(df.veg_fraction/cvStep)*cvStep
    df['MODIS_LAI_bin'] = np.round(df.MODIS_LAI_eff/laiStep)*laiStep
    chmGroup = df.groupby(['MODIS_LAI_bin', 'height_bin'])
    cvGroup = df.groupby(['MODIS_LAI_bin', 'cv_bin'])
    chmArr = np.arange(df.height_bin.min(), 30, chmStep)#df.height_bin.max(), chmStep)
    cvArr = np.arange(0, 1, cvStep)
    laiArr = np.arange(df.MODIS_LAI_bin.min(), 4, laiStep)#df.MODIS_LAI_bin.max(), laiStep)
    ##
    ## Improvement significance test
    ##
    from scipy.stats import ttest_rel
    for wvi in wv:
      for sai in sa:
        lab = '%s_%s' % (sai, wvi)
        dfi = df[['MODIS_%s' % lab, 'Sellers_%s' % lab, 'SPARTACUS_%s' % lab]].dropna()
        for rt in ['Sellers', 'SPARTACUS']:
          dfi['%s_%s_bias' % (rt, lab)] = dfi['%s_%s' % (rt, lab)] - dfi['MODIS_%s' % lab]
        t = ttest_rel(dfi['Sellers_%s_bias' % lab],
                      dfi['SPARTACUS_%s_bias' % lab],
                      alternative='greater')
        print(lidar, wvi, sai, 'p =', t.pvalue, 'significant =', t.pvalue < 0.05)
    """
    if 'finland' in lidar:
      ##
      ## MODIS Snow Albedo as function of LAI & height
      ##
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_Snow_Albedo'
      mArr = np.full(laiArr.shape[0], np.nan)
      cArr = np.full(laiArr.shape[0], np.nan)
      pArr = np.full(laiArr.shape[0], np.nan)
      rArr = np.full(laiArr.shape[0], np.nan)
      for i in range(len(laiArr)):
        if sum(df.MODIS_LAI_bin==laiArr[i])>0:
          dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
          if dfi.height.min()!=dfi.height.max():
            result = linregress(dfi['height'], dfi[albStr])
            mArr[i] = result.slope
            cArr[i] = result.intercept
            pArr[i] = result.pvalue
            rArr[i] = result.rvalue
      ax.plot(laiArr, mArr, c='g')
      ax.set_ylim(-0.01, 0.01)
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
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/modis_snow_albedo.heightdependence.%s.pdf' % lidar)
        plt.close()
      ##
      ## MODIS Snow Albedo as function of LAI & cover
      ##
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_Snow_Albedo'
      mArr = np.full(laiArr.shape[0], np.nan)
      cArr = np.full(laiArr.shape[0], np.nan)
      pArr = np.full(laiArr.shape[0], np.nan)
      rArr = np.full(laiArr.shape[0], np.nan)
      for i in range(len(laiArr)):
        if sum(df.MODIS_LAI_bin==laiArr[i])>0:
          dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
          if dfi.veg_fraction.min()!=dfi.veg_fraction.max():
            result = linregress(dfi['veg_fraction'], dfi[albStr])
            mArr[i] = result.slope
            cArr[i] = result.intercept
            pArr[i] = result.pvalue
            rArr[i] = result.rvalue
      ax.plot(laiArr, mArr, c='g')
      ax.set_ylim(-0.01, 0.01)
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
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/modis_snow_albedo.coverdependence.%s.pdf' % lidar)
        plt.close()
      ##
      ## Linear regression of MODIS/Sellers/SPARTACUS albedo against height for LAI bins
      ##
      fig = plt.figure(figsize=(6,8))
      fig2 = plt.figure(figsize=(6,8))
      gs = GridSpec(13, 2)
      ax = [[fig.add_subplot(gs[i:i+4,j]) for j in range(2)] for i in range(0,12,4)]
      ax2 = [[fig2.add_subplot(gs[i:i+4,j]) for j in range(2)] for i in range(0,12,4)]
      for rt in ['MODIS', 'Sellers', 'SPARTACUS']:
        for si in range(len(sa)):
         for wj in range(len(wv)):
          albStr = '%s_%s_%s' % (rt, sa[si], wv[wj])
          mArr = np.full(laiArr.shape[0], np.nan)
          cArr = np.full(laiArr.shape[0], np.nan)
          pArr = np.full(laiArr.shape[0], np.nan)
          rArr = np.full(laiArr.shape[0], np.nan)
          for i in range(len(laiArr)):
           if sum(df.MODIS_LAI_bin==laiArr[i])>0:
            dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
            if dfi.height.min()!=dfi.height.max():
              result = linregress(dfi['height'], dfi[albStr])
              mArr[i] = result.slope
              cArr[i] = result.intercept
              pArr[i] = result.pvalue
              rArr[i] = result.rvalue
          ax[wj][si].plot(laiArr+laiStep/2., mArr, c=color[rt], ls='-', label=rt)
          ax2[wj][si].plot(laiArr+laiStep/2., rArr, c=color[rt], ls='-', label=rt)
          ax[wj][si].set_ylim(-0.012, 0.012)
          ax[wj][si].set_yticks([-0.01, 0, 0.01])
          ax[wj][si].axhline(0, lw=0.5, c='k')
          ax2[wj][si].set_ylim(-1,1)
          ax2[wj][si].axhline(0, lw=0.5, c='k')
          if wj==0:
            ax[wj][si].set_title(sa[si])
            ax2[wj][si].set_title(sa[si])
          if wj==2:
            ax[wj][si].set_xlabel('MODIS Effective LAI (m²m⁻²)')
            ax2[wj][si].set_xlabel('MODIS Effective LAI (m²m⁻²)')
          if si==0:
            ax[wj][si].text(-0.35, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
            ax2[wj][si].text(-0.35, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
            ax[wj][si].set_ylabel('Gradient (m⁻¹)')
            ax2[wj][si].set_ylabel('R-value')
          if si==1:
            plt.setp(ax[wj][si].get_yticklabels(), visible=False)
            plt.setp(ax2[wj][si].get_yticklabels(), visible=False)
          if wj!=2:
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
            plt.setp(ax2[wj][si].get_xticklabels(), visible=False)
      ax[0][0].legend(loc='upper left', fancybox=False, edgecolor='none', ncols=1)
      ax2[0][0].legend(loc='upper left', fancybox=False, edgecolor='none', ncols=1)
      fig.tight_layout()
      fig2.tight_layout()
      if show:
        fig.show()
        fig2.show()
      else:
        fig.savefig('final_plots/combined.heightdependence.gradient.%s.pdf' % lidar)
        fig2.savefig('final_plots/combined.heightdependence.rvalue.%s.pdf' % lidar)
        plt.close()
      ##
      ## Linear regression of MODIS/Sellers/SPARTACUS albedo against height for LAI bins
      ##
      fig = plt.figure(figsize=(6,8))
      fig2 = plt.figure(figsize=(6,8))
      gs = GridSpec(13, 2)
      ax = [[fig.add_subplot(gs[i:i+4,j]) for j in range(2)] for i in range(0,12,4)]
      ax2 = [[fig2.add_subplot(gs[i:i+4,j]) for j in range(2)] for i in range(0,12,4)]
      for rt in ['MODIS', 'Sellers', 'SPARTACUS']:
        for si in range(len(sa)):
         for wj in range(len(wv)):
          albStr = '%s_%s_%s' % (rt, sa[si], wv[wj])
          mArr = np.full(laiArr.shape[0], np.nan)
          cArr = np.full(laiArr.shape[0], np.nan)
          pArr = np.full(laiArr.shape[0], np.nan)
          rArr = np.full(laiArr.shape[0], np.nan)
          for i in range(len(laiArr)):
           if sum(df.MODIS_LAI_bin==laiArr[i])>0:
            dfi = df.loc[df.MODIS_LAI_bin==laiArr[i]]
            if dfi.veg_fraction.min()!=dfi.veg_fraction.max():
              result = linregress(dfi['veg_fraction'], dfi[albStr])
              mArr[i] = result.slope
              cArr[i] = result.intercept
              pArr[i] = result.pvalue
              rArr[i] = result.rvalue
          ax[wj][si].plot(laiArr+laiStep/2., mArr, c=color[rt], ls='-', label=rt)
          ax2[wj][si].plot(laiArr+laiStep/2., rArr, c=color[rt], ls='-', label=rt)
          ax[wj][si].set_ylim(-0.015, 0.015)
          ax[wj][si].set_yticks([-0.01, 0, 0.01])
          ax[wj][si].axhline(0, lw=0.5, c='k')
          ax2[wj][si].set_ylim(-1,1)
          ax2[wj][si].axhline(0, lw=0.5, c='k')
          if wj==0:
            ax[wj][si].set_title(sa[si])
            ax2[wj][si].set_title(sa[si])
          if wj==2:
            ax[wj][si].set_xlabel('MODIS Effective LAI (m²m⁻²)')
            ax2[wj][si].set_xlabel('MODIS Effective LAI (m²m⁻²)')
          if si==0:
            ax[wj][si].text(-0.35, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
            ax2[wj][si].text(-0.35, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
            ax[wj][si].set_ylabel('Gradient (%⁻¹)')
            ax2[wj][si].set_ylabel('R-value')
          if si==1:
            plt.setp(ax[wj][si].get_yticklabels(), visible=False)
            plt.setp(ax2[wj][si].get_yticklabels(), visible=False)
          if wj!=2:
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
            plt.setp(ax2[wj][si].get_xticklabels(), visible=False)
      ax[0][0].legend(loc='upper left', fancybox=False, edgecolor='none', ncols=1)
      ax2[0][0].legend(loc='upper left', fancybox=False, edgecolor='none', ncols=1)
      fig.tight_layout()
      fig2.tight_layout()
      if show:
        fig.show()
        fig2.show()
      else:
        fig.savefig('final_plots/combined.coverdependence.gradient.%s.pdf' % lidar)
        fig2.savefig('final_plots/combined.coverdependence.rvalue.%s.pdf' % lidar)
        plt.close()
      
      ##
      ## MODIS Snow Cover as function of LAI & canopy height
      ##
      albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_NDSI_Snow_Cover'
      alb = chmGroup[albStr].mean()
      for i in range(len(laiArr)):
        for j in range(len(chmArr)):
          if (laiArr[i], chmArr[j]) in alb.index:
            albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
      c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                        np.append(chmArr, chmArr[-1]+chmStep),
                        albArr.T, vmin=0.25, vmax=0.75)
      ax.set_ylabel('Canopy height (m)')
      ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
      fig.colorbar(c, ax=ax, label='Mean MODIS snow cover', extend='both')
      fig.tight_layout()
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/heightVefflai.MODIS_NDSI_Snow_Cover.%s.pdf' % lidar)
        plt.close()
      
      ##
      ## MODIS Snow Cover as function of LAI & canopy cover
      ##
      albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_NDSI_Snow_Cover'
      alb = cvGroup[albStr].mean()
      for i in range(len(laiArr)):
        for j in range(len(cvArr)):
          if (laiArr[i], cvArr[j]) in alb.index:
            albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
      c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                        np.append(cvArr, cvArr[-1]+cvStep),
                        albArr.T, vmin=0.25, vmax=0.75)
      ax.set_ylabel('Canopy cover (%)')
      ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
      fig.colorbar(c, ax=ax, label='Mean MODIS snow cover', extend='both')
      fig.tight_layout()
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/coverVefflai.MODIS_NDSI_Snow_Cover.%s.pdf' % lidar)
        plt.close()
      
      ##
      ## MODIS Snow Albedo as function of LAI & canopy height
      ##
      albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_Snow_Albedo'
      alb = chmGroup[albStr].mean()
      for i in range(len(laiArr)):
        for j in range(len(chmArr)):
          if (laiArr[i], chmArr[j]) in alb.index:
            albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
      c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                        np.append(chmArr, chmArr[-1]+chmStep),
                        albArr.T, vmin=0.1, vmax=0.6)
      ax.set_ylabel('Canopy height (m)')
      ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
      fig.colorbar(c, ax=ax, label='Mean MODIS Snow Albedo', extend='both')
      fig.tight_layout()
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/heightVefflai.MODIS_Snow_Albedo.%s.pdf' % lidar)
        plt.close()
      ##
      ## MODIS Snow Albedo as function of LAI & canopy cover
      ##
      albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      albStr = 'MODIS_Snow_Albedo'
      alb = cvGroup[albStr].mean()
      for i in range(len(laiArr)):
        for j in range(len(cvArr)):
          if (laiArr[i], cvArr[j]) in alb.index:
            albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
      c = ax.pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                        np.append(cvArr, cvArr[-1]+cvStep),
                        albArr.T, vmin=0.1, vmax=0.6)
      ax.set_ylabel('Canopy cover (%)')
      ax.set_xlabel('MODIS Effective LAI (m²m⁻²)')
      fig.colorbar(c, ax=ax, label='Mean MODIS Snow Albedo', extend='both')
      fig.tight_layout()
      if show:
        fig.show()
      else:
        fig.savefig('final_plots/coverVefflai.MODIS_Snow_Albedo.%s.pdf' % lidar)
        plt.close()
      
      def laiheightplot(rt):
        ## rt albedo as function of LAI and canopy height
        fig = plt.figure(figsize=(6,8))
        gs = GridSpec(3,9)
        ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,8,4)] for i in range(3)]
        for si in range(len(sa)):
         for wj in range(len(wv)):
          albStr = '%s_%s_%s' % (rt, sa[si], wv[wj])
          alb = chmGroup[albStr].mean()
          albArr = np.full((laiArr.shape[0], chmArr.shape[0]), np.nan)
          for i in range(len(laiArr)):
            for j in range(len(chmArr)):
              if (laiArr[i], chmArr[j]) in alb.index:
                albArr[i,j] = alb.loc[laiArr[i], chmArr[j]]
          c = ax[wj][si].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                                    np.append(chmArr, chmArr[-1]+chmStep),
                                    albArr.T, vmin=0.1, vmax=0.6)
          if si == 0:
            ax[wj][si].set_ylabel('Canopy height (%s)' % ('m'))
            ax[wj][si].text(-0.3, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
          else:
            plt.setp(ax[wj][si].get_yticklabels(), visible=False)
          if wj == 0:
            ax[wj][si].set_title('%s' % sa[si])
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
          if wj == 1:
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
          if wj == 2:
            ax[wj][si].set_xlabel('MODIS Effective LAI (m²m$^-$²)')
        fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='Mean %s albedo' % rt, extend='both')
        fig.tight_layout()
        if show:
          fig.show()
        else:
          fig.savefig('final_plots/%s.heightVefflai.%s.pdf' % (rt, lidar))
          plt.close()
    
      def laicoverplot(rt):
        ## rt albedo as function of LAI and canopy cover
        fig = plt.figure(figsize=(6,8))
        gs = GridSpec(3,9)
        ax = [[fig.add_subplot(gs[i,j:j+4]) for j in range(0,8,4)] for i in range(3)]
        for si in range(len(sa)):
         for wj in range(len(wv)):
          albStr = '%s_%s_%s' % (rt, sa[si], wv[wj])
          alb = cvGroup[albStr].mean()
          albArr = np.full((laiArr.shape[0], cvArr.shape[0]), np.nan)
          for i in range(len(laiArr)):
            for j in range(len(cvArr)):
              if (laiArr[i], cvArr[j]) in alb.index:
                albArr[i,j] = alb.loc[laiArr[i], cvArr[j]]
          c = ax[wj][si].pcolormesh(np.append(laiArr, laiArr[-1]+laiStep),
                                    np.append(cvArr, cvArr[-1]+cvStep),
                                    albArr.T, vmin=0.1, vmax=0.6)
          if si == 0:
            ax[wj][si].set_ylabel('Canopy cover (%s)' % ('%'))
            ax[wj][si].text(-0.3, 0.5, wvDict[wv[wj]], ha='center', va='center', rotation=90, fontsize='large', transform=ax[wj][si].transAxes)
          else:
            plt.setp(ax[wj][si].get_yticklabels(), visible=False)
          if wj == 0:
            ax[wj][si].set_title('%s' % sa[si])
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
          if wj == 1:
            plt.setp(ax[wj][si].get_xticklabels(), visible=False)
          if wj == 2:
            ax[wj][si].set_xlabel('MODIS Effective LAI (m²m$^-$²)')
        fig.colorbar(c, cax=fig.add_subplot(gs[:,-1]), label='Mean %s albedo' % rt, extend='both')
        fig.tight_layout()
        if show:
          fig.show()
        else:
          fig.savefig('final_plots/%s.coverVefflai.%s.pdf' % (rt, lidar))
          plt.close()
      for rt in ['MODIS', 'Sellers', 'SPARTACUS']:
        laiheightplot(rt)
        laicoverplot(rt)
    ##
    ## RT model against MODIS comparison
    ##
    fig, ax = plt.subplots(4, 3, figsize=(6,8), sharey=True)
    for wvi in wv:
      for sai in sa:
        for rt in ['Sellers', 'SPARTACUS']:
          if wvi == 'vis':
            c = 0
          elif wvi == 'nir':
            c = 1
          else:
            c = 2
          if rt == 'Sellers':
            if sai == 'BSA':
              r = 0
            else:
              r = 2
          else:
            if sai == 'BSA':
              r = 1
            else:
              r = 3
          lab = '%s_%s' % (sai, wvi)
          dfi = df[['MODIS_%s' % lab, '%s_%s' % (rt, lab)]].dropna()
          sns.histplot(dfi, x='MODIS_%s' % lab, y='%s_%s' % (rt, lab), ax=ax[r,c], binwidth=(0.01, 0.01))
          ax[r,c].plot([0,1], [0,1], c='k', lw=.5)
          res = linregress(dfi['MODIS_%s' % lab], dfi['%s_%s' % (rt, lab)])
          ax[r,c].plot([0, 1], res.slope*np.array([0,1])+res.intercept, c='k')
          if c==0:
            ax[r,c].set_ylabel(rt)
          else:
            ax[r,c].set_ylabel('')
          if r==3:
            ax[r,c].set_xlabel('MODIS')
          else:
            ax[r,c].set_xlabel('')
          ax[r,c].set_aspect('equal')
          ax[r,c].set_xlim(0,1)
          ax[r,c].set_ylim(0,1)
          ax[r,c].set_xticks([0,1])
          ax[r,c].set_yticks([0,1])
          ax[r,c].text(.95, .05, 'R² = %.2f' % (res.rvalue**2), va='bottom', ha='right')
    ax[0,0].set_title('Visible')
    ax[0,1].set_title('NIR')
    ax[0,2].set_title('Broadband')
    fig.tight_layout()
    ax0 = ax[0,0].get_position().extents[1]
    ax1 = ax[1,0].get_position().extents[3]
    ax2 = ax[2,0].get_position().extents[1]
    ax3 = ax[3,0].get_position().extents[3]
    fig.text(0.005, ax1+(ax0-ax1)/2., 'Black Sky Albedo', rotation=90, ha='left', va='center', fontsize='large')
    fig.text(0.005, ax3+(ax2-ax3)/2., 'White Sky Albedo', rotation=90, ha='left', va='center', fontsize='large')
    if show:
      fig.show()
    else:
      fig.savefig('final_plots/MODELvMODIS.%s.pdf' % lidar)
      plt.close()
    ##
    ## RT model bias against zenith comparison
    ##
    fig, ax = plt.subplots(4, 3, figsize=(6,8), sharey=True)
    for wvi in wv:
      for sai in sa:
        for rt in ['Sellers', 'SPARTACUS']:
          if wvi == 'vis':
            c = 0
          elif wvi == 'nir':
            c = 1
          else:
            c = 2
          if rt == 'Sellers':
            if sai == 'BSA':
              r = 0
            else:
              r = 2
          else:
            if sai == 'BSA':
              r = 1
            else:
              r = 3
          lab = '%s_%s' % (sai, wvi)
          dfi = df[['zen', 'MODIS_%s' % lab, '%s_%s' % (rt, lab)]].dropna()
          dfi['%s_%s_bias' % (rt, lab)] = dfi['%s_%s' % (rt, lab)] - dfi['MODIS_%s' % lab]
          sns.histplot(dfi, x='zen', y='%s_%s_bias' % (rt, lab), ax=ax[r,c], binwidth=(0.1, 0.01))
          ax[r,c].plot([0,90], [0,0], c='k', lw=.5)
          res = linregress(dfi['zen'], dfi['%s_%s_bias' % (rt, lab)])
          ax[r,c].plot([0, 90], res.slope*np.array([0,90])+res.intercept, c='k')
          if c==0:
            ax[r,c].set_ylabel(rt)
          else:
            ax[r,c].set_ylabel('')
          if r==3:
            ax[r,c].set_xlabel('Solar Zenith Angle (°)')
          else:
            ax[r,c].set_xlabel('')
          ax[r,c].set_xlim(0,90)
          ax[r,c].set_ylim(-1,1)
          ax[r,c].set_xticks([0,30,60,90])
          ax[r,c].set_yticks([-1, 0, 1])
          ax[r,c].text(.95, .125, 'm = %.2f' % (res.slope), va='bottom', ha='right')
          ax[r,c].text(.95, .05, 'R² = %.2f' % (res.rvalue**2), va='bottom', ha='right')
    ax[0,0].set_title('Visible')
    ax[0,1].set_title('NIR')
    ax[0,2].set_title('Broadband')
    fig.tight_layout()
    ax0 = ax[0,0].get_position().extents[1]
    ax1 = ax[1,0].get_position().extents[3]
    ax2 = ax[2,0].get_position().extents[1]
    ax3 = ax[3,0].get_position().extents[3]
    fig.text(0.005, ax1+(ax0-ax1)/2., 'Black Sky Albedo', rotation=90, ha='left', va='center', fontsize='large')
    fig.text(0.005, ax3+(ax2-ax3)/2., 'White Sky Albedo', rotation=90, ha='left', va='center', fontsize='large')
    if show:
      fig.show()
    else:
      fig.savefig('final_plots/zenithbias.%s.pdf' % lidar)
      plt.close()
    """
