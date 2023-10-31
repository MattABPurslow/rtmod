import os, glob
import datetime
from multiprocessing import Pool

def lazzip(f):
  os.system('wine64 /geos/netdata/LAStools/bin/laszip64.exe %s' % f)

def gatherLAS(alsDir):
  _=os.system('mv %s/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*/*/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*/*/*/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*/*/*/*/*.las %s' % (alsDir, alsDir))
  _=os.system('mv %s/*/*/*/*/*/*/*.las %s' % (alsDir, alsDir))


def mapLidar(f):
  os.system('mapLidar -input %s -res 5 -float -epsg 32635 -height -output %s'\
             % (f, f.replace('.las', '.dsm')))
  os.system('mapLidar -input %s -res 5 -float -epsg 32635 -DTM -output %s'\
            % (f, f.replace('.las', '.dtm')))
  os.system('mapLidar -input %s -res 5 -float -epsg 32635 -cover -output %s'\
            % (f, f.replace('.las', '.cv')))

def warp(f):
  os.system('gdalwarp -t_srs "+proj=sinu +lon_0=0 + x_0=0 +y_0=0 +a=6371007.181\  +b=6371007.181 +units=m +no_defs" -dstnodata -999.0 -r near -of GTiff %s %s'\
  % (f, f.replace('.tif', '.modisgrid.tif')))

if __name__=='__main__':
  alsDir = '/exports/csce/datastore/geos/groups/3d_env/data/purslowm/als'
  zipList = glob.glob('%s/*-*' % alsDir)
  print('Unzipping .zip files @', datetime.datetime.now())
  #command = ' & '.join(['unzip %s -d %s' % (z, alsDir) for z in zipList])
  #os.system(command)
  lazList = glob.glob('%s/*/*/*/*/*/*.laz' % alsDir)
  with Pool(16) as pool:
    print('Unzipping .laz files @', datetime.datetime.now())
    pool.map(lazzip, lazList)
    os.system('mv %s/*/*/*/*/*/*.las %s' % (alsDir, alsDir))
"""
    print('Creating DSM, DTM & cover tiles @', datetime.datetime.now())
    lasList = glob.glob('%s/*.las' % alsDir)
    pool.map(mapLidar, lasList)
    print('Reprojecting to MODIS sinusoid @', datetime.datetime.now())
    tifList = glob.glob('%s/*.tif' % alsDir) 
    pool.map(warp, tifList) 
    print('Merging tiles @', datetime.datetime.now())
    os.system('gdal_merge.py -o /exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.cv.modisgrid.tif %s/cv/*.tif' % alsDir)
    os.system('gdal_merge.py -o /exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.dsm.modisgrid.tif %s/dsm/*.tif' % alsDir)
    os.system('gdal_merge.py -o /exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/sodankyla.dtm.modisgrid.tif %s/dtm/*.tif' % alsDir)
"""
