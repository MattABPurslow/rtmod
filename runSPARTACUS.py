import glob, os
from multiprocessing import Pool

def runSPARTACUS(inFile):
  outFile = inFile.replace('spartacusIn', 'spartacusOut')
  command = ['spartacus_surface', 'config.nam', inFile, outFile]
  if os.path.exists(outFile)==False:
    try:
      os.system(' '.join(command))
    except:
      pass

if __name__=='__main__':
  inDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/spartacusIn'
  inList = sorted(glob.glob(os.path.join(inDir, '*.nc')))
  print(len(inList), 'files to run')
  with Pool(16) as pool:
    pool.map(runSPARTACUS, inList)
