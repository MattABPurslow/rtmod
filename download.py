import os
import pandas as pd
from multiprocessing import Pool

farr = pd.read_csv('zip.links', header=None).values.T[0]

def wget(f):
  os.system('wget %s &' % f)

with Pool(10) as pool:
  pool.map(wget, farr)
