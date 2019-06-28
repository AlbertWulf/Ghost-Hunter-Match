# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:54:02 2019

@author: Lenovo
"""

import h5py
f = h5py.File('ztraining-0.h5')
wfl = f['Waveform']
len(wfl)
ent = wfl[820300]
f.close()
w=ent['Waveform']

import pandas as pd
eid=ent['EventID']; ch=ent['ChannelID']
th = pd.read_hdf("ztraining-0.h5", "GroundTruth")
pe = th.query("EventID=={} & ChannelID=={}"
              .format(ent["EventID"], ent["ChannelID"]))
pt = pe['PETime'].values

from matplotlib import pylab as plt
plt.clf()
plt.figure(figsize=(8,4)) 
tr = range(300, 400)
plt.plot(tr, w[tr])
plt.xlabel('ns'); plt.ylabel('mV')
plt.vlines(pt+7, ymin=900, ymax=970)
plt.title("Waveform with Labels")