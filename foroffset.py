# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:16:02 2019

@author: Lenovo
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
from matplotlib import pylab as plt
import math
import matplotlib.pyplot as plt
import scipy.signal as signal

fr = "ztraining-0.h5"
f = h5py.File(fr)
ww = f['Waveform']
gg = f["GroundTruth"]
wave = ww[2890]

wlf = wave["Waveform"]
print(wave["EventID"])
print(wave["ChannelID"])
ped = np.mean(wlf)
wave = ped-wlf
wave[wave<5]=0
plt.plot(range(280,380),wave[280:380])
