# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:32:29 2019

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
fipt = "ztraining-0.h5"
#fopt = "first-submission-thres.h5"
#
fi = h5py.File(fipt)
sk = fi['Waveform']
#print(len(wfl))
#ent = wfl[100]
gt = pd.read_hdf(fipt, "GroundTruth")
t_y = np.floor(np.zeros([500000,30]),dtype=np.float32)
for i in range(0,500000):
  ci = sk[i]["ChannelID"]
  ei = sk[i]["EventID"]
  pe = gt.query("EventID=={} & ChannelID=={}".format(ei,ci))
  pt = pe["PETime"]
  l_pt = len(pt)
  if i%5000==0:
    print(i)
  #print(l_pt)
  if l_pt <30:
    t_y[i][0:l_pt] = pt
  else:
   t_y[i] = np.floor([pt[0:30]],dtype=np.float32)
#ped = np.mean(ent['Waveform'][0:150])
#
#eid=ent['EventID']; ch=ent['ChannelID']
#th = pd.read_hdf(fipt, "GroundTruth")
#pe = th.query("EventID=={} & ChannelID=={}"
#              .format(ent["EventID"], ent["ChannelID"]))
#pt = pe['PETime'].values
#
#wave = np.array(ped-ent['Waveform'])
#threshold = 15
#plt.plot(wave,color = "r")
#wave[wave<threshold] = 0;
#smooth_wave = np.zeros(len(wave))
#sum = 1#几点平滑
#for i in range(sum,len(wave)-sum):
#    #print(math.floor(np.mean(wave[i-sum:i+sum])))
#    smooth_wave[i] = math.floor(np.mean(wave[i-sum:i+sum]))
#    #smooth_wave[i] = wave[i]
#for i in range(len(wave)-sum,len(wave)):
#    smooth_wave[i] = wave[i]
#rn = range(250,400)
#
##plt.savefig("smooth.png",dpi=300)
#ppf = signal.argrelextrema(wave[rn], np.greater)
#for i in range(0,7):
#    ppf[0][i] = ppf[0][i]+250
##smooth_wave = np.array(smooth_wave, dtype=np.int8)
#plt.plot(rn,wave[rn])
#plt.plot(rn,np.diff(smooth_wave)[rn],color="r")
#plt.vlines(ppf, ymin=0, ymax=300,colors="g")
#plt.savefig("diff_smooth.png",dpi=300)
##fi.close()
#w = ent['Waveform']
#
#eid=ent['EventID']; ch=ent['ChannelID']
#th = pd.read_hdf(fipt, "GroundTruth")
#pe = th.query("EventID=={} & ChannelID=={}"
#              .format(ent["EventID"], ent["ChannelID"]))
#pt = pe['PETime'].values
#
#plt.clf();
#tr = range(250, 450)
#plt.plot(tr, w[tr])
#plt.xlabel('ns'); plt.ylabel('mV')
#plt.vlines(pt, ymin=930, ymax=970)
#plt.title("Waveform with Labels")
#
#w01 = w<962
#plt.plot(tr, w01[tr])
#plt.title("01 Series after Thresholding")
#plt.ylabel("Lower than 962")
#
#plt.plot(tr, np.diff(w01)[tr])
#
#w01i = np.array(w01, dtype=np.int8)
#d01i = np.diff(w01i)
#plt.plot(tr, d01i[tr])
#
#pf = np.where(d01i>=1)[0]
#plt.plot(tr, w[tr])
#plt.xlabel('ns'); plt.ylabel('mV')
#plt.vlines(pf, ymin=930, ymax=970,colors="r")
#plt.vlines(pt, ymin=930, ymax=970)
#plt.title("PE Search by Thresholds")
#
#def hhd(num):
#    ent = wfl[num]
#    w = ent['Waveform']
#
#    eid=ent['EventID']; ch=ent['ChannelID']
#    th = pd.read_hdf(fipt, "GroundTruth")
#    pe = th.query("EventID=={} & ChannelID=={}"
#              .format(ent["EventID"], ent["ChannelID"]))
#    pt = pe['PETime'].values
#    tr = range(250, 450)
#    w01 = w<962
#    w01i = np.array(w01, dtype=np.int8)
#    d01i = np.diff(w01i)
#    pf = np.where(d01i>=1)[0]
#    plt.clf();
#    plt.plot(tr, w[tr])
#    plt.xlabel('ns'); plt.ylabel('mV')
#    plt.vlines(pf, ymin=930, ymax=970,colors="r")
#    plt.vlines(pt, ymin=930, ymax=970)
#    plt.title("PE Search by Thresholds")
#    print(np.mean(w))
#    
#hhd(333)
#    
#    
#
#
#import numpy as np 
#import pylab as pl
#import matplotlib.pyplot as plt
#import scipy.signal as signal
#x=np.array([
#    0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
#    1, 20, 7, 3, 0 ])
#plt.figure(figsize=(16,4))
#plt.plot(np.arange(len(x)),x)
#print (x[signal.argrelextrema(x, np.greater)])
#print (signal.argrelextrema(x, np.greater))
#
#plt.plot(signal.argrelextrema(x,np.greater)[0],x[signal.argrelextrema(x, np.greater)],'o')
#plt.plot(signal.argrelextrema(-x,np.greater)[0],x[signal.argrelextrema(-x, np.greater)],'+')
## plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
#plt.show()







#opd = [('EventID', '<i8'), ('ChannelID', '<i2'),
#       ('PETime', 'f4'), ('Weight', 'f4')]
#global a
#a = 0
#
#
#
#def Final_One(ent):
#    global a
#    ped = np.mean(ent['Waveform'][0:150])
#    #print(ped)
#    wave = np.array(ped-ent['Waveform'])
#    #print(wave)
#    #plt.plot(wave)
#    wave = wave + np.random.uniform(0,0.1,1029)
#    threshold = 1.5
#    wave[wave<threshold] = 0
#    #print(wave)
#    
#    #plt.plot(wave)
#    extre_pos = signal.argrelextrema(wave, np.greater)
#    #print(extre_pos)
#    extre_value = wave[extre_pos]
#    #print(extre_value)
#    ep = np.array([])
#    temp = np.array(extre_pos)
#    ep = temp[0,:]
#    #print(ep)
#    ex_min = np.min(extre_value)
#    #print(ex_min)
#    #ex_max = np.max(extre_value)
#    
#    k = 0
#    for i in range(0,len(ep)):
#        #pf[range(k,k+math.floor(extre_value[i]/ex_min)+1)] = ep[i]
#        k =k + math.floor(extre_value[i]/ex_min)
#    pf = np.zeros(k)
#    j = 0
#    for i in range(0,len(ep)):
#        pf[range(j,j+math.floor(extre_value[i]/ex_min))] = ep[i]
#        j =j + math.floor(extre_value[i]/ex_min)
#        
#    if not len(pf):
#        pf = np.array([300])
#        
#    rst = np.zeros(len(pf), dtype=opd)
#    rst['PETime'] = pf - 7
#    rst['Weight'] = 1
#    rst['EventID'] = ent['EventID']
#    rst['ChannelID'] = ent['ChannelID']
#    #print(a)
#    if a % 100000 == 99999:
#        print(a+1)
#      
#    a = a +1
#    return rst
#
##fipt = "zincm-problem.h5"
#fipt = "ztraining-0.h5"
#fopt = "final-first-submission.h5"
#
#hhd = h5py.File(fipt)
#th = hhd["Waveform"]
#
#
#with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
#    dt = np.concatenate([Final_One(con) for con in ipt['Waveform']])
#    opt.create_dataset('Answer', data=dt, compression='gzip')
#
#
#    
