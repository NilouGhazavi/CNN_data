#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:57:13 2025

@author: niloughazavi
"""

import numpy as np
import os
from scipy import io
import re
import h5py
import math
import matplotlib.pyplot as plt
# from global_scripts import spiketools
import gc
# import torch
from tqdm import tqdm
import warnings
from typing import Tuple
import pickle
import math
import hdf5storage
import cv2



def MEA_spikerates_binned(spikeCounts,sig):
    # sig is input in units of "bins" or "frames"
    # sig = 2
    sig_ms = sig*t_frame     # t_bin has to be in ms. Tells approx. how many ms per bin / frame. The amp is based on ms.
    # st = 0.5
    sr = 60
    st = 10000/sr*6.2/(60*sig_ms)
    
    time_list = np.arange(-3.1,3.1,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig_ms*math.exp((1-time_list[i]**2)/2)
    
    # plt.plot(kern)
    
    spikeRate = np.convolve(spikeCounts,kern,'same')
    # plt.plot(spikeRate)

    return spikeRate
   

def applyLightIntensities(meanIntensity,data,t_frame):

    X = data
    X = X.astype('float32')
    
    X = (X*meanIntensity) + meanIntensity +  (meanIntensity/300)
    
    # X = X * ((2*meanIntensity) + ((2*meanIntensity)/300))
    X = X * 1e-3 * t_frame  # photons per time bin 
    data = X
    return data




def create_dataset(grp,stim_frames,spikeRate,spikeRate_orig,spikeRate_median,unique_movs,t_frame,N_TRIALS):

    d = grp.create_dataset('stim_frames',data=stim_frames[0],compression='gzip')
    d.attrs['t_frame'] = t_frame
    d.attrs['N_TRIALS'] = N_TRIALS
    d.attrs['stim_id'] = unique_movs

    
    d = grp.create_dataset('flips_timestamp',data=stim_frames[1],compression='gzip')
    d.attrs['time_unit'] = 'ms' 
    
    d = grp.create_dataset('spikeRate',data=spikeRate[0],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['sig'] = sig
    d.attrs['N_TRIALS'] = N_TRIALS
    d.attrs['stim_id'] = unique_movs

    
    d = grp.create_dataset('spikeCounts',data=spikeRate[1],compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['N_TRIALS'] = N_TRIALS
    d.attrs['stim_id'] = unique_movs

    
    d = grp.create_dataset('spikeRate_orig',data=spikeRate_orig,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['N_TRIALS'] = N_TRIALS
    d.attrs['stim_id'] = unique_movs


    d = grp.create_dataset('spikeRate_median',data=spikeRate_median,compression='gzip')
    d.attrs['bins'] = 'bin edges defined by dataset flip_times'
    d.attrs['num_units'] = len(uname_all)
    d.attrs['N_TRIALS'] = N_TRIALS
    d.attrs['stim_id'] = unique_movs 
 