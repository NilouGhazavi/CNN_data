#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:23:32 2021

@author: saad


The dataset contains a total of 9 movies. We are going to take the first 7 movies for training and last 2 for validation.

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
 
# %% exp details
expDate = '20230725C'
UP_SAMP = 0
bin_width = 8 #8   # ms
sig = 4#8 # 2   #2 for 16 ms; 4 for 8 ms; 8 for 4 ms; 20 for 1 ms



CONVERT_RSTAR = True
NORM_STIM = 1
NORM_RESP = True
FRAME_RESIZE = 8
stim_upsample = 2#2
t_frame = (1/60.3180)*1000/stim_upsample    # because spikerate is sampled at 120 but stim at 60Hz. So we will upsample stim later. So each frame is 8 ms

# Load the data.
lightLevel_text = ['scotopic','mesopic','photopic']
lightLevel_mul = [0.5,50,6000]

# index of validation movie
idx_movs_val = [0]      #3 [2,3,4,6,7,8]
idx_movs_train = np.setdiff1d(np.arange(0,9),idx_movs_val)


path_raw = os.path.join('D:\\Nilou\\GitHub\Gradients\\RGC_Selective_Simulation\\Natural_Movies\\analyses_parasol_midget_84cells\\data_mike_nat',expDate,'raw')
path_movs = os.path.join(path_raw,'updated_movie_files')
path_oldmovs=os.path.join(path_raw,'bugged_movie_files')
path_db = os.path.join('D:\\Nilou\\GitHub\Gradients\\RGC_Selective_Simulation\\Natural_Movies\\analyses_parasol_midget_84cells\\data_mike_nat',expDate,'datasets')
path_save = os.path.join('D:\\Nilou\\GitHub\Gradients\\RGC_Selective_Simulation\\Natural_Movies\\analyses_parasol_midget_84cells\\data_mike_nat',expDate,'datasets')



if CONVERT_RSTAR==True:
    fname_save = os.path.join(path_save,(expDate+'_dataset_NATSTIM'+str(idx_movs_val[0])+'_CORR_'+'allLightLevels'+'_'+str(bin_width)+'ms'+'_Rstar.h5'))
else:
    fname_save = os.path.join(path_save,(expDate+'_dataset_NATSTIM'+str(idx_movs_val[0])+'_CORR_'+'allLightLevels'+'_'+str(bin_width)+'ms.h5'))


if not os.path.exists(path_save):
    os.mkdir(path_save)

#%% cell IDs 

N_VAL = len(idx_movs_val)   # 2 movies for validation purpose




mesopic_cell_dove = {
'OffM': [11, 78, 158, 347, 423, 636, 643, 661, 678, 712, 732, 748, 783, 787, 809],
'OffP': [5,52,138,360, 498, 49, 632, 648, 713, 721, 726, 741,785],
'OnM': [44, 47, 48, 56, 65, 84, 90, 93, 102, 120, 121,136, 142, 150, 183, 197, 225, 233, 250, 261, 284, 294, 318, 384, 440, 460, 494, 497, 511, 515, 520, 534, 538, 549, 572, 595, 619, 769, 792, 819],
'OnP': [140, 172, 196, 207, 223, 367, 445, 533, 539, 571, 630, 651, 752, 764, 969, 980],
}



mesopic_off_m = mesopic_cell_dove['OffM']
mesopic_off_p = mesopic_cell_dove['OffP']
mesopic_on_m = mesopic_cell_dove['OnM']
mesopic_on_p = mesopic_cell_dove['OnP']


# only parasol cells 
uname_all = list()
ctr = 0
for i in mesopic_on_p:
    ctr+=1
    uname_rgb = 'on_par_%03d'%ctr
    uname_all.append(uname_rgb)
    
ctr = 0
for i in mesopic_off_p:
    ctr+=1
    uname_rgb = 'off_par_%03d'%ctr
    uname_all.append(uname_rgb)

ctr = 0
for i in mesopic_on_m:
    ctr+=1
    uname_rgb = 'on_m_%03d'%ctr
    uname_all.append(uname_rgb)

ctr = 0
for i in mesopic_off_m:
    ctr+=1
    uname_rgb = 'off_m_%03d'%ctr
    uname_all.append(uname_rgb)



totalNum_units = len(mesopic_on_p) + len(mesopic_off_p)+len(mesopic_on_m)+len(mesopic_off_m)


mesopic_ids=mesopic_on_p+mesopic_off_p+mesopic_on_m+mesopic_off_m
mesopic_ids=np.array(mesopic_ids)

idx_allunits=mesopic_ids







#%% main loop 
# % Main loop
spikerate_grand = np.empty((0,totalNum_units,9,10))      # [time,cells,stims,trials]   <-- nee to find a way to do this auto
lightLevel_idx = 1

for lightLevel_idx in [1]:     #[0=0.3; 1=3; 2=30]
    select_lightLevel = lightLevel_text[lightLevel_idx]

    fname = os.path.join(path_raw,expDate+'_Doves_'+select_lightLevel+'.pkl')
    with open(fname,'rb') as f:
        params = pickle.load(f)
    
    #---- Stim 
    
    spike_dict = params['spike_dict']
    cluster_id = params['cluster_id']
    stimulus_index = params['stimulusIndex'].astype(int)
    background_intensity = params['backgroundIntensity'].astype(float)
    frame_times = params['frame_times'] # This is the time of each frame in milliseconds.
    pre_pts = params['pre_pts']
    stim_pts = params['stim_pts']
    tail_pts = params['tail_pts']
    wait_time = params['waitTime']
    bin_rate = params['bin_rate'] # This is the bin rate for the spike times in Hz.


    unique_movs = np.unique(stimulus_index)
    N_TRIALS = int(len(stimulus_index)/len(unique_movs))
    frames_mat = np.zeros((362,int(600/FRAME_RESIZE),int(800/FRAME_RESIZE),len(unique_movs)));frames_mat[:] = np.nan        # [time,y/8,x/8,img,trial]
    spikes_mat = np.zeros((len(unique_movs),N_TRIALS),dtype='object')
    ctr_trial = -1*np.ones(N_TRIALS,dtype='int')
    
    # Loop over movies
    m=5
    for m in tqdm(range(len(unique_movs))):
        movie_idx = unique_movs[m]
        movie_loc = np.where(movie_idx==unique_movs)[0][0]
        
        old_movie_path = os.path.join(path_oldmovs, 'doves_idx_' + str(movie_idx) + '.mat')
        movie_dic = hdf5storage.loadmat(old_movie_path)

        single_movie_path = os.path.join(path_movs, 'doves_idx_' + str(movie_idx) + '.npy')
        frames = np.load(single_movie_path)
    
        bg_mean = movie_dic['image_mean']
        # frames = cv2.resize(frames, dsize=(int(frames.shape[1]/FRAME_RESIZE), int(frames.shape[0]/FRAME_RESIZE)), interpolation=cv2.INTER_LINEAR)
        frames = frames[:,::FRAME_RESIZE,::FRAME_RESIZE,0]
        frames = (frames+frames.max())/(frames.max()-frames.min())
        
        frames_new = (frames - bg_mean) / bg_mean       # convert to  contrast
        
        frames_mat[:,:,:,movie_loc] = frames_new

        frame_rate = movie_dic['frame_rate'][0]


    if CONVERT_RSTAR == True:
        meanIntensity = lightLevel_mul[lightLevel_idx]
        frames_rstar = applyLightIntensities(meanIntensity,frames_mat,t_frame)
        
    stim_frames = np.repeat(frames_rstar,stim_upsample,axis=0).astype('float32')
    stim_frames = np.repeat(stim_frames[:,:,:,:,None],N_TRIALS,axis=-1)

    # Extract spikes
    epoch_count = 0
    for epoch_count in tqdm(range(len(stimulus_index))):
        movie_idx = int(stimulus_index[epoch_count])
        movie_loc = np.where(movie_idx==unique_movs)[0][0]
        ctr_trial[movie_loc] += 1

        epoch_frame_times = np.array(frame_times[epoch_count])  # Get the frame times for the epoch.
        epoch_frame_times -= epoch_frame_times[0]               # Subtract the first frame.
        stim_start = pre_pts[epoch_count]                       # The stimulus starts at the end of pre_pts.
        the_wait_is_over = stim_start + wait_time[epoch_count]  # The first movie frame is then presented for the duration of the wait time.
        stim_end = stim_start + stim_pts[epoch_count]           # The stimulus then plays for the duration of the stim_pts-wait_time.
    
        # Get the relevant spikes
        spikes_epoch = spike_dict[:,epoch_count]
        spikes_stim = []
        i=0
        for i in range(len(spikes_epoch)):
             rgb = spikes_epoch[i]
             rgb = rgb[rgb>stim_start]
             rgb = rgb[rgb<stim_end]
             rgb = rgb - stim_start
             rgb = rgb[rgb>0]
             spikes_stim.append(rgb)
        spikes_mat[movie_loc,ctr_trial[movie_loc]] = spikes_stim
    
    

    # ---- spikes
    assert len(np.unique(stim_pts))==1, 'stim_pts different for different epochs/movies'

    stimLength = stim_frames.shape[0]
    
    flips = np.arange(0,(stimLength+1)*t_frame,t_frame)
    
    numStims = len(unique_movs)
    spikeRate_cells = np.empty((stimLength,totalNum_units,numStims,N_TRIALS))
    spikeCounts_cells = np.empty((stimLength,totalNum_units,numStims,N_TRIALS))
    spikeTimes_cells = np.empty((totalNum_units,numStims,N_TRIALS),dtype='object')
    
    
    ctr_units = -1
    U = 0
    # had to be modified for one light level
    for U in range(0, len(idx_allunits)):
        ctr_units += 1
        cluster_unit = idx_allunits[U]
        idx_unit_array = np.where(cluster_id == cluster_unit)[0]
        # if len(idx_unit_array) == 0:
        #     continue  # Skip if cluster_unit is not found in cluster_id
        idx_unit = idx_unit_array[0]  # Convert array to integer
        
        tr = 0
        s = 0
        for s in range(numStims):
            for tr in range(N_TRIALS):
                startTime = 0  # Start time for stimulus
                endTime = startTime + flips[-1]  # End time for stimulus
                
                spikes = spikes_mat[s, tr][idx_unit]
                if len(spikes) == 0:
                    spikes = 0
                    spikeCounts = np.zeros(stimLength)
                else:
                    spikeCounts = np.histogram(spikes, flips)[0]  # Bin the spikes
                spikeRate = MEA_spikerates_binned(spikeCounts, sig)  # Calculate spike rates
                
                spikeRate_cells[:, ctr_units, s, tr] = spikeRate
                spikeCounts_cells[:, ctr_units, s, tr] = spikeCounts
                spikeTimes_cells[ctr_units, s, tr] = spikes
    


    # Normalize responses if required
    if NORM_RESP:
        rgb = np.squeeze(spikeRate_cells)
        rgb[rgb == 0] = np.nan
        # resp_median = np.nanmedian(rgb, axis=0)
        # nilou added
        if np.all(np.isnan(rgb)):
            resp_median = np.zeros_like(rgb[0])
        else:
            resp_median = np.nanmedian(rgb, axis=0)
        resp_norm = rgb / resp_median[None, :]
        resp_norm[np.isnan(resp_norm)] = 0
        resp_orig = spikeRate_cells
    else:
        resp_norm = spikeRate_cells
        resp_orig = spikeRate_cells

    # Prepare training and validation datasets
    idx_frames_start = 0  # Discard first 121 frames
    stim_frames_train = (stim_frames[idx_frames_start:, :, :, idx_movs_train, :], flips[idx_frames_start:])
    spikeRate_train = (resp_norm[idx_frames_start:, :, idx_movs_train, :], spikeCounts_cells[idx_frames_start:, :, idx_movs_train, :], spikeTimes_cells[:, idx_movs_train, :])
    spikeRate_orig_train = spikeRate_cells[idx_frames_start:, :, idx_movs_train, :]
    spikeRate_median_train = resp_median[:, idx_movs_train, :]
    unique_movs_train = unique_movs[idx_movs_train]

    stim_frames_val = (stim_frames[idx_frames_start:, :, :, idx_movs_val, :], flips[idx_frames_start:])
    spikeRate_val = (resp_norm[idx_frames_start:, :, idx_movs_val, :], spikeCounts_cells[idx_frames_start:, :, idx_movs_val, :], spikeTimes_cells[:, idx_movs_val, :])
    spikeRate_orig_val = spikeRate_cells[idx_frames_start:, :, idx_movs_val, :]
    spikeRate_median_val = resp_median[:, idx_movs_val, :]
    unique_movs_val = unique_movs[idx_movs_val]
    
    # Update grand spikerate
    spikerate_grand = np.concatenate((spikerate_grand, spikeRate_cells), axis=0)

    # Save dataset
    
        # Create directories if necessary before saving
    os.makedirs(os.path.dirname(fname_save), exist_ok=True)
    
    with h5py.File(fname_save, 'a') as f:
        try:
            # Save experimental date and unit information
            f.create_dataset('expDate', data=np.array(expDate, dtype='bytes'))
            f.create_dataset('units', data=np.array(uname_all, dtype='bytes'))
        except:
            pass  # If datasets already exist, skip creating them
    
        # Save training data
        grp = f.create_group('/' + select_lightLevel + '/train')
        create_dataset(grp, stim_frames_train, spikeRate_train, spikeRate_orig_train, spikeRate_median_train, unique_movs_train, t_frame, N_TRIALS)
    
        # Save validation data
        grp = f.create_group('/' + select_lightLevel + '/val')
        create_dataset(grp, stim_frames_val, spikeRate_val, spikeRate_orig_val, spikeRate_median_val, unique_movs_val, t_frame, N_TRIALS)
    
    # Flatten and reshape spike rate data for saving
    spikerate_grand_flattened = np.moveaxis(spikerate_grand, 0, -1)
    spikerate_grand_flattened = spikerate_grand_flattened.reshape(spikerate_grand_flattened.shape[0], -1)
    resp_median_grand = np.nanmedian(spikerate_grand_flattened, axis=-1)
    
    # Save grand spike rate and median response data
    with h5py.File(fname_save, 'a') as f:
        if 'spikerate_grand' in f:
            del f['spikerate_grand']  # Delete existing dataset if it exists
        if 'resp_median_grand' in f:
            del f['resp_median_grand']  # Delete existing dataset if it exists
        f.create_dataset('spikerate_grand', data=spikerate_grand_flattened, compression='gzip')
        f.create_dataset('resp_median_grand', data=resp_median_grand, compression='gzip')