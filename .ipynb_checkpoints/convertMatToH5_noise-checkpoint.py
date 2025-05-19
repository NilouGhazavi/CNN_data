#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 4


@author: saad
"""

#%% import libraries 
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
# Change the current directory
os.chdir(r"D:\Nilou\GitHub\Gradients\RetinaPredictors-main")

# Verify the change
print("Current working directory:", os.getcwd())

from model.load_savedModel import *
from model.utils_si import *
import model.data_handler
from model.data_handler import load_data, prepare_data_cnn2d, prepare_data_cnn3d, prepare_data_convLSTM, prepare_data_pr_cnn2d,merge_datasets,isintuple
from model.data_handler_mike import load_h5Dataset
from model.performance import getModelParams, model_evaluate,model_evaluate_new,paramsToName, get_weightsDict, get_weightsOfLayer,estimate_noise
from model import metrics
from model import featureMaps
from model.models import modelFileName
from model.train_model import chunker
import model.gradient_tools
from model.featureMaps import spatRF2DFit, get_strf, decompose
from model.train_model import chunker
from model.train_model import chunker
#%% Functions 
def MEA_spikerates_binned(spikeCounts,sig):

    sig_ms = sig*t_frame     # t_bin has to be in ms. Tells approx. how many ms per bin / frame. The amp is based on ms.
    
    sr = 60
    st = 10000/sr*6.2/(60*sig_ms)
    
    time_list = np.arange(-3.1,3.1,st)
    kern = np.zeros((len(time_list)))
    
    for i in range(len(time_list)):
        kern[i] = 250/sig_ms*math.exp((1-time_list[i]**2)/2)
    
    plt.plot(kern)
    
    spikeRate = np.convolve(spikeCounts,kern,'same')
    plt.plot(spikeRate)

    return spikeRate




def applyLightIntensities(meanIntensity,data,t_frame):

    X = data
    X = X.astype('float32')
    
    X = (X*meanIntensity) + meanIntensity + (meanIntensity/300)
    
    # idx_low = X<0.5
    # idx_high = X>0.5
    # X[idx_high] = 2*meanIntensity
    # X[idx_low] = (2*meanIntensity)/300
    
    X = X * 1e-3 * t_frame  # photons per time bin 

        
    data = X
    return data

# Get the frame sequence for the FastNoiseStimulus
def get_spatial_noise_frames(numXStixels: int,
                        numYStixels: int,
                        numXChecks: int,
                        numYChecks: int,
                        chromaticClass: str,
                        numFrames: int,
                        stepsPerStixel: int,
                        seed: int,
                        frameDwell: int=1) -> np.ndarray:
    """
    
        numXStixels = int(params['numXStixels'][idx])
        numYStixels = int(params['numYStixels'][idx])
        numXChecks = int(params['numXChecks'][idx])
        numYChecks = int(params['numYChecks'][idx])
        chromaticClass = params['chromaticClass'][idx]
        numFrames = int(params['numFrames'][idx])
        stepsPerStixel = int(params['stepsPerStixel'][idx])
        seed = int(params['seed'][idx])
        frameDwell = int(params['frameDwell'][idx])
    
    Get the frame sequence for the FastNoiseStimulus.
    Parameters:
        numXStixels: number of stixels in the x direction.
        numYStixels: number of stixels in the y direction.
        numXChecks: number of checks in the x direction.
        numYChecks: number of checks in the y direction.
        chromaticClass: chromatic class of the stimulus.
        numFrames: number of frames in the stimulus.
        stepsPerStixel: number of steps per stixel.
        seed: seed for the random number generator.
        frameDwell: number of frames to dwell on each frame.

    Returns:
    frames: 4D array of frames (n_frames, x, y, n_colors).
    """
    # Seed the random number generator.
    np.random.seed( seed )

    # First, generate the larger grid of stixels.
    if (chromaticClass == 'BY'):
        tfactor = 2
    elif (chromaticClass == 'RGB'):
        tfactor = 3
    else: # Black/white checks
        tfactor = 1

    # Get the size of the time dimension; expands for RGB, etc.
    tsize = np.ceil(numFrames*tfactor/frameDwell).astype(int)

    if (tfactor == 2 and (tsize % 2) != 0):
        tsize += 1

    # Generate the random grid of stixels.
    gridValues = np.random.rand(tsize, numXStixels*numYStixels)
    gridValues = np.reshape(gridValues, (tsize, numXStixels, numYStixels))
    gridValues = np.transpose(gridValues, (0, 2, 1))
    gridValues = np.round(gridValues)
    gridValues = (2*gridValues-1).astype(np.float32) # Convert to contrast

    # Translate to the full grid
    fullGrid = np.zeros((tsize,numYStixels*stepsPerStixel,numXStixels*stepsPerStixel), dtype=np.float32)

    for k in range(numYStixels*stepsPerStixel):
        yindex = math.floor(k/stepsPerStixel)
        for m in range(numXStixels*stepsPerStixel):
            xindex = math.floor(m/stepsPerStixel)
            fullGrid[:, k, m] = gridValues[:, yindex, xindex]

    # Generate the motion trajectory of the larger stixels.
    np.random.seed( seed ) # Re-seed the number generator

    # steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps = np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    steps[:,0] = (stepsPerStixel-1) - steps[:,0]
    # steps = (stepsPerStixel-1) - np.round( (stepsPerStixel-1) * np.random.rand(tsize, 2) )
    # Get the frame values for the finer grid.
    # frameValues = np.zeros((tsize,numYChecks,numXChecks),dtype=np.uint8)
    frameValues = np.zeros((tsize,numYChecks,numXChecks),dtype=np.float32)
    for k in range(tsize):
        x_offset = steps[math.floor(k/tfactor), 0].astype(int)
        y_offset = steps[math.floor(k/tfactor), 1].astype(int)
        frameValues[k,:,:] = fullGrid[k, y_offset : numYChecks+y_offset, x_offset : numXChecks+x_offset]

    # Create your output stimulus. (t, y, x, color)
    stimulus = np.zeros((np.ceil(tsize/tfactor).astype(int),numYChecks,numXChecks,3), dtype=np.float32)

    # Get the pixel values into the proper color channels
    if (chromaticClass == 'BY'):
        stimulus[:,:,:,0] = frameValues[0::2,:,:]
        stimulus[:,:,:,1] = frameValues[0::2,:,:]
        stimulus[:,:,:,2] = frameValues[1::2,:,:]
    elif (chromaticClass == 'RGB'):
        stimulus[:,:,:,0] = frameValues[0::3,:,:]
        stimulus[:,:,:,1] = frameValues[1::3,:,:]
        stimulus[:,:,:,2] = frameValues[2::3,:,:]
    else: # Black/white checks
        stimulus[:,:,:,0] = frameValues
        stimulus[:,:,:,1] = frameValues
        stimulus[:,:,:,2] = frameValues
    # return stimulus

    # Deal with the frame dwell.
    if frameDwell > 1:
        stim = np.zeros((numFrames,numYChecks,numXChecks,3), dtype=np.float32)
        for k in range(numFrames):
            idx = np.floor(k / frameDwell).astype(int)
            stim[k,:,:,:] = stimulus[idx,:,:,:]
        return stim
    else:
        return stimulus


# Comparing spike rates and spike counts
# visualize_cell_activity

def visualize_cell_activity(spikeCounts, psth_data, sig, t_frame, frame_rate, window_ms=4800, figsize=(15, 12)):
    """
    Creates 4-panel visualization of neural activity
    """
    # Calculate samples for window
    samples_per_window = int(window_ms / t_frame)
    
    # Create time axis
    time_axis = np.arange(samples_per_window) * t_frame
    
    # Create kernel
    st = 10000/60*6.2/(60*sig*t_frame)
    time_list = np.arange(-3.1, 3.1, st)
    kern = np.zeros(len(time_list))
    
    for i in range(len(time_list)):
        kern[i] = 250/(sig*t_frame)*np.exp((1-time_list[i]**2)/2)
    
    # Calculate spike rate
    spikeRate = np.convolve(spikeCounts, kern, 'same')
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2], hspace=0.4)
    
    # Plot kernel
    ax1 = fig.add_subplot(gs[0])
    kernel_time = np.arange(len(kern)) * st - 3.1
    ax1.plot(kernel_time, kern, 'b-', label='Gaussian Kernel')
    ax1.set_title('Gaussian Kernel', pad=10)
    ax1.set_xlabel('Time (normalized)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # Plot PSTH
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_axis, psth_data[:samples_per_window], 'g-', label='PSTH')
    ax2.set_title('PSTH', pad=10)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Spike Count')
    ax2.grid(True)
    ax2.legend()
    
    # Plot smoothed spike rate
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_axis, spikeRate[:samples_per_window], 'b-', label='Smoothed Rate', alpha=0.8)
    ax3.set_title('Smoothed Spike Rate', pad=10)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Spike Rate')
    ax3.grid(True)
    ax3.legend()
    
    # Plot spike times and rate combined
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(time_axis, spikeRate[:samples_per_window], 'b-', label='Smoothed Rate', alpha=0.8)
    spike_times = np.where(spikeCounts[:samples_per_window])[0] * t_frame
    if len(spike_times) > 0:
        ax4.vlines(spike_times, 0, np.max(spikeRate[:samples_per_window]), 
                  color='r', alpha=0.3, label='Spike Times')
    ax4.set_title('Spike Times and Rate', pad=10)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Spike Rate')
    ax4.grid(True)
    ax4.legend()
    
    return fig, (ax1, ax2, ax3, ax4)



# plot cell response
def plot_cell_response(cell_idx, spikes, psth, cluster_id, sig, t_frame, frame_rate):
    """
    Plots the complete response for a specific cell
    """
    spikeCounts = spikes[cell_idx]
    psth_data = np.mean(psth[cell_idx], axis=0)  # Average across trials
    
    fig, axes = visualize_cell_activity(spikeCounts, psth_data, sig, t_frame, frame_rate)
    plt.suptitle(f'Cell Response Analysis - Cluster ID: {cluster_id[cell_idx]}', y=1.02)
    
    # Save the figure
    save_dir = r"D:\Nilou\GitHub\Gradients\RGC_Selective_Simulation\White_Noise\analyses_43cells_Nilou\data_mike_noise\20230725C\datasets\PSTH_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cell_response_cluster_{cluster_id[cell_idx]}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


# %% exp details
expDate = '20230725C'
UP_SAMP = 0
bin_width = 8   # ms
sig = 4#8 # 2   #2 for 16 ms; 4 for 8 ms; 8 for 4 ms; 20 for 1 ms


CONVERT_RSTAR = True
NORM_STIM = 1
NORM_RESP = True


path_raw = os.path.join('D:\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_43cells_Nilou\\data_mike_noise',expDate,'raw')
path_db = os.path.join('D:\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_43cells_Nilou\\data_mike_noise',expDate,'datasets')
path_save = os.path.join('D:\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_43cells_Nilou\\data_mike_noise\\',expDate,'datasets')
if CONVERT_RSTAR==True:
    fname_save = os.path.join(path_save,(expDate+'_dataset_CB_CORR_'+'allLightLevels'+'_'+str(bin_width)+'ms'+'_Rstar.h5'))
else:
    fname_save = os.path.join(path_save,(expDate+'_dataset_CB_CORR_'+'allLightLevels'+'_'+str(bin_width)+'ms.h5'))


if not os.path.exists(path_save):
    os.mkdir(path_save)

select_colChan = 0
stride = 2
depth = 61 # You'll need to increase this for the mesopic and scotopic noise runs because the kinetics are slower.

# Load the data.
lightLevel_text = ['scotopic','mesopic','photopic']
lightLevel_mul = [0.5,50,6000]

# %% Cell info


mesopic_cell_dove = {
    'OffP': [5,49,52,130,137, 138, 258, 316, 360, 444, 498, 521, 575, 594, 632, 648, 688, 713, 721, 741, 785, 998],
    'OnP': [12, 20,140, 172, 196, 207, 223, 313, 342, 445, 446, 507, 533, 571, 630, 651, 752, 764, 788,969, 980]
}




mesopic_off_p = mesopic_cell_dove['OffP']
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




totalNum_units = len(mesopic_on_p) + len(mesopic_off_p)


mesopic_ids=mesopic_on_p+mesopic_off_p
mesopic_ids=np.array(mesopic_ids)

idx_allunits=mesopic_ids



# %% Main loop

spikerate_grand = np.empty((0,totalNum_units,1))
idx_allunits= mesopic_ids                           
lightLevel_idx = 1


for lightLevel_idx in [1]:     #[0=0.3; 1=3; 2=30]
    select_lightLevel = lightLevel_text[lightLevel_idx]

    fname = os.path.join(path_raw,expDate+'_Noise_'+select_lightLevel+'.pkl')
    with open(fname,'rb') as f:
        params = pickle.load(f)
    
    psth = params['psth']
    cluster_id = params['cluster_id']
    pre_pts = params['pre_pts']
    stim_pts = params['stim_pts']
    frame_rate = params['frame_rate'] # The precise update rate of the monitor computed from the timing frame flips.
    num_checkers_x = int(np.unique(params['numXChecks']))
    num_checkers_y = int(np.unique(params['numYChecks']))
    checkSize_um = 30

    t_frame = (1/frame_rate)*1000/2    # because spikerate is sampled at 120 but stim at 60Hz. So we will upsample stim later. So each frame is 8 ms
    
    binned_spikes = psth[:,:,pre_pts[0]:pre_pts[0]+int(params['numFrames'][0]*stride)]
    
    num_units_inDataset = psth.shape[0]

    # ---- stim
    idx = 1
    stim_frames = np.zeros((0,75,100))
    spikes = np.zeros((num_units_inDataset,0))
    for idx in tqdm(range(len(params['numXStixels'])), desc=''):
        frames_temp = get_spatial_noise_frames(
            int(params['numXStixels'][idx]),
            int(params['numYStixels'][idx]),
            int(params['numXChecks'][idx]),
            int(params['numYChecks'][idx]),
            params['chromaticClass'][idx],
            int(params['numFrames'][idx]),
            int(params['stepsPerStixel'][idx]),
            int(params['seed'][idx]),
            int(params['frameDwell'][idx]))
        
        # frames_temp[frames_temp>0] = 1
        # frames_temp[frames_temp<0] = 0
        if CONVERT_RSTAR == True:
            meanIntensity = lightLevel_mul[lightLevel_idx]
            frames_temp = applyLightIntensities(meanIntensity,frames_temp,t_frame)


        stim_frames = np.concatenate((stim_frames,frames_temp[:,:,:,select_colChan]))
        spikes_temp = binned_spikes[:,idx,:]
        spikes = np.concatenate((spikes,spikes_temp),axis=-1)
    
    
    stim_frames = np.reshape(stim_frames,(stim_frames.shape[0],stim_frames.shape[1]*stim_frames.shape[2]),order='F')
    
    stim_frames = np.repeat(stim_frames,stride,axis=0)
    assert stim_frames.shape[0] == spikes.shape[-1],'num of frames does not match num of spikes'


    # a = np.unique(stim_frames,axis=0)
    
    
    # ---- spike rates
    stimLength = stim_frames.shape[0]
    
    flips = np.arange(0,(stimLength+1)*t_frame,t_frame)
    
    numTrials = 1
    spikeRate_cells = np.empty((stimLength,totalNum_units,numTrials))
    spikeCounts_cells = np.empty((stimLength,totalNum_units,numTrials))
    spikeTimes_cells = np.empty((totalNum_units,numTrials),dtype='object')
    
    
    ctr_units = -1
    print("Total units to process:", len(idx_allunits))
    print("Cluster IDs available:", len(np.unique(cluster_id)))
    print("\nStarting unit processing...")
    U=0
    skipped_units = []
    found_units=[]
    for U in range(0, len(idx_allunits)):
        ctr_units += 1
        cluster_unit = idx_allunits[U]
        idx_unit_array = np.where(cluster_id==cluster_unit)[0]
        # if len(idx_unit_array) == 0:
        #     skipped_units.append(cluster_unit)
        #     print(f"Skipping unit {cluster_unit} - not found in cluster_id")
        #     continue  # Skip if cluster_unit is not found in cluster_id
        
        # found_units.append(cluster_unit)
        idx_unit = idx_unit_array[0]
        tr=0
        for tr in range(numTrials):
            startTime = 0#stimulus_start_times[tr]
            endTime = startTime + flips[-1]
            
            spikeCounts = spikes[idx_unit]
            spikeRate = MEA_spikerates_binned(spikeCounts,sig)
            # plt.plot(spikeRate)
        
            spikeRate_cells[:,ctr_units,tr] = spikeRate
            spikeCounts_cells[:,ctr_units,tr] = spikeCounts
            spikeTimes_cells[ctr_units,tr] = np.where(spikeCounts)[0]
            
            
            # Add visualization here
            fig, axes = plot_cell_response(
                cell_idx=idx_unit,
                spikes=spikes,
                psth=psth,
                cluster_id=cluster_id,
                sig=sig,
                t_frame=t_frame,
                frame_rate=frame_rate
            )
            plt.show()
        
    if NORM_RESP==True:
        rgb = np.squeeze(spikeRate_cells)
        rgb[rgb==0]=np.nan
        resp_median = np.nanmedian(rgb,axis=0)
        resp_norm = rgb/resp_median[None,:]
        resp_norm[np.isnan(resp_norm)] = 0
        resp_orig = spikeRate_cells
    else:
        resp_norm = spikeRate_cells
        resp_orig = spikeRate_cells
    
    stim_frames_train = (stim_frames,flips)
    spikeRate_train = (resp_norm,spikeCounts_cells,spikeTimes_cells)   # dims [stim_files][0=spikeRate,1=spikeCounts,2=spikeTimes]
    spikeRate_orig = spikeRate_cells
    spikeRate_median = resp_median
    
    spikerate_grand = np.concatenate((spikerate_grand,spikeRate_cells),axis=0)

    # ---- Save dataset
    with h5py.File(fname_save,'a') as f:
        try:
            f.create_dataset('expDate',data=np.array(expDate,dtype='bytes'))
            f.create_dataset('units',data=np.array(uname_all,dtype='bytes'))
        except:
            pass
    
        grp = f.create_group('/'+select_lightLevel+'/train')
        d = grp.create_dataset('stim_frames',data=stim_frames_train[0],compression='gzip')
        d.attrs['num_checkers_x'] = num_checkers_x
        d.attrs['num_checkers_y'] = num_checkers_y
        d.attrs['checkSize_um'] = checkSize_um
        d.attrs['t_frame'] = t_frame
        
        
        d = grp.create_dataset('flips_timestamp',data=stim_frames_train[1],compression='gzip')
        d.attrs['time_unit'] = 'ms' 
        
        d = grp.create_dataset('spikeRate',data=spikeRate_train[0],compression='gzip')
        d.attrs['bins'] = 'bin edges defined by dataset flips_timestamp'
        d.attrs['num_units'] = len(uname_all)
        d.attrs['sig'] = sig
        
        d = grp.create_dataset('spikeCounts',data=spikeRate_train[1],compression='gzip')
        d.attrs['bins'] = 'bin edges defined by dataset flip_times'
        d.attrs['num_units'] = len(uname_all)
        
        d = grp.create_dataset('spikeRate_orig',data=spikeRate_orig,compression='gzip')
        d.attrs['bins'] = 'bin edges defined by dataset flip_times'
        d.attrs['num_units'] = len(uname_all)

        d = grp.create_dataset('spikeRate_median',data=spikeRate_median,compression='gzip')
        d.attrs['bins'] = 'bin edges defined by dataset flip_times'
        d.attrs['num_units'] = len(uname_all)


        

spikerate_grand_flattened = np.moveaxis(spikerate_grand,0,-1)
spikerate_grand_flattened = spikerate_grand_flattened.reshape(spikerate_grand_flattened.shape[0],-1)
resp_median_grand = np.nanmedian(spikerate_grand_flattened,axis=-1)



with h5py.File(fname_save, 'a') as f:
    if 'spikerate_grand' in f:
        del f['spikerate_grand']  # Delete existing dataset if it exists
    if 'resp_median_grand' in f:
        del f['resp_median_grand']  # Delete existing dataset if it exists
    f.create_dataset('spikerate_grand', data=spikerate_grand_flattened, compression='gzip')
    f.create_dataset('resp_median_grand', data=resp_median_grand, compression='gzip')



# %%

f = h5py.File(fname_save,'r')
a = f['mesopic']['train']['stim_frames']
print(a.shape)
f.close()

# %% STA
from model.featureMaps import decompose
f = h5py.File(fname_save,'r')
stim_frames_train = np.array(f['mesopic']['train']['stim_frames'])
spikeRate_train = np.array(f['mesopic']['train']['spikeRate'])
spikeRate_train = np.squeeze(np.array(f['mesopic']['train']['spikeCounts']))
f.close()

# %%
idx_data = np.arange(0,10000)
data_lim = 100000
stim = stim_frames_train[:data_lim]
stim = np.reshape(stim,(stim.shape[0],num_checkers_y,num_checkers_x),order='F')       
stim = stim-stim.mean()

idx_cell = 5
spikes = spikeRate_train[:,idx_cell][:data_lim]
spikes = np.where(spikes)[0]


num_spikes = len(spikes)
spikeCount = 0
nFrames =100
sta = np.zeros((nFrames,stim.shape[1],stim.shape[2]))

idx_start = np.where(spikes>nFrames)[0][0]
for i in tqdm(range(idx_start,num_spikes)):
    if i>spikes.shape[0]:
        break
    else:
        last_frame = spikes[i]
        first_frame = last_frame - nFrames
        
        sta = sta + stim[first_frame:last_frame,:,:]
        spikeCount+=1
sta = sta/spikeCount

spatial_feature, temporal_feature = decompose(sta)
# plt.plot(1,2,1)
# plt.imshow(spatial_feature,cmap='winter')
# plt.plot(1,2,2)
# plt.plot(temporal_feature)
# plt.title(f'RGC: {uname_all[idx_cell]}, Cell ID {idx_allunits[idx_cell]}')
# plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot spatial feature on the left subplot
ax1.imshow(spatial_feature, cmap='gray')
ax1.set_title('Spatial Feature')

# Plot temporal feature on the right subplot
ax2.plot(temporal_feature)
ax2.set_title('Temporal Feature')

# Add overall title for the figure
fig.suptitle(f'RGC: {uname_all[idx_cell]}, Cell ID {idx_allunits[idx_cell]}')

plt.tight_layout()
plt.show()
# %% RWA
from pyret.filtertools import sta, decompose
from model.data_handler import rolling_window

idx_cell = 0
t_start = 0
t_end = 50000
temporal_window = 30
# dataset = 'train'
stim = stim_frames_train[0][t_start:t_end]

stim = np.reshape(stim,(stim.shape[0],num_checkers_y,num_checkers_x),order='F')       
flips = stim_frames_train[1][t_start:t_end]
spikeRate = spikeRate_train[0][t_start:t_end,idx_cell,0]

stim = rolling_window(stim,temporal_window)
spikeRate = spikeRate[temporal_window:]
rwa = np.nanmean(stim*spikeRate[:,None,None,None],axis=0)



spatial_feature, temporal_feature = decompose(rwa)
# plt.imshow(spatial_feature,cmap='winter')
plt.plot(temporal_feature)


# %%
def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        # arr = arr[:,np.newaxis,:,:]
        return array
    
    
    
    
    
    
    
#%% load different dataset to compare spike rate  Visualize Saad 
fname_save = 'C:\\Users\\Nilou Ghazavi\\Desktop\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_57cell_Nilou\\data_mike_noise\\20230725C\\datasets\\20230725C_dataset_CB_CORR_allLightLevels_8ms_Rstar.h5'

    
def load_data(fname_save, light_level='mesopic'):
    # Original mesopic cell definitions
    mesopic_cell_dove = {
        'OffP': [5,49,52,130,137, 138, 258, 316, 360, 444, 498, 521, 575, 594, 632, 648, 688, 713, 721, 741, 785, 998],
        'OnP': [12, 20,140, 172, 196, 207, 223, 313, 342, 445, 446, 507, 533, 571, 630, 651, 752, 764, 788,969, 980]
    }
    
    # Create unit names as before
    uname_all = []
    for i in range(len(mesopic_cell_dove['OnP'])):
        uname_all.append(f'on_par_{i+1:03d}')
    for i in range(len(mesopic_cell_dove['OffP'])):
        uname_all.append(f'off_par_{i+1:03d}')
    
    with h5py.File(fname_save, 'r') as f:
        train_group = f[f'{light_level}/train']
        
        # Get all the data
        spikes = train_group['spikeCounts'][:].transpose(1, 0, 2)  # (units, timepoints, 1)
        spike_rate = train_group['spikeRate'][:].transpose(1, 0)   # (units, timepoints)
        
        # The data is already sorted by the parasol cells (first 43 cells)
        spikes = spikes[:43]  # Take only the parasol cells
        spike_rate = spike_rate[:43]  # Take only the parasol cells
        
        t_frame = train_group['stim_frames'].attrs['t_frame']
        sig = train_group['spikeRate'].attrs['sig']
        
    return spikes, spike_rate, t_frame, sig, uname_all

# Test the loading
spikes, spike_rate, t_frame, sig, uname_all = load_data(fname_save, light_level='mesopic')

print("Loaded data shapes:")
print(f"Spikes shape: {spikes.shape}")
print(f"Spike rate shape: {spike_rate.shape}")
print(f"Number of cells: {len(uname_all)}")





def load_data(fname_save, light_level='mesopic'):
    # Define cluster IDs
#57 cells
    # mesopic_cell_dove = {
    #     'OffP': [3,5,49,52,59,59,59,63, 130,137, 138,198, 258, 316, 360, 444, 498, 521,527, 575, 594, 632, 648, 688, 713, 721, 741, 785, 785,998],
    #     'OnP': [12, 20,140, 172, 196, 196, 207, 223, 290, 313, 342, 445, 446, 507, 533,539,  571, 571, 630, 651,662, 706, 752, 764, 788,969, 980]
    # }
    # Original mesopic cell definitions
    mesopic_cell_dove = {
        'OffP': [5,49,52,130,137, 138, 258, 316, 360, 444, 498, 521, 575, 594, 632, 648, 688, 713, 721, 741, 785, 998],
        'OnP': [12, 20,140, 172, 196, 207, 223, 313, 342, 445, 446, 507, 533, 571, 630, 651, 752, 764, 788,969, 980]
    }
    
    
    # Create mapping of cluster IDs to array indices
    all_clusters = mesopic_cell_dove['OnP'] + mesopic_cell_dove['OffP']
    cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(all_clusters)}
    
    with h5py.File(fname_save, 'r') as f:
        train_group = f[f'{light_level}/train']
        
        spikes = train_group['spikeCounts'][:].transpose(1, 0, 2)
        spike_rate = train_group['spikeRate'][:].transpose(1, 0)
        
        t_frame = train_group['stim_frames'].attrs['t_frame']
        sig = train_group['spikeRate'].attrs['sig']
    
    return spikes, spike_rate, t_frame, sig, mesopic_cell_dove, cluster_to_index

# Load data
spikes, spike_rate, t_frame, sig, mesopic_cell_dove, cluster_to_index = load_data(fname_save)

# Plot for specific cluster IDs
def plot_cluster(cluster_id, array_idx, spikeCounts, spikeRate, sig, t_frame, is_on=True):
    cell_type = "ON" if is_on else "OFF"
    
    fig, axes = visualize_cell_activity(
        spikeCounts=spikeCounts,
        sig=sig,
        t_frame=t_frame
    )
    plt.suptitle(f'Cell Response Analysis - Cluster ID: {cluster_id} (Array Index: {array_idx}, {cell_type} parasol)', y=1.02)
    plt.show()

# Plot for ON parasol cells
for cluster_id in mesopic_cell_dove['OnP']:
    array_idx = cluster_to_index[cluster_id]
    print(f"Plotting ON parasol cell {cluster_id} (Array index: {array_idx})")
    plot_cluster(
        cluster_id=cluster_id,
        array_idx=array_idx,
        spikeCounts=spikes[array_idx],
        spikeRate=spike_rate[array_idx],
        sig=sig,
        t_frame=t_frame,
        is_on=True
    )

# Plot for OFF parasol cells
for cluster_id in mesopic_cell_dove['OffP']:
    array_idx = cluster_to_index[cluster_id]
    print(f"Plotting OFF parasol cell {cluster_id} (Array index: {array_idx})")
    plot_cluster(
        cluster_id=cluster_id,
        array_idx=array_idx,
        spikeCounts=spikes[array_idx],
        spikeRate=spike_rate[array_idx],
        sig=sig,
        t_frame=t_frame,
        is_on=False
    )