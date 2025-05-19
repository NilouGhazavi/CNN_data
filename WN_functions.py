#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 22:13:03 2025

@author: niloughazavi
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
os.chdir("/Users/niloughazavi/Documents/GitHub/RetinaPredictors-main")

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
