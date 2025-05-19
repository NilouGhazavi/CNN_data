#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4


@author: saad
"""


# Save training, testing and validation datasets to be read by jobs on cluster
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt


os.chdir(r"D:\Nilou\GitHub\Gradients\RetinaPredictors-main")
import os
import re
import h5py
import numpy as np
from model.data_handler import check_trainVal_contamination
from model.data_handler_mike import load_data_allLightLevels_cb, load_data_allLightLevels_natstim, save_h5Dataset
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])


lightLevel = 'allLightLevels'     # ['scotopic', 'photopic','scotopic_photopic']
#datasetsToLoad = ['mesopic',]#,'photopic'];    #['scotopic','photopic','scotopic_photopic']
# datasetsToLoad= ['scotopic','mesopic','photopic']
datasetsToLoad=['mesopic']
N_split = 0


natstim_idx_val = 0
REP_TRAINING_DATA = 0
STIM = 'CB_CORR' # 'CB'  'NATSTIM' # 'CB'  'NATSTIM'
STIM_NAT ='NATSTIM'+str(natstim_idx_val)+'_CORR'
file_suffix = 'Rstar'
NORM_STIM = 0
NORM_RESP = True
D_TYPE = 'f4'


expDate = '20230725C'     
path_dataset = os.path.join('D:\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_43cells_Nilou\\data_mike_noise\\',expDate,'datasets\\')
path_save = os.path.join('D:\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_43cells_Nilou\\data_mike_noise\\',expDate,'datasets\\')
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'gradient_analysis/datasets/')
    
t_frame = 8


fname_dataFile = os.path.join(path_dataset,(expDate+'_dataset_'+STIM+'_allLightLevels'+'_'+str(t_frame)+'ms_'+file_suffix+'.h5'))




filt_temporal_width = 0
idx_cells = None
thresh_rr = 0


frac_val = 0.05
frac_test = 0.01 


    
def stim_vecToMat(data,num_checkers_y,num_checkers_x):
    X = data.X
    X = np.reshape(X,(X.shape[0],num_checkers_y,num_checkers_x),order='F')    # convert stim frames back into spatial dimensions
    data = Exptdata_spikes(X,data.y,data.spikes)
    return data




# % Calculate grand response median
totalNum_units = 43
spikerate_grand = np.empty((totalNum_units,0))
STIMS_ALL = ('NATSTIM'+str(natstim_idx_val)+'_CORR','CB_CORR')
for i in range(len(STIMS_ALL)):
    fname = os.path.join(path_dataset,(expDate+'_dataset_'+STIM+'_allLightLevels'+'_'+str(t_frame)+'ms_'+file_suffix+'.h5'))
    with h5py.File(fname,'r') as f:
        spikerate_grand = np.concatenate((spikerate_grand,np.array(f['spikerate_grand'])),axis=-1)
        
spikerate_grand[spikerate_grand==0]=np.nan
resp_med_grand = np.nanmedian(spikerate_grand,axis=-1)


# %%


dataset = datasetsToLoad[0]



for dataset in datasetsToLoad:
    fname_noise = os.path.join(path_save,(expDate+'_dataset_train_val_test_'+STIM_NAT+'_'+dataset+'-'+file_suffix+'_'+D_TYPE+'_'+str(t_frame)+'ms'+'.h5'))


    
    if STIM[:7] == 'NATSTIM':
        data_train,data_val,data_test,data_quality,dataset_rr,resp_orig,_ = load_data_allLightLevels_natstim(fname_dataFile,dataset,frac_val=frac_val,frac_test=frac_test,
                                                                                                   filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,
                                                                                                   resp_med_grand=resp_med_grand,thresh_rr=thresh_rr,N_split=N_split,
                                                                                                   CHECK_CONTAM=False,NORM_RESP=NORM_RESP)


        if REP_TRAINING_DATA>0:
            print('RESAMPLING TRAINING SAMPLES')
            X = np.tile(data_train.X,[REP_TRAINING_DATA,1,1,1,1])
            y = np.tile(data_train.y,[REP_TRAINING_DATA,1,1,1])
            spikes = np.tile(data_train.spikes,[REP_TRAINING_DATA,1,1,1])
            
            data_train = Exptdata_spikes(X,y,spikes)
            
        


        # if data_quality['var_noise']==None:
        #     with h5py.File(fname_noise) as f:
        #         obs_noise = np.array(f['data_quality']['var_noise'])


        #     data_quality['var_noise'] =  obs_noise


            
    elif 'CB' in STIM:
        data_train,data_val,data_test,data_quality,dataset_rr,resp_orig = load_data_allLightLevels_cb(fname_dataFile,dataset,frac_val=frac_val,frac_test=frac_test,
                                                                                                   filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,
                                                                                                   resp_med_grand=resp_med_grand,thresh_rr=thresh_rr,N_split=N_split,
                                                                                                   CHECK_CONTAM = False,NORM_RESP=NORM_RESP)

        
        import os
        if os.path.exists(fname_noise):
            with h5py.File(fname_noise) as f:
                obs_noise = np.array(f['data_quality']['var_noise'])
            data_quality['var_noise'] = obs_noise
        else:
            print(f"Warning: {fname_noise} does not exist. Skipping noise data reading.")
            data_quality['var_noise'] = None  # or some default value


    if REP_TRAINING_DATA>0:
        fname_data_train_val_test = os.path.join(path_save,(expDate+'_dataset_train_val_test_'+STIM+'_REP-'+str(REP_TRAINING_DATA)+'_'+dataset+'-'+file_suffix+'_'+D_TYPE+'_'+str(t_frame)+'ms'))
    else:
        fname_data_train_val_test = os.path.join(path_save,(expDate+'_dataset_train_val_test_'+STIM+'_'+dataset+'-'+file_suffix+'_'+D_TYPE+'_'+str(t_frame)+'ms'))
    
    f = h5py.File(fname_dataFile,'r')
    samps_shift = 0#np.array(f[dataset]['val']['spikeRate'].attrs['samps_shift'])
    if 'num_checkers_x' in f[dataset]['train']['stim_frames'].attrs.keys():
        num_checkers_x = np.array(f[dataset]['train']['stim_frames'].attrs['num_checkers_x'])
        num_checkers_y = np.array(f[dataset]['train']['stim_frames'].attrs['num_checkers_y'])
        checkSize_um = np.array(f[dataset]['train']['stim_frames'].attrs['checkSize_um'])
    else:
        num_checkers_x = np.array(f[dataset]['train']['stim_frames'].shape[2])
        num_checkers_y = np.array(f[dataset]['train']['stim_frames'].shape[1])
        checkSize_um = 3.8  # 3.8 um/pixel


    t_frame_inData = np.array(f[dataset]['train']['stim_frames'].attrs['t_frame'])
    parameters = {
    't_frame': t_frame_inData,
    'filt_temporal_width': filt_temporal_width,
    'frac_val': frac_val,
    'frac_test':frac_test,
    'thresh_rr': thresh_rr,
    'samps_shift': samps_shift,
    'num_checkers_x': num_checkers_x,
    'num_checkers_y': num_checkers_y,
    'checkSize_um': checkSize_um
    }
    f.close()
    
    if data_train.X.ndim == 2:
       data_train = stim_vecToMat(data_train,parameters['num_checkers_y'],parameters['num_checkers_x'])
       data_val = stim_vecToMat(data_val,parameters['num_checkers_y'],parameters['num_checkers_x'])
       data_test = stim_vecToMat(data_test,parameters['num_checkers_y'],parameters['num_checkers_x'])
    
    # fname_data_train_val_test = fname_data_train_val_test + '_StimNorm-'+str(NORM_STIM) #+ '_RespNorm-'+str(NORM_RESP)


    save_h5Dataset(fname_data_train_val_test+'.h5',data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig,dtype=D_TYPE)




import h5py
file_path_nilou='C:\\Users\\Nilou Ghazavi\\Desktop\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\analyses_57cells\\data_mike_noise\\20230725C\\datasets\\20230725C_dataset_train_val_test_CB_CORR_mesopic-Rstar_f4_8ms.h5'
file_path_saad = "C:\\Users\\Nilou Ghazavi\\Desktop\\Nilou\\GitHub\\Gradients\\RGC_Selective_Simulation\\White_Noise\\data_Saad\\20230725C_dataset_train_val_test_CB_CORR_scotopic-Rstar_f4_8ms.h5"

with h5py.File(file_path_nilou, 'r') as file:
    # Print all keys (top level datasets and groups)
    keys = list(file.keys())
    print("Keys: %s" % keys)
    
    # Access 'dataset_rr' if it's in the file
    if 'data_quality' in file:
        item = file['data_quality']
        # Check if it is a group or dataset
        if isinstance(item, h5py.Group):
            print("Item 'dataset_quality' is a group.")
            subkeys = list(item.keys())
            print("It contains the following subkeys: %s" % subkeys)
            # Optionally, inspect subitems if needed
            for subkey in subkeys:
                print(f"Details of {subkey}:")
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    print(f"Shape of dataset {subkey}: {subitem.shape}")
                    print(f"Data of {subkey}: {subitem[:]}")  # Be cautious with large datasets
        elif isinstance(item, h5py.Dataset):
            print("Shape of dataset 'dataset_quality':", item.shape)
            print("Data of 'dataset_quality':", item[:])  # Again, be cautious with large datasets
        else:
            print("Unknown type for 'dataset_quality'.")
    else:
        print("Item 'dataset_quality' not found in the file.")
        
        
        
        


        