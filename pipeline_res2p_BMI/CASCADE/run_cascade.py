"""
This script performs spike inference on input dF/F traces to give output spks traces of the same shape.
Uses CASCADE method from Helchen lab: https://github.com/HelmchenLabSoftware/Cascade
"""


### batch_run stuff
from pathlib import Path

import sys
path_script, path_params, dir_save = sys.argv
dir_save = Path(dir_save)
                
import json
with open(path_params, 'r') as f:
    params = json.load(f)

import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

# params = {
#     'smoothing_win': 200,
#     'Fs': 5.14,
#     'model': 'Global_EXC_5Hz_smoothing200ms',
#     'batchSize_nROIs': 10,
#     'dir_github': '/n/data1/hms/neurobio/sabatini/rich/github_repos',
#     'path_denoisingVariables_data': dir_data,
#     'verbosity_cascadePredict': 0,
# }




import numpy as np

from tqdm import tqdm

import copy
import time
import gc

from pathlib import Path
import yaml

# check environment
import os
print(f'Conda Environment: ' + os.environ['CONDA_DEFAULT_ENV'])

from platform import python_version
print(f'python version: {python_version()}')


import sys
# sys.path.append('/n/data1/hms/neurobio/sabatini/rich/github_repos/')
# sys.path.append(params['dir_github'])
from bnpm import ca2p_preprocessing, timeSeries, math_functions, indexing, file_helpers


import cascade2p
from cascade2p import checks
checks.check_packages()
from cascade2p import cascade # local folder
# from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth

import tensorflow as tf
## check if GPU is available
print(f'GPU available: {tf.test.is_gpu_available()}')
print(f'GPU name: {tf.test.gpu_device_name()}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')
print(f'Built with gpu support: {tf.test.is_built_with_gpu_support()}')



## import neural data

print(f'Starting: Importing data from {params["make_dFoF"]["dir_s2p"]}')
F , Fneu , iscell , ops , spks , stat , num_frames_S2p = ca2p_preprocessing.import_s2p(Path(params['make_dFoF']['dir_s2p']))
print(f'Completed: Importing data')


channelOffset_correction = params['make_dFoF']['channelOffset_correction']
percentile_baseline = params['make_dFoF']['percentile_baseline']

print(f'Starting: Calculating dFoF from F and Fneu, using channelOffset_correction={channelOffset_correction} and percentile_baseline={percentile_baseline}')
dFoF , dF , F_neuSub , F_baseline = ca2p_preprocessing.make_dFoF(
    F=F + channelOffset_correction,
    Fneu=Fneu + channelOffset_correction,
    neuropil_fraction=params['make_dFoF']['neuropil_fraction'],
    percentile_baseline=percentile_baseline,
    multicore_pref=True,
    verbose=True
)
print(f'Completed: Calculating dFoF')


## smooth dFoF

print(f'Starting: Smoothing dFoF with gaussian kernel of width {params["smoothing_win"]} ms')
dFoF_smooth = timeSeries.convolve_along_axis(
    dFoF,
    kernel=math_functions.gaussian(
        x=np.arange(-15,15), 
        mu=0, 
        sig=(params['smoothing_win']/1000)*params['Fs'], 
        plot_pref=False
    ),
    axis=1,
    mode='same',
    multicore_pref=True,
    verbose=True
)
print(f'Completed: Smoothing dFoF')



## run cascade

print(f'Loading model: {params["model"]}')
cascade.download_model('update_models', model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'),verbose = 1)
cascade.download_model(params['model'], model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'), verbose = 1)
print(f'Loaded model')

yaml_file = open(str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models' / 'available_models.yaml'))
X = yaml.load(yaml_file, Loader=yaml.Loader)
list_of_models = list(X.keys())
print('\n List of available models: \n')
for model in list_of_models:
    print(model)

    
    
print(f'STARTING CASCADE, time: {time.ctime()}')
print(f'Number of ROIs: {dFoF_smooth.shape[0]}')

spike_prob = np.concatenate([cascade.predict(
    model_name=params['model'],
    traces=batch, 
    model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'), 
    padding=0,
    verbosity=params['verbosity_cascadePredict']
) for batch in tqdm(indexing.make_batches(
    dFoF_smooth, 
    batch_size=params['batchSize_nROIs']
), total=int(np.ceil(dFoF_smooth.shape[0]/params['batchSize_nROIs'])))], axis=0).astype(np.float32)

print(f'COMPLETED CASCADE, time: {time.ctime()}')

print(f'SAVING, time: {time.ctime()}')

np.save(str(Path(dir_save) / 'spike_prob.npy'), spike_prob)

print(f'COMPLETED ALL, time: {time.ctime()}')