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


import sys
# sys.path.append('/n/data1/hms/neurobio/sabatini/rich/github_repos/')
sys.path.append(params['dir_github'])
from basic_neural_processing_modules import ca2p_preprocessing, timeSeries, math_functions, indexing, file_helpers


import cascade2p
from cascade2p import checks
checks.check_packages()
from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth




denoisingVariables = file_helpers.pickle_load(params['path_denoisingVariables_data'])

dFoF = denoisingVariables['dFoF_denoised']
goodROIs_denoising = denoisingVariables['goodROIs_denoising']


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




cascade.download_model('update_models', model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'),verbose = 1)

cascade.download_model(params['model'], model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'), verbose = 1)

yaml_file = open(str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models' / 'available_models.yaml'))
X = yaml.load(yaml_file, Loader=yaml.Loader)
list_of_models = list(X.keys())
print('\n List of available models: \n')
for model in list_of_models:
    print(model)

    
    
print(f'STARTING CASCADE, time: {time.ctime()}')

spike_prob = np.concatenate([cascade.predict(
    model_name=params['model'],
    traces=batch, 
    model_folder=str(Path(params['dir_github']) / 'Cascade' / 'Pretrained_models'), 
    padding=0,
    verbosity=params['verbosity_cascadePredict']
) for batch in tqdm(indexing.make_batches(
    dFoF_smooth, 
    batch_size=params['batchSize_nROIs']
))], axis=0)

print(f'COMPLETED CASCADE, time: {time.ctime()}')

print(f'SAVING, time: {time.ctime()}')

np.save(str(Path(dir_save) / 'spike_prob.npy'), spike_prob)

print(f'COMPLETED ALL, time: {time.ctime()}')