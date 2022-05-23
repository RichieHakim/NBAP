# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))

## Import general libraries
from pathlib import Path
import os
import sys
import copy

import numpy as np
import itertools
import glob

### Import personal libraries
# dir_github = '/media/rich/Home_Linux_partition/github_repos'
dir_github = '/n/data1/hms/neurobio/sabatini/rich/github_repos'

import sys
sys.path.append(dir_github)
# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import container_helpers, server
# from s2p_on_o2 import remote_run_s2p


args = sys.argv
path_selfScript = args[0]
dir_save = args[1]
path_script = args[2]
name_job = args[3]
name_slurm = args[4]
dir_data = args[5]

print(path_selfScript, dir_save, path_script, name_job, dir_data)

## set paths
# dir_save = '/n/data1/hms/neurobio/sabatini/rich/analysis/suite2p_output/'
Path(dir_save).mkdir(parents=True, exist_ok=True)


# path_script = '/n/data1/hms/neurobio/sabatini/rich/github_repos/s2p_on_o2/remote_run_s2p.py'


### Define directories for data and output.
## length of both lists should be the same
# dirs_data_all = ['/n/data1/hms/neurobio/sabatini/rich/analysis/suite2p_output']
# dirs_save_all = [str(Path(dir_save) / 'test_s2p_on_o2')]



params_template = {
    'paths': {
        'dir_sessionData': dir_data,
        'dir_sub_s2p': '/suite2p_o2_output/jobNum_0/suite2p/plane0',
        'fileName_re_motionCorrection': '*Motion*.csv',
        'fileName_save_denoisingVariables': 'denoising_variables',
        'dir_github': '/n/data1/hms/neurobio/sabatini/rich/github_repos',
    },
    'make_dFoF': {
        'channelOffset_correction': 200,
        'percentile_baseline': 20,
        'neuropil_fraction': 0.7,
    },
    'ransac': {
        'z_offset_smoothing_windows': [11, 41, 121, 625],
        'clip_scaling': 1.0,
        'clip_offset': 0.1,
        'residual_threshold_scaling': 0.1,
        'min_samples': 0.1,
        'noise_calculation_clip_range': [-1, 1],
        'thresh_EVR_ransac': 0.1
    },
    'tqm': {
        'thresh_dict': {
            'var_ratio': 0.5,
            'EV_F_by_Fneu': 0.5,
            'base_FneuSub': 25,
            'base_F': 0,
            'peter_noise_levels': 5,
            'rich_nsr': 3,
            'max_dFoF': 50,
            'baseline_var': 0.1,
        },
        'clip_range': (-1, 1),
    }
}

## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [params]
# params = [container_helpers.deep_update_dict(params, ['db', 'save_path0'], str(Path(val).resolve() / (name_save+str(ii)))) for val in dir_save]
# params = [container_helpers.deep_update_dict(param, ['db', 'save_path0'], val) for param, val in zip(params, dirs_save_all)]
# params = container_helpers.flatten_list([[container_helpers.deep_update_dict(p, ['lr'], val) for val in [0.00001, 0.0001, 0.001]] for p in params])

# params_unchanging, params_changing = container_helpers.find_differences_across_dictionaries(params)


## notes that will be saved as a text file in the outer directory
notes = \
"""
First attempt
"""
with open(str(Path(dir_save) / 'notes.txt'), mode='a') as f:
    f.write(notes)



## copy script .py file to dir_save
import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));



# ## save parameters to file
# parameters_batch = {
#     'params': params,
#     # 'params_unchanging': params_unchanging,
#     # 'params_changing': params_changing
# }
# import json
# with open(str(Path(dir_save) / 'parameters_batch.json'), 'w') as f:
#     json.dump(parameters_batch, f)

# with open(str(Path(dir_save) / 'parameters_batch.json')) as f:
#     test = json.load(f)


## run batch_run function
paths_scripts = [path_script]
params_list = params
# sbatch_config_list = [sbatch_config]
max_n_jobs=1
name_save=name_job


## define print log paths
paths_log = [str(Path(dir_save) / f'{name_save}{jobNum}' / 'print_log_%j.log') for jobNum in range(len(params))]

## define slurm SBATCH parameters
sbatch_config_list = \
[f"""#!/usr/bin/bash
#SBATCH --job-name={name_slurm}
#SBATCH --output={path}
#SBATCH --partition=short
#SBATCH -c 20
#SBATCH -n 1
#SBATCH --mem=36GB
#SBATCH --time=0-01:00:00

unset XDG_RUNTIME_DIR

cd /n/data1/hms/neurobio/sabatini/rich/

date

echo "loading modules"
module load gcc/9.2.0

echo "activating environment"
source activate NBAP

echo "starting job"
python "$@"
""" for path in paths_log]

server.batch_run(
    paths_scripts=paths_scripts,
    params_list=params_list,
    sbatch_config_list=sbatch_config_list,
    max_n_jobs=max_n_jobs,
    dir_save=str(dir_save),
    name_save=name_save,
    verbose=True,
)
