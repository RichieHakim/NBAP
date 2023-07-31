## Import general libraries
from pathlib import Path
import os
import sys
import copy

print(f"dispatcher environment: {os.environ['CONDA_DEFAULT_ENV']}")

from bnpm.server import batch_run

path_self, path_script, dir_save, dir_s2p, name_job, name_slurm, name_env = sys.argv


# date = '20221011'

# path_script = f'/n/data1/hms/neurobio/sabatini/rich/github_repos/face-rhythm/scripts/pipeline_basic.py'
# dir_save = f'/n/data1/hms/neurobio/sabatini/rich/analysis/faceRhythm/{mouse}/run_20230701/'
# name_job = f'faceRhythm_{date}_'
# name_slurm = f'rh_{date}'
# name_env = f'/n/data1/hms/neurobio/sabatini/rich/virtual_envs/FR'

## set paths
Path(dir_save).mkdir(parents=True, exist_ok=True)


params_template = {
    'dir_s2p': str(Path(dir_s2p).resolve()),
    'dFoF_params': {
        "channelOffset_correction": 0,
        "percentile_baseline": 30,
        "neuropil_fraction": 0.7,
        "rolling_percentile_window": Fs*15*60,
    },
    'thresh': {
        'var_ratio__Fneu_over_F': (0, 0.5),
        'EV__F_by_Fneu': (0, 0.5),
        'base_FneuSub': (50, 2000),
        'base_F': (100, 5000),
        'nsr_autoregressive': (0, 10),
        'noise_derivMAD': (0, 0.025),
        'max_dFoF': (0.75, 40),
        'baseline_var': (0, 0.025),
    },
}


## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [params]
# params = [container_helpers.deep_update_dict(params_template, ['db', 'save_path0'], str(Path(val).resolve() / (name_save+str(ii)))) for val in dir_save]
# params = [helpers.deep_update_dict(param, ['db', 'save_path0'], val) for param, val in zip(params_template, dirs_save_all)]
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
Path(dir_save).mkdir(parents=True, exist_ok=True)
print(f'Copying {path_script} to {str(Path(dir_save) / Path(path_script).name)}')
shutil.copyfile(path_script, str(Path(dir_save) / Path(path_script).name))



## save parameters to file
parameters_batch = {
    'params': params,
    # 'params_unchanging': params_unchanging,
    # 'params_changing': params_changing
}
import json
with open(str(Path(dir_save) / 'parameters_batch.json'), 'w') as f:
    json.dump(parameters_batch, f)


## run batch_run function
paths_scripts = [path_script]
params_list = params
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
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=32GB
#SBATCH --time=0-00:5:00

unset XDG_RUNTIME_DIR

cd /n/data1/hms/neurobio/sabatini/rich/

date

echo "loading modules"
module load gcc/9.2.0

echo "activating environment"
source activate {name_env}

echo "starting job"
python "$@"
""" for path in paths_log]

# SBATCH --gres=gpu:1,vram:23G
# SBATCH --partition=gpu_requeue


batch_run(
    paths_scripts=paths_scripts,
    params_list=params_list,
    sbatch_config_list=sbatch_config_list,
    max_n_jobs=max_n_jobs,
    dir_save=str(dir_save),
    name_save=name_save,
    verbose=True,
)
