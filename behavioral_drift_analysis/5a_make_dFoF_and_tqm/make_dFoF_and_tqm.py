
########################################
## Import parameters from CLI
########################################

import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")

## Import path_params and directory_save from CLI
import sys
path_params, directory_save = sys.argv[1], sys.argv[2]


import copy
from pathlib import Path

import numpy as np

# %load_ext autoreload
# %autoreload 2
import bnpm

params = bnpm.file_helpers.json_load(path_params)
import bnpm

# dir_s2p = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/s2p/20230430/plane0/'
dir_s2p = params['dir_s2p']

F, Fneu, iscell, ops, spks, stat = bnpm.ca2p_preprocessing.import_s2p(dir_s2p=dir_s2p)

Fs = ops['fs']

# dFoF_params = {
#     "channelOffset_correction": 0,
#     "percentile_baseline": 30,
#     "neuropil_fraction": 0.7,
#     "rolling_percentile_window": Fs*15*60,
# }
dFoF_params = params['dFoF_params']

dFoF , dF , F_neuSub , F_baseline_roll = bnpm.ca2p_preprocessing.make_dFoF(
    F,
    Fneu=Fneu,
    roll_centered=True,
    roll_stride=1,
    roll_interpolation='linear',
    multicore_pref=True,
    verbose=True,
    **dFoF_params,
)

# %matplotlib inline

# thresh = {
#     'var_ratio__Fneu_over_F': (0, 0.5),
#     'EV__F_by_Fneu': (0, 0.5),
#     'base_FneuSub': (50, 2000),
#     'base_F': (100, 5000),
#     'nsr_autoregressive': (0, 10),
#     'noise_derivMAD': (0, 0.025),
#     'max_dFoF': (0.75, 40),
#     'baseline_var': (0, 0.025),
# }
# # thresh = {
# #     'var_ratio__Fneu_over_F': np.inf,
# #     'EV__F_by_Fneu': np.inf,
# #     'base_FneuSub': -np.inf,
# #     'base_F': -np.inf,
# #     'nsr_autoregressive': np.inf,
# #     'noise_derivMAD': np.inf,
# #     'max_dFoF': np.inf,
# #     'baseline_var': np.inf,
# # }
thresh = params['thresh']
    
tqm, iscell_tqm = bnpm.ca2p_preprocessing.trace_quality_metrics(
    F=F,
    Fneu=Fneu,
    dFoF=dFoF,
    F_neuSub=F_neuSub,
    F_baseline_roll=F_baseline_roll,
    percentile_baseline=dFoF_params['percentile_baseline'],
    Fs=Fs,
    # plot_pref=True,
    thresh=thresh,
    device='cpu',
)


## Save
### Save the dFoF and tqm
np.save(str(Path(directory_save) / 'dFoF.npy'), dFoF)
bnpm.file_helpers.pickle_save(obj=tqm, filepath=str(Path(directory_save) / 'tqm.pkl'))
np.save(str(Path(directory_save) / 'iscell_tqm.npy'), iscell_tqm)
