
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
import torch
import tensorly as tl

# %load_ext autoreload
# %autoreload 2
import bnpm

tl.set_backend('pytorch')

## Import face_rhythm TCA factors and spectrogram tensor

params = bnpm.file_helpers.json_load(path_params)

# dir_save = r'/home/rich/Desktop/'

# directory_FR_template = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/20230430/run_from_o2'
# directory_FR_current = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/20230502//jobNum_0'

directory_FR_template = params['directory_FR_template']
directory_FR_current = params['directory_FR_current']

tca_template = bnpm.h5_handling.simple_load(str(Path(directory_FR_template) / 'analysis_files' / 'TCA.h5'))

tca_current = bnpm.h5_handling.simple_load(str(Path(directory_FR_current) / 'analysis_files' / 'TCA.h5'))

params_template = bnpm.file_helpers.json_load(str(Path(directory_FR_template) / 'params.json'))

DEVICE_data = bnpm.torch_helpers.set_device(use_GPU=False)

def cp_dict_to_cp_tensor(cp_dict, device='cpu'):
    """A function for converting a raw list of factor matrices into tensorly's CPTensor format"""
    return tl.cp_tensor.CPTensor((None, [torch.as_tensor(v, dtype=torch.float32, device=device) for v in cp_dict.values()]))

def make_cp_init(k_tensor, shape_dense_tensor, modes_fixed=[0,1,], device='cpu'):
    """Makes a CPTensor for initializing a TCA run. The k_tensor matrices will be used for each of the fixed modes and will be shuffle permuted for each of the non-fixed modes."""
    import copy
    n_modes = len(k_tensor)
    kt = [None]*n_modes
    for i_mode in range(len(kt)):
        if i_mode in modes_fixed:
            kt[i_mode] = torch.as_tensor(k_tensor[i_mode], dtype=torch.float32, device=device)
        else:
            perm = torch.randperm(k_tensor[i_mode].shape[0], device=device)
            kt[i_mode] = torch.as_tensor(k_tensor[i_mode], dtype=torch.float32, device=device)[perm]
        
    return tl.cp_tensor.CPTensor((None, kt))

spec_current = bnpm.h5_handling.simple_load(str(Path(directory_FR_current) / 'analysis_files' / 'VQT_Analyzer.h5'))

## Prepare the current session spectrogram for refitting
### flatten the (xy points) dimension
s = spec_current['spectrograms']['0'].copy()
s = s.transpose(2,3,0,1)
s = s.reshape(s.shape[0], s.shape[1], -1)
s = s.transpose(2,0,1)
s = torch.as_tensor(s, dtype=torch.float32, device=DEVICE_data)

## prepare tca factors into a tensorly CPTensor
cp_template = cp_dict_to_cp_tensor(tca_template['factors_rearranged']['0'], device=DEVICE_data)
cp_current = cp_dict_to_cp_tensor(tca_current['factors_rearranged']['0'], device=DEVICE_data)


DEVICE_tca = bnpm.torch_helpers.set_device(use_GPU=False)

modes_fixed = [0,1,]

cp_init = make_cp_init(cp_template.factors, s.shape, modes_fixed=modes_fixed, device=DEVICE_tca)

params_tca = copy.deepcopy(params_template['TCA']['fit']['params_method'])

params_tca['n_iter_max'] = 100
params_tca['init'] = cp_init

model_tca = tl.decomposition.CP_NN_HALS(
    **params_tca,
    fixed_modes=modes_fixed,
)

model_tca.fit(s.to(DEVICE_tca))

cp_refit = model_tca.decomposition_
cp_refit = tl.cp_tensor.CPTensor((cp_refit.weights.cpu(), [f.cpu() for f in cp_refit.factors]))

EV_rec_refit = bnpm.similarity.cp_reconstruction_EV(
    tensor_dense=s,
    tensor_CP=cp_refit.factors,
)

EV_rec_original = bnpm.similarity.cp_reconstruction_EV(
    tensor_dense=s,
    tensor_CP=cp_current.factors,
)

print(f'EV_rec_original: {EV_rec_original}')
print(f'EV_rec_refit: {EV_rec_refit}')

tca_refit = {
    'factors_refit': {key: val.cpu().numpy() for key, val in zip(tca_template['factors_rearranged']['0'].keys(), cp_refit.factors)},
    'factors_original': {str(ii): f.cpu().numpy() for ii,f in enumerate(cp_template.factors)},
    'cp_init': {str(ii): f.cpu().numpy() for ii,f in enumerate(cp_init.factors)},
    'modes_fixed': modes_fixed,
    'EV_rec_original': EV_rec_original,
    'EV_rec_refit': EV_rec_refit,
    'directory_template': directory_FR_template,
    'directory_current':directory_FR_current,
}

bnpm.h5_handling.simple_save(
    dict_to_save=tca_refit,
    path=str(Path(directory_save) / 'tca_refit.h5'),
    write_mode='w-',
    verbose=True,
)