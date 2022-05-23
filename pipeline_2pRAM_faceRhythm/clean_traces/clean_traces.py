
# params = {
#     'paths': {
#         'dir_sessionData': '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG22/2022_05_16/',
#         'dir_sub_s2p': '/suite2p_o2_output/jobNum_0/suite2p/plane0',
#         'fileName_re_motionCorrection': '*Motion*.csv',
#         'fileName_save_denoisingVariables': 'denoising_variables',
#         'dir_github': '/n/data1/hms/neurobio/sabatini/rich/github_repos',
#     },
#     'make_dFoF': {
#         'channelOffset_correction': 200,
#         'percentile_baseline': 20,
#         'neuropil_fraction': 0.7,
#     },
#     'ransac': {
#         'z_offset_smoothing_windows': [11, 41, 121, 625],
#         'clip_scaling': 1.0,
#         'clip_offset': 0.1,
#         'residual_threshold_scaling': 0.1,
#         'min_samples': 0.1,
#         'noise_calculation_clip_range': [-1, 1],
#         'thresh_EVR_ransac': 0.1
#     },
#     'tqm': {
#         'thresh_dict': {
#             'var_ratio': 0.5,
#             'EV_F_by_Fneu': 0.5,
#             'base_FneuSub': 25,
#             'base_F': 0,
#             'peter_noise_levels': 5,
#             'rich_nsr': 3,
#             'max_dFoF': 50,
#             'baseline_var': 0.1,
#         },
#         'clip_range': (-1, 1),
#     }
# }

"""
This script makes and cleans dF/F traces for a single session.
Cleaning is done in 3 steps:
    1. Regress out z-motion
        a. Define z-offsets using imported Motion.csv file from scanimage
        b. Use RANSAC to regress out z-artifacts
    2. Exclude traces with high correlation to motion correction
    3. Exclude traces with poor trace quality metrics
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



### notebook prep

## widen notebook

# from IPython.display import display, HTML
# display(HTML("<style>.container {width:95% !important; }</style>"))



## imports

## standard libraries

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import scipy.io
import scipy.interpolate

import rastermap

from tqdm import tqdm

import copy
import time
import gc
from pathlib import Path


## my library

import sys
dir_github = params['paths']['dir_github']
sys.path.append(dir_github)

from basic_neural_processing_modules import ca2p_preprocessing, path_helpers, similarity, pickle_helpers, misc, featurization



## Set Paths

dir_sessionData = str(Path(params['paths']['dir_sessionData']).resolve())  ## directory containing s2p and motion correction data
dir_sub_s2p = str(Path(params['paths']['dir_sub_s2p']).resolve())  ## String. relative sub path from dir_sessionData to the s2p output (F.npy, stat.npy, etc.)
fileName_re_motionCorrection = params['paths']['fileName_re_motionCorrection']  ## File name of the motion correction csv. Uses regular expression pattern format for searching.
fileName_save_denoisingVariables = params['paths']['fileName_save_denoisingVariables']  ## File name of the output dictionary of variables from this script. Leave off a suffix.

# dir_save = dir_sessionData
dir_s2p = str(Path(dir_sessionData).joinpath(dir_sub_s2p.lstrip('/').lstrip(r"\\")))

path_motionCorrection = str(sorted(Path(dir_sessionData).glob(fileName_re_motionCorrection))[-1])  ## searches for fileName_re_motionCorrection, sorts findings, and chooses last one (in case there are multiple files)
path_save_denoisingVariables = str((Path(dir_save) / fileName_save_denoisingVariables).with_suffix('.pkl'))

table_motionCorrection = pd.read_csv(path_motionCorrection, sep=', ', engine='python')




## import neural data

F , Fneu , iscell , ops , spks , stat , num_frames_S2p = ca2p_preprocessing.import_s2p(Path(dir_s2p))

F = F[:100]
Fneu = Fneu[:100]

channelOffset_correction = params['make_dFoF']['channelOffset_correction']
percentile_baseline = params['make_dFoF']['percentile_baseline']

dFoF , dF , F_neuSub , F_baseline = ca2p_preprocessing.make_dFoF(
    F=F + channelOffset_correction,
    Fneu=Fneu + channelOffset_correction,
    neuropil_fraction=params['make_dFoF']['neuropil_fraction'],
    percentile_baseline=percentile_baseline,
    multicore_pref=True,
    verbose=True
)



### Process motion correction data

# table_motionCorrection

offsets_z = np.stack([misc.get_nums_from_str(val) for val in table_motionCorrection['drPixel']], axis=0)[::3][:,2]

len_neural_data = dFoF.shape[1]
offsets_z_interp = scipy.interpolate.interp1d(table_motionCorrection['frameNumber'][::3], offsets_z, kind='cubic', bounds_error=False, fill_value=(offsets_z[0], offsets_z[-1]))(np.arange(len_neural_data))  ## interpolate dropped frames

### regress off z-motion

regressors = np.concatenate([
    np.stack([scipy.signal.savgol_filter(offsets_z_interp, win,3) for win in params['ransac']['z_offset_smoothing_windows']], axis=0).T,
    np.linspace(-1, 1, num=dFoF.shape[1])[:,None],
    featurization.mspline_grid(order=3, num_basis_funcs=3, nt=dFoF.shape[1]).T,
], axis=1)

regressors = np.concatenate((
    regressors,
    (regressors[:,0] * regressors[:,5])[:,None]/10,
    (regressors[:,0] * regressors[:,6])[:,None]/10,
    (regressors[:,0] * regressors[:,7])[:,None]/10,
), axis=1)

regressors = regressors.astype(np.float32)

# %matplotlib inline
# %matplotlib notebook


def regress_out(X, y, noise):

    clip_upperLim = (noise*params['ransac']['clip_scaling']) + params['ransac']['clip_offset']

    y_raw = y
    y = np.clip(y_raw, -clip_upperLim, clip_upperLim)

    ransac = sklearn.linear_model.RANSACRegressor(
        residual_threshold=clip_upperLim * params['ransac']['residual_threshold_scaling'],
        loss='squared_error', 
        min_samples=params['ransac']['min_samples']
    )
    ransac.fit(X=X, y=y)
#     inlier_mask = ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask)
    y_ransac = ransac.predict(X).T
    return y_ransac

snr, sig, noise = ca2p_preprocessing.snr_autoregressive(
    np.clip(
        dFoF, 
        params['ransac']['noise_calculation_clip_range'][0],
        params['ransac']['noise_calculation_clip_range'][1]
    ),
    axis=1, 
    center=True, 
    standardize=True,
)
dFoF_ransac = np.stack([regress_out(regressors, dFoF[idx], noise[idx]) for idx in tqdm(range(dFoF.shape[0]))], axis=0)

dFoF_denoised = dFoF - dFoF_ransac

v1_orth, EVR_ransac, EVR_total_weighted_ransac, EVR_total_unweighted_ransac = similarity.pairwise_orthogonalization(dFoF.T, dFoF_ransac.T, center=True)

print(f'Total Explain Variance Ratio of dFoF by dFoF_ransac: {EVR_total_weighted_ransac}')




### Inclusion criteria for dFoF

thresh_EVR_ransac = 0.10  ## Only ROIs with EVRs below this value will be kept
goodROIs_ransacEVR = EVR_ransac < thresh_EVR_ransac
badROIs_ransacEVR = np.logical_not(goodROIs_ransacEVR)
print(f'RANSAC EVR inclusion threshold: num good cells: {goodROIs_ransacEVR.sum()}, num bad cells: {badROIs_ransacEVR.sum()}')


tqm, goodROIs_tqm = ca2p_preprocessing.trace_quality_metrics(
    F, Fneu, dFoF_denoised, dF, F_neuSub, F_baseline,
    percentile_baseline=percentile_baseline, Fs=ops['fs'],
    plot_pref=False, 
    thresh=params['tqm']['thresh_dict'],
    clip_range=params['tqm']['clip_range'],
)


print(f'Trace quality metric inclusion: num good cells: {goodROIs_tqm.sum()}, num bad cells: {(~goodROIs_tqm).sum()}')

goodROIs = goodROIs_ransacEVR * goodROIs_tqm
print(f'Conjuntive denoising inclusion: num good cells: {goodROIs.sum()}, num bad cells: {(~goodROIs).sum()}')



### output plots

plt.figure()
plt.plot(regressors[:,:4]);
plt.xlabel('frame number')
plt.ylabel('z-offset (slices)')
plt.savefig(str(dir_save / 'z_offsets.png'))


plt.figure()
plt.plot(np.sort(EVR_ransac))
plt.plot([0,dFoF.shape[0]], [thresh_EVR_ransac, thresh_EVR_ransac], 'k')
plt.xlabel('ROI num (sorted by EVR_ransac)')
plt.ylabel('EVR_ransac')
plt.savefig(str(dir_save / 'EVR_ransac.png'))


fig, axs = plt.subplots(len(tqm['metrics']), figsize=(7,10))
for ii, val in enumerate(tqm['metrics']):
    if val=='peter_noise_levels':
        axs[ii].hist(tqm['metrics'][val][np.where(goodROIs_tqm==1)[0]], 300, histtype='step')
        axs[ii].hist(tqm['metrics'][val][np.where(goodROIs_tqm==0)[0]], 300, histtype='step')
        axs[ii].set_xlim([0,20])
    # elif val=='baseline_var':
    #     axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
    #     axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')
    #     axs[ii].set_xlim(right=50)
    #     # axs[ii].set_xscale('log')
    else:
        axs[ii].hist(tqm['metrics'][val][np.where(goodROIs_tqm==1)[0]], 300, histtype='step')
        axs[ii].hist(tqm['metrics'][val][np.where(goodROIs_tqm==0)[0]], 300, histtype='step')

    axs[ii].title.set_text(f"{val}: {np.sum(tqm['classifications'][val]==0)} excl")
    axs[ii].set_yscale('log')

    axs[ii].plot(np.array([tqm['thresh'][val],tqm['thresh'][val]])  ,  np.array([0,100]), 'k')
fig.legend(('thresh', 'included','excluded'))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(str(dir_save / 'trace_quality_metrics.png'))


plt.figure()
plt.plot(goodROIs_tqm)
plt.plot(scipy.signal.savgol_filter(goodROIs_tqm, 101,1))
plt.xlabel('ROI number')
plt.ylabel('included')
plt.savefig(str(dir_save / 'goodROIs_tqm__overROInumber.png'))

print(f'ROIs excluded: {int(np.sum(1-goodROIs_tqm))} / {len(goodROIs_tqm)}')
print(f'ROIs included: {int(np.sum(goodROIs_tqm))} / {len(goodROIs_tqm)}')

rmap = rastermap.Rastermap(
    n_components=1,
    n_X=40,
    nPC=200,
    init='pca',
    alpha=1.0,
    K=1.0,
    mode='basic',
    verbose=True,
    annealing=True,
    constraints=2,
)
embedding = rmap.fit_transform(np.clip(dFoF_denoised, -1, 1)[goodROIs])
isort_dFoF = rmap.isort

plt.figure(figsize=(25,10))
plt.imshow(
    np.clip(dFoF_denoised, -1, 1)[goodROIs][isort_dFoF], 
    vmin=-0.05, vmax=1, 
    aspect='auto'
)
plt.title('rastermap: denoised, good ROIs')
plt.savefig(str(dir_save / 'rastermap_denoised_goodROIs.png'))

rmap = rastermap.Rastermap(
    n_components=1,
    n_X=40,
    nPC=200,
    init='pca',
    alpha=1.0,
    K=1.0,
    mode='basic',
    verbose=True,
    annealing=True,
    constraints=2,
)
embedding = rmap.fit_transform(np.clip(dFoF_denoised, -1, 1)[~goodROIs])
isort_dFoF = rmap.isort

plt.figure(figsize=(25,10))
plt.imshow(
    np.clip(dFoF_denoised, -1, 1)[~goodROIs][isort_dFoF], 
    vmin=-0.05, vmax=1, 
    aspect='auto'
)
plt.title('rastermap: denoised, bad ROIs')
plt.savefig(str(dir_save / 'rastermap_denoised_badROIs.png'))



## Saving

ransac_denoising = {
    'dFoF_denoised': dFoF_denoised,
    'goodROIs_denoising': goodROIs,
    'trace_quality_metrics': {
        'tqm_obj': tqm,
        'goodROIs_tqm': goodROIs_tqm,
    },
    'ransac':{
        'dFoF_ransac': dFoF_ransac,
        'regressors': regressors,
        'EVR_ransac': EVR_ransac,
        'EVR_total_weighted_ransac': EVR_total_weighted_ransac,
        'thresh_EVR_ransac': thresh_EVR_ransac,
        'goodROIs_ransacEVR': goodROIs_ransacEVR,    
    }
}

path_helpers.mkdir(Path(path_save_denoisingVariables).parent)
pickle_helpers.simple_save(ransac_denoising, str(path_save_denoisingVariables))



