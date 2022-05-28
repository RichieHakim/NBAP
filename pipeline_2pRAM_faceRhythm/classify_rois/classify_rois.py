# # widen jupyter notebook window
# from IPython.display import display, HTML
# display(HTML("<style>.container {width:95% !important; }</style>"))

# check environment
import os
print(f'Conda Environment: ' + os.environ['CONDA_DEFAULT_ENV'])

from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import scipy.stats

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
#     'dir_github': '/media/rich/Home_Linux_partition/github_repos/',
#     'dir_s2p': '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/suite2p_o2_output/jobNum_0/suite2p/plane0/',
#     'path_params_nnTraining': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/params.json',
#     'path_state_dict': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/network/ConvNext_tiny__1_0_unfrozen__simCLR.pth',
#     'path_classifier_vars': '/media/rich/Home_Linux_partition/github_repos/NBAP/pipeline_2pRAM_faceRhythm/classify_ROIs/classifier.pkl',
#     'pref_saveFigs': False,
#     'useGPU': True,
#     'classes_toInclude': [0,1,2]
# }

import sys
sys.path.append(params['dir_github'])


# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import pickle_helpers, indexing, torch_helpers

# %load_ext autoreload
# %autoreload 2
from NBAP.pipeline_2pRAM_faceRhythm.classify_rois import util


dir_save_network_files = str(Path(dir_save).resolve() / 'network_files')

import gdown
gdown.download_folder(id=params['gdriveID_networkFiles'], output=dir_save_network_files, quiet=True, use_cookies=False)
sys.path.append(dir_save_network_files)
print(sys.path)
import model

path_state_dict = str(Path(dir_save_network_files).resolve() / params['fileName_state_dict'])
path_nnTraining = str(Path(dir_save_network_files).resolve() / params['fileName_params_nnTraining'])
# path_model = str(Path(dir_save_network_files).resolve() / params['fileName_model'])
path_classifier = str(Path(dir_save_network_files).resolve() / params['fileName_classifier'])




path_stat = str(Path(params['dir_s2p']) / 'stat.npy')
path_ops = str(Path(params['dir_s2p']) / 'ops.npy')

sf_all = util.import_multiple_stat_files(   
    paths_statFiles=[path_stat],
    out_height_width=[36,36],
    max_footprint_width=1441,
    plot_pref=True
)

sf_ptiles = np.array([np.percentile(np.sum(sf>0, axis=(1,2)), 90) for sf in tqdm(sf_all)])

scales_forRS = (250/sf_ptiles)**0.6

sf_rs = [np.stack([util.resize_affine(img, scale=scales_forRS[ii], clamp_range=True) for img in sf], axis=0) for ii, sf in enumerate(tqdm(sf_all))]

sf_all_cat = np.concatenate(sf_all, axis=0)
sf_rs_concat = np.concatenate(sf_rs, axis=0)

import scipy.signal

figs, axs = plt.subplots(2,1, figsize=(7,10))
axs[0].plot(np.sum(sf_all_cat > 0, axis=(1,2)))
axs[0].plot(scipy.signal.savgol_filter(np.sum(sf_all_cat > 0, axis=(1,2)), 501, 3))
axs[0].set_xlabel('ROI number');
axs[0].set_ylabel('mean npix');
axs[0].set_title('ROI sizes raw')

axs[1].plot(np.sum(sf_rs_concat > 0, axis=(1,2)))
axs[1].plot(scipy.signal.savgol_filter(np.sum(sf_rs_concat > 0, axis=(1,2)), 501, 3))
axs[1].set_xlabel('ROI number');
axs[1].set_ylabel('mean npix');
axs[1].set_title('ROI sizes resized')

if params['pref_saveFigs']:
    plt.savefig(str(Path(dir_save) / 'ROI_sizes.png'))

transforms_classifier = torch.nn.Sequential(
    util.ScaleDynamicRange(scaler_bounds=(0,1)),
    
    torchvision.transforms.Resize(
        size=(224, 224),
#         size=(180, 180),
#         size=(72, 72),        
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR), 
    
    util.TileChannels(dim=0, n_channels=3),
)

scripted_transforms_classifier = torch.jit.script(transforms_classifier)


dataset_labeled = util.dataset_simCLR(
        X=torch.as_tensor(sf_rs_concat, device='cpu', dtype=torch.float32),
        y=torch.as_tensor(torch.zeros(sf_rs_concat.shape[0]), device='cpu', dtype=torch.float32),
        n_transforms=1,
        class_weights=np.array([1]),
        transform=scripted_transforms_classifier,
        DEVICE='cpu',
        dtype_X=torch.float32,
    )
    
dataloader_labeled = out = torch.utils.data.DataLoader( 
        dataset_labeled,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=36,
        persistent_workers=True,
#         prefetch_factor=2
)



import json
with open(path_nnTraining) as f:
    params_nnTraining = json.load(f)

model_nn = model.make_model(
    torchvision_model=params_nnTraining['torchvision_model'],
    n_block_toInclude=params_nnTraining['n_block_toInclude'],
    pre_head_fc_sizes=params_nnTraining['pre_head_fc_sizes'],
    post_head_fc_sizes=params_nnTraining['post_head_fc_sizes'],
    head_nonlinearity=params_nnTraining['head_nonlinearity'],
    image_shape=[3, 224, 224],
#     image_shape=[params_nnTraining['augmentation']['TileChannels']['n_channels']] + params_nnTraining['augmentation']['WarpPoints']['img_size_out']
);

for param in model_nn.parameters():
    param.requires_grad = False
model_nn.eval();

# model_nn.load_state_dict(torch.load(params['path_state_dict']))
model_nn.load_state_dict(torch.load(path_state_dict))

DEVICE = torch_helpers.set_device(use_GPU=params['useGPU'])

model_nn = model_nn.to(DEVICE)

features_nn = torch.cat([model_nn(data[0][0].to(DEVICE)).detach() for data in tqdm(dataloader_labeled)], dim=0).cpu()

from kymatio import Scattering2D

def get_latents_swt(sfs, swt, device_model):
    sfs = torch.as_tensor(np.ascontiguousarray(sfs[None,...]), device=device_model, dtype=torch.float32)
    latents_swt = swt(sfs[None,...]).squeeze()
    latents_swt = latents_swt.reshape(latents_swt.shape[0], -1)
    return latents_swt


scattering = Scattering2D(J=2, L=8, shape=sf_rs_concat[0].shape[-2:])
if DEVICE != 'cpu':
    scattering = scattering.cuda()

latents_swt = get_latents_swt(sf_rs_concat, scattering.cuda(), DEVICE).cpu()



from util import Classifier
classifier_vars = pickle_helpers.simple_load(params['path_classifier_vars'])
classifier = classifier_vars['classifier']

preds = classifier(features_nn, latents_swt, return_preds_proba='preds')
proba = classifier(features_nn, latents_swt, return_preds_proba='proba')

goodROIs = np.isin(preds, params['classes_toInclude'])

print(f'num good ROIs: {goodROIs.sum()}, num bad ROIs: {(~goodROIs).sum()}')

from umap import UMAP
umap = UMAP(
    n_neighbors=30,
    n_components=2,
    metric='euclidean',
    metric_kwds=None,
    output_metric='euclidean',
    output_metric_kwds=None,
    n_epochs=None,
    learning_rate=1.0,
    init='spectral',
    min_dist=0.1,
    spread=1.0,
    low_memory=True,
    n_jobs=-1,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    transform_queue_size=4.0,
    a=None,
    b=None,
    random_state=None,
    angular_rp_forest=False,
    target_n_neighbors=-1,
    target_metric='categorical',
    target_metric_kwds=None,
    target_weight=0.5,
    transform_seed=42,
    transform_mode='embedding',
    force_approximation_algorithm=False,
    verbose=False,
    tqdm_kwds=None,
    unique=False,
    densmap=False,
    dens_lambda=2.0,
    dens_frac=0.3,
    dens_var_shift=0.1,
    output_dens=False,
    disconnection_distance=None,
    precomputed_knn=(None, None, None),
)

emb_umap = umap.fit_transform(features_nn)
# emb_nn = umap.fit_transform(scores_nn)
# emb_swt = umap.fit_transform(latents_swt)

bool_good, dist_mat_pruned = util.get_spread_out_points(emb_umap, thresh_dist=0.25, n_iter=10)
idx_good = np.nonzero(bool_good)[0]

# %matplotlib notebook
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(img):
    return OffsetImage(img, cmap='gray')

x, y, ims_subset = emb_umap[idx_good,0], emb_umap[idx_good,1], sf_rs_concat[idx_good]

fig, ax = plt.subplots(figsize=(20,20))

ax.scatter(emb_umap[:,0], emb_umap[:,1], cmap=plt.get_cmap('gist_rainbow'), c=preds)

for x0, y0, path in zip(x, y, ims_subset):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)

if params['pref_saveFigs']:
    plt.savefig(str(Path(dir_save) / 'UMAP.png'))



stat = np.load(path_stat, allow_pickle=True)
ops = np.load(path_ops, allow_pickle=True)[()]

sf_sparse = util.stat_to_sparse_spatial_footprints(
    stat=stat,
    frame_height=ops['Ly'], frame_width=ops['Lx'],
    dtype=np.float32,
)

import sparse
sf_sparse_scaled = sf_sparse.multiply(sf_sparse.max(axis=1).power(-1))
sf_sparse_scaled_rsFOV = sparse.COO(sf_sparse_scaled).reshape((len(stat), ops['Ly'], ops['Lx']))

colors = util.simple_cmap(([0,0.6,1], [0,0.9,0.2], [0.7,0.5,0], [1,0,0]))
n_classes = 4
sf_sparse_scaled_rsFOV_colored = np.stack([(sf_sparse_scaled_rsFOV[preds==ii,:,:,None] * np.array(colors(ii/(n_classes-1)))[None,None,:3]).sum(0) for ii in range(n_classes)], axis=-1).sum(-1).todense()

def reshape_mROI_to_square(image, Ly, Lx):
    Ly = (Ly//3)*3
    return np.reshape(image.T[:,:Ly], (Lx, (Ly//3), 3), order='F').transpose(1,0,2).reshape(Ly//3, Lx*3, order='F')

sf_sparse_scaled_rsFOV_colored_square = np.stack([reshape_mROI_to_square(image, ops['Ly'], ops['Lx']) for image in sf_sparse_scaled_rsFOV_colored.transpose((2,0,1))], axis=-1)

plt.figure(figsize=(40,30))
plt.imshow(sf_sparse_scaled_rsFOV_colored_square, aspect='auto')

if params['pref_saveFigs']:
    plt.savefig(str(Path(dir_save) / 'FOV_colored.png'))



classification_output = {
    'goodROIs': goodROIs,
    'preds': preds,
    'proba': proba,
    'embedding_umap': emb_umap,
    'sf_sparse': sf_sparse,
}

pickle_helpers.simple_save(classification_output, str(Path(dir_save) / 'classification_output.pkl'))