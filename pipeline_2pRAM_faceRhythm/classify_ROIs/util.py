from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn import Module

import PIL



def statFile_to_centered_spatialFootprints(path_statFile=None, statFile=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
    """
    Converts a stat file to a list of spatial footprint images.
    RH 2021

    Args:
        path_statFile (pathlib.Path or str):
            Path to the stat file.
            Optional: if statFile is provided, this
             argument is ignored.
        statFile (dict):
            Suite2p stat file dictionary
            Optional: if path_statFile is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    
    Returns:
        sf_all (list):
            List of spatial footprints images
    """
    assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "RH: 'out_height_width' must be list of 2 EVEN integers"
    assert max_footprint_width%2 != 0 , "RH: 'max_footprint_width' must be odd"
    if statFile is None:
        stat = np.load(path_statFile, allow_pickle=True)
    else:
        stat = statFile
    n_roi = stat.shape[0]
    
    # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
    sf_big_width = max_footprint_width # make odd number
    sf_big_mid = sf_big_width // 2

    sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
    for ii in range(n_roi):
        sf_big[ii , stat[ii]['ypix'] - np.int16(stat[ii]['med'][0]) + sf_big_mid, stat[ii]['xpix'] - np.int16(stat[ii]['med'][1]) + sf_big_mid] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

    sf = sf_big[:,  
                sf_big_mid - out_height_width[0]//2:sf_big_mid + out_height_width[0]//2,
                sf_big_mid - out_height_width[1]//2:sf_big_mid + out_height_width[1]//2]
    if plot_pref:
        plt.figure()
        plt.imshow(np.max(sf, axis=0)**0.2)
        plt.title('spatial footprints cropped MIP^0.2')
    
    return sf

def import_multiple_stat_files(paths_statFiles=None, dir_statFiles=None, fileNames_statFiles=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
    """
    Imports multiple stat files.
    RH 2021 
    
    Args:
        paths_statFiles (list):
            List of paths to stat files.
            Elements can be either str or pathlib.Path.
        dir_statFiles (str):
            Directory of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        fileNames_statFiles (list):
            List of file names of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.

    Returns:
        stat_all (list):
            List of stat files.
    """
    if paths_statFiles is None:
        paths_statFiles = [Path(dir_statFiles) / fileName for fileName in fileNames_statFiles]

    sf_all_list = [statFile_to_centered_spatialFootprints(path_statFile=path_statFile,
                                                 out_height_width=out_height_width,
                                                 max_footprint_width=max_footprint_width,
                                                 plot_pref=plot_pref)
                  for path_statFile in paths_statFiles]
    return sf_all_list

def import_multiple_label_files(paths_labelFiles=None, dir_labelFiles=None, fileNames_labelFiles=None, plot_pref=True):
    """
    Imports multiple label files.
    RH 2021

    Args:
        paths_labelFiles (list):
            List of paths to label files.
            Elements can be either str or pathlib.Path.
        dir_labelFiles (str):
            Directory of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        fileNames_labelFiles (list):
            List of file names of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        plot_pref (bool):
            If True, plots the label files.
    """
    if paths_labelFiles is None:
        paths_labelFiles = [Path(dir_labelFiles) / fileName for fileName in fileNames_labelFiles]

    labels_all_list = [np.load(path_labelFile, allow_pickle=True) for path_labelFile in paths_labelFiles]

    if plot_pref:
        for ii, labels in enumerate(labels_all_list):
            plt.figure()
            plt.hist(labels, 20)
            plt.title('labels ' + str(ii))
    return labels_all_list


def stat_to_sparse_spatial_footprints(
    stat, 
    frame_height=512, 
    frame_width=1024,
    dtype=np.uint8,
    ):
    """
    Imports and converts multiple stat files to spatial footprints
     suitable for CellReg.
    Output will be a list of arrays of shape (n_roi, height, width).
    RH 2022
    """
    import scipy.sparse
    from tqdm import tqdm

    isInt = np.issubdtype(dtype, np.integer)

    # stats = [np.load(path, allow_pickle=True) for path in paths_statFiles]
    num_rois = stat.size
    # sf_all = np.zeros((num_rois, frame_height, frame_width), dtype) 
    # for jj, roi in enumerate(stat):
    #     lam = np.array(roi['lam'])
    #     if isInt:
    #         lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
    #     else:
    #         lam = lam / lam.sum()
    #     sf_all[jj, roi['ypix'], roi['xpix']] = lam
    # return scipy.sparse.csr_matrix(sf_all.reshape(num_rois, -1))

    sf_all = np.zeros((num_rois, frame_height, frame_width), dtype) 
    # for jj, roi in enumerate(stat):
    def make_sf_sparse(roi):
        sf = np.zeros((frame_height, frame_width), dtype) 
        lam = np.array(roi['lam'])
        if isInt:
            lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
        else:
            lam = lam / lam.sum()
        sf[roi['ypix'], roi['xpix']] = lam
        return scipy.sparse.csr_matrix(sf.reshape(1, -1))
    sf_all = [make_sf_sparse(roi) for roi in tqdm(stat)]
    return scipy.sparse.vstack(sf_all)


def resize_affine(img, scale, clamp_range=False):
    """
    Wrapper for torchvision.transforms.Resize.
    Useful for resizing images to match the size of the images
     used in the training of the neural network.
    RH 2022
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
#         img=torch.as_tensor(img[None,...]),
        img=PIL.Image.fromarray(img),
        angle=0, translate=[0,0], shear=0,
        scale=scale,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    ))
    
    if clamp_range:
        clamp_high = img.max()
        clamp_low = img.min()
    
        img_rs[img_rs>clamp_high] = clamp_high
        img_rs[img_rs<clamp_low] = clamp_low
    
    return img_rs


class dataset_simCLR(Dataset):
    """    
    demo:
    
    transforms = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    
    torchvision.transforms.GaussianBlur(5,
                                        sigma=(0.01, 1.)),
    
    torchvision.transforms.RandomPerspective(distortion_scale=0.6, 
                                             p=1, 
                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                             fill=0),
    torchvision.transforms.RandomAffine(
                                        degrees=(-180,180),
                                        translate=(0.4, 0.4),
                                        scale=(0.7, 1.7), 
                                        shear=(-20, 20, -20, 20), 
                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                                        fill=0, 
                                        fillcolor=None, 
                                        resample=None),
    )
    scripted_transforms = torch.jit.script(transforms)

    dataset = util.dataset_simCLR(  torch.tensor(images), 
                                labels, 
                                n_transforms=2, 
                                transform=scripted_transforms,
                                DEVICE='cpu',
                                dtype_X=torch.float32,
                                dtype_y=torch.int64 )
    
    dataloader = torch.utils.data.DataLoader(   dataset,
                                            batch_size=64,
        #                                     sampler=sampler,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=False,
                                            num_workers=0,
                                            )
    """
    def __init__(   self, 
                    X, 
                    y, 
                    n_transforms=2,
                    class_weights=None,
                    transform=None,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64,
                    temp_uncertainty=1,
                    expand_dim=True
                    ):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021 / JZ 2021

        Args:
            X (torch.Tensor / np.array / list of float32):
                Images.
                Shape: (n_samples, height, width)
                Currently expects no channel dimension. If/when
                 it exists, then shape should be
                (n_samples, n_channels, height, width)
            y (torch.Tensor / np.array / list of ints):
                Labels.
                Shape: (n_samples)
            n_transforms (int):
                Number of transformations to apply to each image.
                Should be >= 1.
            transform (callable, optional):
                Optional transform to be applied on a sample.
                See torchvision.transforms for more information.
                Can use torch.nn.Sequential( a bunch of transforms )
                 or other methods from torchvision.transforms. Try
                 to use torch.jit.script(transform) if possible.
                If not None:
                 Transform(s) are applied to each image and the 
                 output shape of X_sample_transformed for 
                 __getitem__ will be
                 (n_samples, n_transforms, n_channels, height, width)
                If None:
                 No transform is applied and output shape
                 of X_sample_trasformed for __getitem__ will be 
                 (n_samples, n_channels, height, width)
                 (which is missing the n_transforms dimension).
            DEVICE (str):
                Device on which the data will be stored and
                 transformed. Best to leave this as 'cpu' and do
                 .to(DEVICE) on the data for the training loop.
            dtype_X (torch.dtype):
                Data type of X.
            dtype_y (torch.dtype):
                Data type of y.
        
        Returns:
            torch.utils.data.Dataset:
                torch.utils.data.Dataset object.
        """

        self.expand_dim = expand_dim
        
        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE) # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.X = self.X[:,None,...] if expand_dim else self.X
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.
        
        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        self.n_transforms = n_transforms

        self.temp_uncertainty = temp_uncertainty

        self.headmodel = None

        self.net_model = None
        self.classification_model = None
        
        
        # self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=DEVICE)

        # self.classModelParams_coef_ = mp.Array(np.ctypeslib.as_array(mp.Array(ctypes.c_float, feature)))

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')
    
    def tile_channels(X_in, dim=-3):
        """
        Expand dimension dim in X_in and tile to be 3 channels.

        JZ 2021 / RH 2021

        Args:
            X_in (torch.Tensor or np.ndarray):
                Input image. 
                Shape: [n_channels==1, height, width]

        Returns:
            X_out (torch.Tensor or np.ndarray):
                Output image.
                Shape: [n_channels==3, height, width]
        """
        dims = [1]*len(X_in.shape)
        dims[dim] = 3
        return torch.tile(X_in, dims)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves and transforms a sample.
        RH 2021 / JZ 2021

        Args:
            idx (int):
                Index / indices of the sample to retrieve.
            
        Returns:
            X_sample_transformed (torch.Tensor):
                Transformed sample(s).
                Shape: 
                    If transform is None:
                        X_sample_transformed[batch_size, n_channels, height, width]
                    If transform is not None:
                        X_sample_transformed[n_transforms][batch_size, n_channels, height, width]
            y_sample (int):
                Label(s) of the sample(s).
            idx_sample (int):
                Index of the sample(s).
        """

        y_sample = self.y[idx]
        idx_sample = self.idx[idx]
        
        if self.classification_model is not None:
            # features = self.net_model(tile_channels(self.X[idx][:,None,...], dim=1))
            # proba = self.classification_model.predict_proba(features.cpu().detach())[0]
            proba = self.classification_model.predict_proba(self.tile_channels(self.X[idx_sample][:,None,...], dim=-3))[0]
            
            # sample_weight = loss_uncertainty(torch.as_tensor(proba, dtype=torch.float32), temperature=self.temp_uncertainty)
            sample_weight = 1
        else:
            sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):

                # X_sample_transformed.append(tile_channels(self.transform(self.X[idx_sample]), dim=0))
                X_transformed = self.transform(self.X[idx_sample])
                X_sample_transformed.append(X_transformed)
        else:
            X_sample_transformed = self.tile_channels(self.X[idx_sample], dim=-3)
        
        return X_sample_transformed, y_sample, idx_sample, sample_weight



class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=0, n_channels=3):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"
        
class ScaleDynamicRange(Module):
    """
    Min-max scaling of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        
        self.epsilon = epsilon
    
    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max()+self.epsilon))
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.bounds})"

    

def set_device(use_GPU=True, verbose=True):
    """
    Set torch.cuda device to use.
    Assumes that only one GPU is available or
     that you wish to use cuda:0 only.
    RH 2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != "cuda:0":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"device: '{device}'") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device


class Classifier():
    def __init__(
        self,
        sk_logreg,
        pca_preprocessing=False,
        pca_meanSub=True,
        pc_loadings_nn=None,
        pc_loadings_swt=None,
        n_pcs_toKeep_nn=40,
        n_pcs_toKeep_swt=10,
    ):
        self.logreg = sk_logreg
        self.pca_preprocessing = pca_preprocessing
        self.pca_meanSub = pca_meanSub
        
        self.n_pcs_toKeep_nn = n_pcs_toKeep_nn
        self.n_pcs_toKeep_swt = n_pcs_toKeep_swt
        
        self.pc_loadings_nn = pc_loadings_nn[:, :n_pcs_toKeep_nn]
        self.pc_loadings_swt = pc_loadings_swt[:, :n_pcs_toKeep_swt]
        
    def __call__(
        self,
        output_nn,
        output_swt,
        return_preds_proba='preds'
    ):
        if self.pca_preprocessing:
            if self.pca_meanSub:
                output_nn_ms = output_nn - torch.mean(output_nn, dim=0)
                output_swt_ms = output_swt - torch.mean(output_swt, dim=0)
            scores_nn = output_nn_ms @ self.pc_loadings_nn
            scores_swt = output_swt_ms @ self.pc_loadings_swt
        else:
            scores_nn = output_nn
            scores_swt = output_swt
        
        scores_all = torch.cat([val / torch.std(val, dim=0).mean() for val in [scores_nn, scores_swt]], dim=1)
        
        if return_preds_proba == 'preds':
            return np.int64(self.logreg.predict(scores_all))
        if return_preds_proba == 'proba':
            return self.logreg.predict_proba(scores_all)
    def __repr__(self):
        return f'pca_preprocessing={self.pca_preprocessing}, pca_meanSub={self.pca_meanSub}, n_pcs_toKeep_nn={self.n_pcs_toKeep_swt}, n_pcs_toKeep_swt={self.n_pcs_toKeep_swt}'


## crappy heuristic for spreading out points

def get_spread_out_points(embeddings, thresh_dist=0.3, n_iter=3):
    import random
    import copy
    import sklearn

    def make_dist_mat(embeddings):
        dist_mat = sklearn.neighbors.NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None).fit(embeddings).kneighbors_graph(embeddings, n_neighbors=300, mode='distance').toarray()
        dist_mat[dist_mat==0.0] = np.nan
        return dist_mat

    def prune(dist_mat_pruned, thresh_dist, bool_good):
        idx_rand = np.random.permutation(np.arange(dist_mat_pruned.shape[0]))

        for idx in idx_rand:
            if np.nanmin(dist_mat_pruned[idx]) < thresh_dist:

                dist_mat_pruned[idx] = np.nan
                dist_mat_pruned[:,idx] = np.nan

                bool_good[idx] = False
        return dist_mat_pruned, bool_good

    def grow(dist_mat_pruned, dist_mat_raw, thresh_dist, bool_good):
        idx_good = np.nonzero(bool_good)[0]
        idx_bad = np.nonzero(~bool_good)[0]
        for idx in idx_bad:
            if np.nanmin(dist_mat_raw[idx][idx_good]) > thresh_dist:
                dist_mat_pruned[idx] = dist_mat_raw[idx]
                dist_mat_pruned[:,idx] = dist_mat_raw[:,idx]
                bool_good[idx] = True
        return dist_mat_pruned, bool_good

    dist_mat_raw = make_dist_mat(embeddings)

    n_sf = embeddings.shape[0]

    bool_good = np.ones(n_sf, dtype=np.bool8)
    
    dist_mat_pruned = copy.deepcopy(dist_mat_raw)
    
    for ii in range(n_iter):
        dist_mat_pruned, bool_good = prune(dist_mat_pruned, thresh_dist, bool_good)
        dist_mat_pruned, bool_good = grow(dist_mat_pruned, dist_mat_raw, thresh_dist, bool_good)
    dist_mat_pruned, bool_good = prune(dist_mat_pruned, thresh_dist, bool_good)

    return bool_good, dist_mat_pruned


## importing simple_cmap because I don't want to figure out cv2 import stuff necessary in plotting_helpers
def simple_cmap(colors, name='none'):
    """Create a colormap from a sequence of rgb values.
    Stolen with love from Alex (https://gist.github.com/ahwillia/3e022cdd1fe82627cbf1f2e9e2ad80a7ex)
    
    Args:
        colors (list):
            List of RGB values
        name (str):
            Name of the colormap

    Returns:
        cmap:
            Colormap

    Demo:
    cmap = simple_cmap([(1,1,1), (1,0,0)]) # white to red colormap
    cmap = simple_cmap(['w', 'r'])         # white to red colormap
    cmap = simple_cmap(['r', 'b', 'r'])    # red to blue to red
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # convert colors to rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})