"""
Made on 2022.06.05 by RH

These functions are designed to temporally align the data from the 2pRAM
 face-rhythm experiment.
"""

import copy

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

import time

###################
##### IMPORTS #####
###################

def import_cameraCSV(path_cameraCSV):
    path_cameraCSV = Path(path_cameraCSV).resolve()

    cameraCSV = np.array(pd.read_csv(path_cameraCSV, sep=',',header=None))
    signal_GPIO = cameraCSV[:,0]
    
    return cameraCSV , signal_GPIO


#####################
##### ALIGNMENT #####
#####################

def get_pulseTimes(sig, thresh_derivative=0.5, pref_plot=True, path_save_plot=None):
    """
    This function takes in the signal from the
     X-Galvo and returns the frame times in values
     of the indices of the wavesurfer timeseries.
    RH 2022.06.05

    Args:
        sig (np.ndarray):
            A signal with rising edges to be found.
            shape: (n_samples,)
        thresh_derivative (float):
            The threshold for the derivative.
            Derivative peaks GREATER than this will
             be considered as frame times.
        pref_plot (bool):
            Whether to plot the signal, derivative,
             and the frame times.
        path_save_plot (str):
            The path to save the plot.
            If None, no plot will be saved.

    Returns:
        frameTimes (np.ndarray):
            The frame times in values of the indices
             of the wavesurfer timeseries.
    """
    if sig.ndim==1:
        sig = np.array(copy.deepcopy(sig))[:, None]

    signal_derivative = np.diff(sig, axis=0, n=1, prepend=0)
    sig_idxPeaks__idx_ws = [scipy.signal.find_peaks(sig > thresh_derivative)[0] for sig in signal_derivative.T]
    
    if pref_plot:
        [print(f'num peaks: {len(s)}') for s in sig_idxPeaks__idx_ws]
        
        fig = plt.figure(figsize=(40,10))
        plt.plot(sig)
        plt.plot(signal_derivative)
        plt.plot([0, sig.shape[0]], [thresh_derivative, thresh_derivative])
        # plt.plot(sig_idxPeaks__idx_ws, sig[sig_idxPeaks__idx_ws], 'o');
        [plt.plot(s, sig[:,ii][s], 'o') for ii,s in enumerate(sig_idxPeaks__idx_ws)]
        plt.xlim([0, 5*10**4])
        
        if path_save_plot:
            plt.savefig(path_save_plot)
    
    return sig_idxPeaks__idx_ws


def convert_camTimeDates_toAbsoluteSeconds(camTimeDates, verbose=False):

    camTimes_absolute = np.array(np.array(camTimeDates , dtype='datetime64') - np.datetime64(camTimeDates[0])  , dtype='float64')/10**9

    return camTimes_absolute

def align_camFrames_toWS(sig_camPulseTimes__idx_cam, sig_camDatetimesAbsolute__idx_cam, sig_camPulses__idx_ws):
    ## Get camera frame times in ws ind

    cp_c = sig_camPulseTimes__idx_cam
    cp_w = sig_camPulses__idx_ws
    ct = sig_camDatetimesAbsolute__idx_cam

    first_cpc = cp_c[0]
    last_cpc = cp_c[-1]
    ct_aligned = ct - ct[first_cpc]
    sig_wsIdx__idx_cam = (ct_aligned / ct_aligned[last_cpc]) * (cp_w[-1] - cp_w[0])
    sig_wsIdxRounded__idx_cam = np.int64(sig_wsIdx__idx_cam)

    return sig_wsIdx__idx_cam, sig_wsIdxRounded__idx_cam 

def convert_camTimeDates_toAbsoluteSeconds(camTimeDates, verbose=False):

    tic = time.time()
    n_timepoints = len(camTimeDates)
    camTimes_absolute = np.array(np.array(camTimeDates , dtype='datetime64') - np.datetime64(camTimeDates[0])  , dtype='float64')/10**9

    if verbose:
        print(f'Completed converting camera dates from camera data to absolute time. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return camTimes_absolute