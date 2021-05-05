import numpy as np
from pathlib import Path
import pandas as pd
    
def import_S2p(dir_S2p):
    dir_S2p = Path(dir_S2p).resolve()

    F = np.load(dir_S2p / 'F.npy')
    Fneu = np.load(dir_S2p / 'Fneu.npy')
    iscell = np.load(dir_S2p / 'iscell.npy')
    ops = np.load(dir_S2p / 'ops.npy', allow_pickle=True)
    spks = np.load(dir_S2p / 'spks.npy')
    stat = np.load(dir_S2p / 'stat.npy', allow_pickle=True)

    num_frames_S2p = F.shape[1]

    return F , Fneu , iscell , ops , spks , stat , num_frames_S2p


def import_ws(path_ws):
    import pywavesurfer.ws

    data_as_dict = pywavesurfer.ws.loadDataFile(filename=path_ws, format_string='double' )
    ws_data = data_as_dict[f'{list(data_as_dict.keys())[1]}']['analogScans']

    return ws_data


def import_roiClassifier(dir_roiClassifier):
    dir_roiClassifier = Path(dir_roiClassifier).resolve()

    IsCell_ROIClassifier = np.load(dir_roiClassifier / 'IsCell_ROIClassifier.npy')
    ROI_Classifier_manual_selection_vars = np.load(dir_roiClassifier / 'manual_selection_vars.npy', allow_pickle=True)
    
    return IsCell_ROIClassifier , ROI_Classifier_manual_selection_vars


def import_cameraCSV(path_cameraCSV):
    path_cameraCSV = Path(path_cameraCSV).resolve()

    cameraCSV = np.array(pd.read_csv(path_cameraCSV, sep=',',header=None))
    signal_GPIO = cameraCSV[:,0]
    
    return cameraCSV , signal_GPIO


def import_temporalFactorsFR(path_temporalFactorsFR):
    path_temporalFactorsFR = Path(path_temporalFactorsFR).resolve()

    temporalFactors_FR = np.load(path_temporalFactorsFR , allow_pickle=True)
    
    return temporalFactors_FR

