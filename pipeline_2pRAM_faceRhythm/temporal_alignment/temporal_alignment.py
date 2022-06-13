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
#     'paths': {
#         'dir_github': '/media/rich/Home_Linux_partition/github_repos/',
#         'dir_s2p': r'/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_16/suite2p_output/suite2p/plane0',
#         'path_ws': r'/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_16/AEG21_2022-05-16_0001.h5',
#         'path_cameraCSV': r'/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_16/20220516AEG21_csv32022-05-16T14_27_31.csv',
#         'path_faceRhythmNWB': r'/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_16/faceRhythm/jobNum_0/batchRun/data/session_batch.nwb',
#     },
#     'device_interp': 'cuda:0',
# }

import time

print(f'## IMPORT LIBRARIES.  time: {time.ctime()}')

import sys
sys.path.append(params['paths']['dir_github'])

from basic_neural_processing_modules import  h5_handling, spectral, indexing

from NBAP.pipeline_2pRAM_faceRhythm.temporal_alignment import util


import copy

import pywavesurfer.ws
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pynwb

import torch

from pathlib import Path
from tqdm import tqdm

import torchinterp1d




print(f'## IMPORT DATA.  time: {time.ctime()}')

dir_s2p              = Path(params['paths']['dir_s2p'])

path_ws              = Path(params['paths']['path_ws'])

path_cameraCSV       = Path(params['paths']['path_cameraCSV'])

path_faceRhythmNWB   = Path(params['paths']['path_faceRhythmNWB'])

F = np.load(dir_s2p / 'F.npy')

ws_dict = pywavesurfer.ws.loadDataFile(filename=path_ws, format_string='double' )

cameraCSV , signal_GPIO = util.import_cameraCSV(path_cameraCSV)

h5_handling.dump_nwb(path_faceRhythmNWB)


with pynwb.NWBHDF5IO(path_faceRhythmNWB, 'r') as io:
    nwbfile = io.read()
    
    pos_CDR = nwbfile.processing['Face Rhythm']['Optic Flow']['positions_convDR_meanSub'].data[:]




print(f'## PREPROCESS DATA.  time: {time.ctime()}')

## Preprocess some signals

ws_channel_idx = [
    'behaviorPulse',
    'acc_z',
    'acc_y',
    'acc_x',
    'lick_R',
    'lick_L',
    'X_galvo',
    'tone',
    'masterClock',
    'camPulses',
]

ws = pd.DataFrame(
    data=np.vstack((ws_dict[list(ws_dict.keys())[1]]['analogScans'], ws_dict[list(ws_dict.keys())[1]]['digitalScans'])).T,
    columns=ws_channel_idx,
)


print(f'## Convert tone and accelerometer signals with Variable Q Transform.  time: {time.ctime()}')

pref_plot_vqt = False
pref_plot_vqtOutputs = True
device = 'cpu'

### convert the tone signal into bandpassed enveloped
vqt = spectral.VQT(
    Fs_sample=1000,
    Q_lowF=2,
    Q_highF=2,
    F_min=70,
    F_max=110,
    n_freq_bins=1,
    win_size=201,
    downsample_factor=1,
    DEVICE_compute=device,
    DEVICE_return='cpu',
    return_complex=False,
    filters=None,
    plot_pref=pref_plot_vqt,
)
sig_toneSpec__idx_ws = vqt(np.array(ws['tone']))[0].numpy()[0]

if pref_plot_vqtOutputs:
    plt.figure()
    plt.plot(sig_toneSpec__idx_ws)

### convert accelerometer signal into bandpassed envelope
###  and average x,y,z together
vqt = spectral.VQT(
    Fs_sample=1000,
    Q_lowF=3,
    Q_highF=3,
    F_min=25,
    F_max=25,
    n_freq_bins=1,
    win_size=501,
    downsample_factor=1,
    DEVICE_compute=device,
    DEVICE_return='cpu',
    return_complex=False,
    filters=None,
    plot_pref=pref_plot_vqt,
)
sig = np.array(ws[['acc_x', 'acc_y', 'acc_z']]).T
sig = sig - sig.mean(1, keepdims=True)
sig_accSpec__idx_ws = vqt(sig).numpy().squeeze().mean(0)

if pref_plot_vqtOutputs:
    plt.figure()
    plt.plot(sig_accSpec__idx_ws)


print(f'## PREPROCESS DATA.  time: {time.ctime()}')

## Get event times for different wavesurfer signals

pref_plot_peakFinding = False

print(f'## finding peaks in wavesurfer signals. time: {time.ctime()}')

sig_SIFrameTimes__idx_ws = util.get_pulseTimes(ws['X_galvo'] > 0, thresh_derivative=0.5, pref_plot=pref_plot_peakFinding)[0]

sig_lickTimes__idx_ws = util.get_pulseTimes(np.vstack((ws['lick_L'] , ws['lick_R'])).T * -1, thresh_derivative=0.5, pref_plot=pref_plot_peakFinding)

sig_camPulseTimes__idx_ws = util.get_pulseTimes(sig=ws['camPulses'] > 3.5, thresh_derivative=0.5, pref_plot=pref_plot_peakFinding)[0]
sig_camPulseTimes__idx_cam = util.get_pulseTimes(sig=cameraCSV[:,0], thresh_derivative=0.5, pref_plot=pref_plot_peakFinding)[0]

sig_toneTimes__idx_ws = scipy.signal.find_peaks(sig_toneSpec__idx_ws * (sig_toneSpec__idx_ws>0.5), distance=200)[0]
if pref_plot_peakFinding:
    plt.figure(); plt.plot(sig_toneSpec__idx_ws); plt.plot(sig_toneTimes__idx_ws, sig_toneSpec__idx_ws[sig_toneTimes__idx_ws], '.')

## Check to make sure indices check out

match__wsFlybacks_s2pFrames = (len(sig_SIFrameTimes__idx_ws) == F.shape[1])
print(f'number of flybacks matches number of s2p samples') if match__wsFlybacks_s2pFrames else print(f'WARNING: number of ws flybacks {len(sig_SIFrameTimes__idx_ws)} does not match number of s2p frames {F.shape[1]}')

match__wsFlybacks_s2pFrames = (len(sig_camPulseTimes__idx_cam) == len(sig_camPulseTimes__idx_ws))
print(f'number of camPulses sent/received on ws matches number of camPulses from camera CSV file') if match__wsFlybacks_s2pFrames else print(f'WARNING: number of sent camPulses from ws {len(sig_camPulseTimes__idx_cam)} does not match number of received camPulses from camera CSV file {len(sig_camPulseTimes__idx_ws)}')

### convert some wavesurfer signals into SI indices

## first, convert the binary signals from ws into boolean traces, still in ws indices
print(f'## converting wavesurfer signal event times into boolean traces. time: {time.ctime()}')

num_samples_ws = ws.shape[0]

sig_lickBool__idx_ws = np.array([indexing.idx2bool(sig, length=num_samples_ws) for sig in sig_lickTimes__idx_ws]).T

sig_toneBool__idx_ws = indexing.idx2bool(sig_toneTimes__idx_ws, length=num_samples_ws)


## second, integrate boolean signals over SI indices
print(f'## converting wavesurfer boolean signals into SI indices. time: {time.ctime()}')

num_frames_SI = len(sig_SIFrameTimes__idx_ws)
print(f'number of frames from scanimage movie: {num_frames_SI}')

sig_lickBool__idx_SI = np.zeros((num_frames_SI, sig_lickBool__idx_ws.shape[1]))
for i_frame, frame_ind in enumerate(sig_SIFrameTimes__idx_ws):
    if i_frame==0:
        continue
    sig_lickBool__idx_SI[i_frame] = np.sum(sig_lickBool__idx_ws[sig_SIFrameTimes__idx_ws[i_frame-1] : sig_SIFrameTimes__idx_ws[i_frame]], axis=0) > 0.5

## third, convert the analog signals from ws into SI indices
print(f'## converting wavesurfer analog signals into SI indices. time: {time.ctime()}')

sig_accSpec_idx_SI = scipy.interpolate.interp1d(
    x=np.arange(0, num_samples_ws),
    y=sig_accSpec__idx_ws,
    kind='cubic',
)(sig_SIFrameTimes__idx_ws)

sig_toneSpec_idx_SI = scipy.interpolate.interp1d(
    x=np.arange(0, num_samples_ws),
    y=sig_toneSpec__idx_ws,
    kind='cubic',
)(sig_SIFrameTimes__idx_ws)


## Prepare camPulse times for interpolation
print(f'## creating SI indices. time: {time.ctime()}')

sig_camPulses__idx_cam = cameraCSV[:,0]
sig_camDatetimes__idx_cam = cameraCSV[:,3]

sig_camPulseTimes__idx_cam = util.get_pulseTimes(
    sig=sig_camPulses__idx_cam,
    thresh_derivative=0.5,
    pref_plot=True
)[0]

sig_camDatetimesAbsolute__idx_cam = util.convert_camTimeDates_toAbsoluteSeconds(sig_camDatetimes__idx_cam)

sig_wsIdx__idx_cam, sig_wsIdxRounded__idx_cam = util.align_camFrames_toWS(
    sig_camPulseTimes__idx_cam=sig_camPulseTimes__idx_cam,
    sig_camDatetimesAbsolute__idx_cam=sig_camDatetimesAbsolute__idx_cam,
    sig_camPulses__idx_ws=sig_camPulseTimes__idx_ws
)




print(f'## SPECTROGRAM CALCULATION AND INTERPOLATION.  time: {time.ctime()}')

device = params['device_interp']

vqt = spectral.VQT(
    Fs_sample=270,
    Q_lowF=2.5,
    Q_highF=10,
    F_min=1.7,
    F_max=50,
    n_freq_bins=30,
    win_size=1001,
    downsample_factor=1,
    DEVICE_compute=device,
    DEVICE_return=device,
    return_complex=False,
    filters=None,
    plot_pref=True,
    progressBar=False,
)

interp = torchinterp1d.interp1d.Interp1d()

y = torch.as_tensor(pos_CDR[:,:,:], dtype=torch.float32)

x = torch.as_tensor(sig_wsIdx__idx_cam, dtype=torch.float32)
xnew = torch.as_tensor(sig_SIFrameTimes__idx_ws, dtype=torch.float32)

x_tiled = torch.tile(x, (len(vqt.freqs), 1)).to(device)
xnew_tiled = xnew.to(device)

sig_Sxx__idx_s2p = torch.stack([
    torch.stack([
        interp(
            x=x_tiled, 
            y=vqt(y[ii,jj,...]).squeeze(), 
            xnew=xnew_tiled,
        ).cpu() for ii in tqdm(range(y.shape[0]))
    ], dim=0) for jj in range(y.shape[1])
], dim=0).permute(1,0,2,3)

spectrogram_exponent = 1

tmp = (sig_Sxx__idx_s2p * vqt.freqs[None,None,:,None]) ** spectrogram_exponent
tmp_normFactor = torch.mean(tmp, dim=(0,1,2))

norm_factor = 0.9

sig_SxxNorm__idx_s2p = tmp / ((tmp_normFactor[None,None,None,:] * norm_factor) + (1-norm_factor))
sig_SxxNorm__idx_s2p = sig_SxxNorm__idx_s2p.type(torch.float32)


x_tiled = torch.tile(x, (y.shape[1], 1)).to(device)
xnew_tiled = xnew.to(device)

sig_posCDR__idx_s2p = torch.stack([interp(x=x_tiled, y=y[ii,:,:].to(device), xnew=xnew_tiled).cpu() for ii in tqdm(range(y.shape[0]))], dim=0)




print(f'## SAVE OUTPUTS.  time: {time.ctime()}')
## Save outputs

np.save(str(Path(dir_save) / 'SxxNorm_idxS2p.npy'), sig_SxxNorm__idx_s2p)
np.save(str(Path(dir_save) / 'positionsCDR_idxS2p.npy'), sig_posCDR__idx_s2p)

h5_handling.simple_save(
    dict_to_save={
        'sig_lickBool__idx_SI': sig_lickBool__idx_SI,
        'sig_accSpec_idx_SI': sig_accSpec_idx_SI,
        'sig_toneSpec_idx_SI': sig_toneSpec_idx_SI,
    },
    path=str(Path(dir_save) / 'ws_signals_aligned_to_SI.h5'),
)

print(f'## RUN COMPLETED.  time: {time.ctime()}')



# resource requirements
# RAM: 64GB
# VRAM: 8GB
# CPU: any number
# 10 min