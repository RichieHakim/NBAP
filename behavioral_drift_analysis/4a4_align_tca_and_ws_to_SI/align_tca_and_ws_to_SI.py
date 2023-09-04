from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate

import bnpm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_ws', type=str, required=True)
parser.add_argument('--path_csv', type=str, required=True)
parser.add_argument('--path_tca_all', type=str, required=True)
parser.add_argument('--path_idxLaserCam', type=str, required=True)
parser.add_argument('--path_vqt', type=str, required=True)
parser.add_argument('--directory_save', type=str, required=True)
args = parser.parse_args()
path_ws, path_csv, path_tca_all, path_idxLaserCam, path_vqt, directory_save = args.path_ws, args.path_csv, args.path_tca_all, args.path_idxLaserCam, args.path_vqt, args.directory_save

date = Path(path_ws).parent.name

# path_ws          = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/wavesurfer_files/20230510/exp_0001.h5'
# path_csv         = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/cam4_CSVs/20230510/times_cam42023-05-10T13_12_02.csv'
# path_tca_all     = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/run_20230701/factors_refit_bigTCA.h5'
# path_idxLaserCam = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/eye_laser_trace_extraction/20230510/jobNum_0/idx_eye_laser.pkl'
# path_vqt         = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/run_20230701/20230510/jobNum_0/analysis_files/VQT_Analyzer.h5'

# directory_save   = r'/media/rich/bigSSD/analysis_data/face_rhythm/mouse_0322N/'



ts = {}

ws_raw = bnpm.h5_handling.simple_load(
    filepath=path_ws,
    return_dict=True,
    verbose=True,
)

names_ws = [
    'sync_pulse',
    'laser_pickoff',
    'treadmill',
    'lickometer',
    'rewards',
    'cursor',
    'y_galvo',
]

name_sweep = list(ws_raw.keys())[-1]
print(f"using sweep: {name_sweep}")
ws = {name: trace for name, trace in zip(names_ws, ws_raw[name_sweep]['analogScans'])}

trace = ws['laser_pickoff']

def get_laser_pickoff_start_and_end_indices(trace, idx_fastForward=60*59*120, starting_frames=1000):
    get_diff_smooth = lambda x: np.diff(bnpm.timeSeries.simple_smooth(x, sig=4), n=1)

    trace_start = trace[:starting_frames]
    idx_fastForward = int(idx_fastForward)
    trace_end = trace[idx_fastForward:]
    idx_start = np.argmax(get_diff_smooth(trace_start))
    idx_end = np.argmin(get_diff_smooth(trace_end)) + idx_fastForward
    return idx_start, idx_end

val_idxWsStartLaser, val_idxWsEndLaser = get_laser_pickoff_start_and_end_indices(
    trace=ws['laser_pickoff'],
    idx_fastForward=60*59*120, 
    starting_frames=1000,
)

ts['val_idxWsStartLaser'] = val_idxWsStartLaser
ts['val_idxWsEndLaser'] = val_idxWsEndLaser







csv = pd.read_csv(
    filepath_or_buffer=path_csv,
    delimiter=',',
    header=None,
)

val_abstimeModulated__idx_cam = csv[2]

ts['val_abstimeModulated__idx_cam'] = val_abstimeModulated__idx_cam

val_abstime__idx_cam = bnpm.indexing.moduloCounter_to_linearCounter(
    trace=ts['val_abstimeModulated__idx_cam'],
    modulus=2**32,
    plot_pref=False,
)

ts['val_abstimeModulated__idx_cam'] = val_abstimeModulated__idx_cam
ts['val_abstime__idx_cam'] = val_abstime__idx_cam









tca_all = bnpm.h5_handling.simple_load(
    filepath=path_tca_all,
    return_dict=True,
    verbose=True,
)

tca = tca_all[date]

val_tca__idx_tca = tca['time']

ts['val_tca__idx_tca'] = val_tca__idx_tca

val_yGalvo__idx_ws = ws['y_galvo']

ts['val_yGalvo__idx_ws'] = val_yGalvo__idx_ws





# do the interpolation

peaks = scipy.signal.find_peaks(
    x=-np.diff(ws['y_galvo'], n=1),
    height=200,
)[0] + 1

val_idxWs__idx_SI = peaks

ts['val_idxWs__idx_SI'] = val_idxWs__idx_SI

val_idxTca__idx_tca = np.arange(val_tca__idx_tca.shape[0])

ts['val_idxTca__idx_tca'] = val_idxTca__idx_tca

val_tca__idx_SI = scipy.interpolate.interp1d(
    x=ts['val_idxTca__idx_tca'],
    y=ts['val_tca__idx_tca'],
    kind='cubic',
    axis=0,
    bounds_error=False,
    fill_value=np.nan,
)(ts['val_idxWs__idx_SI'])





idx_eye_laser = bnpm.file_helpers.pickle_load(path_idxLaserCam)

val_idxCamStartLaser = idx_eye_laser['idx_start']
val_idxCamEndLaser   = idx_eye_laser['idx_end']

ts['val_idxCamStartLaser'] = val_idxCamStartLaser
ts['val_idxCamEndLaser']   = val_idxCamEndLaser





vqt = bnpm.h5_handling.simple_load(
    filepath=path_vqt,
    return_dict=False,
    verbose=True,
)

val_idxCam__idx_tca = vqt['x_axis']['0'][:]

ts['val_idxCam__idx_tca'] = val_idxCam__idx_tca

n_frames_cam = len(ts['val_abstime__idx_cam'])

val_idxCam__idxCam = np.arange(n_frames_cam)
# val_idxNormLaser__idx_cam = (val_idxCam__idxCam - ts['val_idxCamStartLaser']) / ts['val_idxCamEndLaser']
t = ts['val_abstime__idx_cam']
val_idxNormLaser__idx_cam = (t - t[ts['val_idxCamStartLaser']]) / (t[ts['val_idxCamEndLaser']] - t[ts['val_idxCamStartLaser']])

ts['val_idxNormLaser__idx_cam'] = val_idxNormLaser__idx_cam
ts['val_idxCam__idxCam'] = val_idxCam__idxCam



val_idxNormLaser__idx_tca = scipy.interpolate.interp1d(
    x=ts['val_idxCam__idxCam'],
    y=ts['val_idxNormLaser__idx_cam'],
    kind='linear',
    bounds_error=False,
)(ts['val_idxCam__idx_tca'])

ts['val_idxNormLaser__idx_tca'] = val_idxNormLaser__idx_tca

n_frames_ws = len(ts['val_yGalvo__idx_ws'])

val_idxWs__idx_ws = np.arange(n_frames_ws)

val_idxNormLaser__idx_ws = (val_idxWs__idx_ws - ts['val_idxWsStartLaser']) / ts['val_idxWsEndLaser']

ts['val_idxNormLaser__idx_ws'] = val_idxNormLaser__idx_ws
ts['val_idxWs__idx_ws'] = val_idxWs__idx_ws

val_idxNormLaser__idx_SI = scipy.interpolate.interp1d(
    x=ts['val_idxWs__idx_ws'],
    y=ts['val_idxNormLaser__idx_ws'],
    kind='linear',
    bounds_error=False,
)(ts['val_idxWs__idx_SI'])

ts['val_idxNormLaser__idx_SI'] = val_idxNormLaser__idx_SI


val_tca__idx_SI = scipy.interpolate.interp1d(
    x=ts['val_idxNormLaser__idx_tca'],
    y=ts['val_tca__idx_tca'],
    axis=0,
    kind='cubic',
    bounds_error=False,
    fill_value=np.nan,
)(ts['val_idxNormLaser__idx_SI'])

ts['val_tca__idx_SI'] = val_tca__idx_SI



val_ws__idx_SI = {key: scipy.interpolate.interp1d(
    x=ts['val_idxNormLaser__idx_ws'],
    y=val_ws__idx_ws,
    axis=0,
    kind='cubic',
    bounds_error=False,
    fill_value=np.nan,
)(ts['val_idxNormLaser__idx_SI']) for key, val_ws__idx_ws in ws.items()}

ts['val_ws__idx_SI'] = val_ws__idx_SI



Path(directory_save).mkdir(parents=True, exist_ok=True)

bnpm.file_helpers.pickle_save(
    obj=ts,
    filepath=str(Path(directory_save) / 'alignment_ts.pkl'),
)

bnpm.file_helpers.pickle_save(
    obj=ts['val_ws__idx_SI'],
    filepath=str(Path(directory_save) / 'ws_idxSI.pkl'),
)

np.save(
    file=str(Path(directory_save) / 'tca_idxSI.npy'),
    arr=ts['val_tca__idx_SI'],
)