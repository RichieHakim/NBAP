from pathlib import Path

import numpy as np
import torch
import decord

import bnpm

import sys
path_script, path_params, dir_save = sys.argv

params = bnpm.file_helpers.json_load(path_params)

path_vid = params['path_vid']
path_mask = params['path_mask']

DEVICE = bnpm.torch_helpers.set_device(use_GPU=False)
print(f'Device set: {DEVICE}')

mask = torch.as_tensor(np.load(path_mask), dtype=torch.float32).to(DEVICE)
print(f'Mask loaded: {path_mask}')
vid = decord.VideoReader(path_vid, ctx=decord.cpu(0))
print(f'Video loaded: {path_vid}')

prepare_frames = lambda f: torch.as_tensor(f.asnumpy()[...,0], dtype=torch.float32).to(DEVICE)
extract_trace = lambda f: torch.einsum('fhw,hw -> f', prepare_frames(f), mask)

trace_start = torch.cat([extract_trace(v) for v in bnpm.indexing.make_batches(vid, batch_size=100, length=10000)], dim=0)

idx_fastForward = int(60*59 * vid.get_avg_fps())

trace_end = torch.cat([extract_trace(v) for v in bnpm.indexing.make_batches(vid, batch_size=100, idx_start=idx_fastForward)], dim=0)
print(f'traces_extracted. trace_start.shape: {trace_start.shape}, trace_end.shape: {trace_end.shape}')

get_diff_smooth = lambda x: np.diff(bnpm.timeSeries.simple_smooth(x, sig=4), n=1)

idx_start = np.argmax(get_diff_smooth(trace_start))

idx_end = np.argmin(get_diff_smooth(trace_end)) + idx_fastForward
print(f'indices found. idx_start: {idx_start}, idx_end: {idx_end}')

bnpm.file_helpers.pickle_save(
    obj={
        'idx_start': idx_start,
        'idx_end': idx_end,
        'trace_start': trace_start,
        'trace_end': trace_end,
        'path_vid': path_vid,
        'path_mask': path_mask,
    },
    filepath=str(Path(dir_save) / 'idx_eye_laser.pkl'),
)
print(f'idx_eye_laser.pkl saved to {dir_save}')