import pynwb
import numpy as np
from numba import njit, jit, prange

def dump_nwb(nwb_path):
    """
    Print out nwb contents

    Args:
        nwb_path (str): path to the nwb file

    Returns:
    """
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        for interface in nwbfile.processing['Face Rhythm'].data_interfaces:
            print(interface)
            time_series_list = list(nwbfile.processing['Face Rhythm'][interface].time_series.keys())
            for ii, time_series in enumerate(time_series_list):
                data_tmp = nwbfile.processing['Face Rhythm'][interface][time_series].data
                print(f"     {time_series}:    {data_tmp.shape}   ,  {data_tmp.dtype}   ,   {round((data_tmp.size * data_tmp.dtype.itemsize)/1000000000, 6)} GB")
    

@njit(parallel=True)
def round_numba(x):
    output = np.zeros_like(x)
    for ii in prange(x.shape[0]):
        for jj in prange(x.shape[1]):
            output[ii,jj] = np.round(x[ii,jj])
    return output