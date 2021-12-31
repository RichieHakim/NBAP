import time
import numpy as np
import scipy.stats
import dateutil.parser

from numba import njit, jit, prange

from matplotlib import pyplot as plt

from .helpers import round_numba

def align_ws_toS2p(ws_galvoFlyBackTrace , num_frames_S2p , plot_pref):
    # Get ws frame times in ws time. Everything should be aligned to ws frame times (S2pInd)
    
    # Outputs:
    # - ws_YGalvoFlybacks_bool_wsTime
    # - ws_frameTimes_wsTime  

    tic = time.time()
    ws_YGalvoFlybacks_bool_wsTime = np.diff(np.int8(np.diff(ws_galvoFlyBackTrace) < -1)) > 0.5

    ws_frameTimes_wsTime = np.array(np.where(ws_YGalvoFlybacks_bool_wsTime))[0,:]
    ws_frameTimes_wsTime = ws_frameTimes_wsTime[0:num_frames_S2p+1]

    if plot_pref:
        plt.figure()
        plt.plot(ws_frameTimes_wsTime,np.ones(len(ws_frameTimes_wsTime)),'.')
        plt.plot(ws_galvoFlyBackTrace)
    print(f'frames in scanimage movie = {num_frames_S2p}')
    print(f'frames from ws galvo extraction movie = {ws_frameTimes_wsTime.shape[0]}')

    ws_samples_per_S2p_frame_rough = (ws_frameTimes_wsTime[-1] - ws_frameTimes_wsTime[0]) / num_frames_S2p
    print(f'number of wavesurfer samples per imaging frame:  {ws_samples_per_S2p_frame_rough}')
    print(f'Completed aligning WS to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return ws_YGalvoFlybacks_bool_wsTime , ws_frameTimes_wsTime , ws_samples_per_S2p_frame_rough


def align_licks_toS2p(ws_licks , threshold , ws_frameTimes_wsTime , num_frames_S2p , plot_pref):
    ## Get lick times in S2p ind

    # Outputs:
    # - ws_licks_bool_wsTime
    # - ws_licks_bool_S2pInd
    # - ws_lickTimes_S2pInd  

    tic = time.time()
    ws_licks_bool_wsTime = np.diff(np.int8(np.diff(ws_licks) < threshold)) > 0.5

    ws_licks_bool_S2pInd = np.zeros(num_frames_S2p)
    for frame_num, frame_ind in enumerate(ws_frameTimes_wsTime[0:]):
        if frame_num==0:
            continue
        ws_licks_bool_S2pInd[frame_num] = sum(ws_licks_bool_wsTime[ws_frameTimes_wsTime[frame_num-1] : ws_frameTimes_wsTime[frame_num]]) > 0.5

    ws_lickTimes_S2pInd = np.array(np.where(ws_licks_bool_S2pInd))

    if plot_pref:
        plt.figure()
        plt.plot(ws_licks)
        plt.plot(ws_licks_bool_wsTime)
        plt.plot(ws_frameTimes_wsTime, ws_licks_bool_S2pInd)
    print(f'Completed aligning Licks to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return ws_licks_bool_wsTime , ws_licks_bool_S2pInd , ws_lickTimes_S2pInd


def align_rewards_toS2p(ws_rewards , threshold , ws_frameTimes_wsTime , num_frames_S2p , plot_pref):
    ## Get reward delivery times in S2p ind

    # Outputs:
    # - ws_rewards_bool_wsTime
    # - ws_rewards_bool_S2pInd
    # - ws_rewardTimes_S2pInd  

    tic = time.time()
    ws_rewards_bool_wsTime = np.diff(np.int8(np.diff(ws_rewards) > threshold)) > 0.5


    ws_rewards_bool_S2pInd = np.zeros(num_frames_S2p)
    for frame_num, frame_ind in enumerate(ws_frameTimes_wsTime[0:]):
        if frame_num==0:
            continue
        ws_rewards_bool_S2pInd[frame_num] = sum(ws_rewards_bool_wsTime[ws_frameTimes_wsTime[frame_num-1] : ws_frameTimes_wsTime[frame_num]]) > 0.5

    ws_rewardTimes_S2pInd = np.array(np.where(ws_rewards_bool_S2pInd))

    if plot_pref:
        plt.figure()
        plt.plot(ws_rewards)
        plt.plot(ws_rewards_bool_wsTime)
        plt.plot(ws_frameTimes_wsTime, ws_rewards_bool_S2pInd)
    print(f'Completed aligning Rewards to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')
       
    return ws_rewards_bool_wsTime , ws_rewards_bool_S2pInd , ws_rewardTimes_S2pInd


def align_treadmill_toS2p(ws_treadmill , ws_frameTimes_wsTime , num_frames_S2p , ws_samples_per_S2p_frame_rough , plot_pref):
    ## Get treadmill times in S2p ind

    # Outputs:
    # - ws_treadmill_S2pInd
    # - ws_treadmill_bool_S2pInd

    tic = time.time()
    ws_treadmill_wsTime = ws_treadmill - np.percentile(ws_treadmill, 20) # this is to center it around quiescence

    ws_treadmill_S2pInd = np.zeros(num_frames_S2p)
    for frame_num, frame_ind in enumerate(ws_frameTimes_wsTime[0:]):
        if frame_num==0:
            continue
        ws_treadmill_S2pInd[frame_num] = sum(ws_treadmill_wsTime[ws_frameTimes_wsTime[frame_num-1] : ws_frameTimes_wsTime[frame_num]])
    ws_treadmill_S2pInd = ws_treadmill_S2pInd / ws_samples_per_S2p_frame_rough

    if plot_pref:
        plt.figure()
        plt.plot(ws_treadmill_wsTime)
        plt.plot(ws_frameTimes_wsTime, ws_treadmill_S2pInd)
    print(f'Completed aligning Treadmill to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')
       
    return ws_treadmill_S2pInd 


def extract_camPulses_camIdx(signal_GPIO_pulses , plot_pref):
    ## Get camera signal times in ws ind

    # Outputs:
    # - signal_GPIO_bool_camTime
    # - signal_GPIO_camTimes

    tic = time.time()
    input_signal_GPIO = scipy.stats.zscore(np.double(signal_GPIO_pulses))

    signal_GPIO_bool_camTime = np.abs(np.hstack(([0] , np.diff(input_signal_GPIO)))) > 1
    # plt.figure()
    # plt.plot(signal_GPIO_bool_camTime)

    signal_GPIO_camTimes = np.where(signal_GPIO_bool_camTime)[0]

    if plot_pref:
        plt.figure()
        plt.plot(input_signal_GPIO+1)
        plt.plot(signal_GPIO_camTimes, np.ones(len(signal_GPIO_camTimes)).T ,'.')
    print(f'Completed extracting camera pulses from camera data. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return signal_GPIO_bool_camTime , signal_GPIO_camTimes


def align_ws_camPulses_toWS(ws_camPulses , plot_pref):
    ## Get camera signal captured in ws converted to ws ind

    # Outputs:
    # - ws_camSignal_bool_wsTime
    # - ws_camSignal_wsTimes

    tic = time.time()
    ws_camSignal_bool_wsTime = np.abs(np.hstack(([0] , np.diff(ws_camPulses)))) > 1

    ws_camSignal_wsTimes = np.where(ws_camSignal_bool_wsTime)[0]

    if plot_pref:
        plt.figure()
        plt.plot(ws_camPulses)
        plt.plot(ws_camSignal_bool_wsTime)
        plt.plot(ws_camSignal_wsTimes, np.ones(len(ws_camSignal_wsTimes)).T ,'.')
    print(f'Completed aligning camera pulses to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return ws_camSignal_bool_wsTime , ws_camSignal_wsTimes


def convert_camTimeDates_toAbsoluteSeconds(camTimeDates, verbose=False):

    tic = time.time()
    n_timepoints = len(camTimeDates)
    camTimes_absolute = np.array(np.array(camTimeDates , dtype='datetime64') - np.datetime64(camTimeDates[0])  , dtype='float64')/10**9

    if verbose:
        print(f'Completed converting camera dates from camera data to absolute time. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return camTimes_absolute
   

def align_camFrames_toWS(signal_GPIO_camTimes , camTimes_absolute , ws_camSignal_wsTimes):
    ## Get camera frame times in ws ind

    tic = time.time()

    first_camPulse_camIdx = signal_GPIO_camTimes[0]
    last_camPulse_camIdx = signal_GPIO_camTimes[-1]
    camTimes_alignedToFirstPulse = camTimes_absolute - camTimes_absolute[first_camPulse_camIdx]
    camTimes_wsInd = (camTimes_alignedToFirstPulse / camTimes_alignedToFirstPulse[signal_GPIO_camTimes[-1]]) * (ws_camSignal_wsTimes[-1] - ws_camSignal_wsTimes[0])
    camTimes_wsInd_rounded = np.int64(camTimes_wsInd)
    print(f'Completed aligning camera frames to wavesurfer. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return camTimes_wsInd , camTimes_wsInd_rounded , first_camPulse_camIdx , last_camPulse_camIdx


def align_camSignal_toS2p_andToWS(camSignal , camTimes_wsInd , num_camera_frames , ws_frameTimes_wsTime , first_camPulse_camIdx , last_camPulse_camIdx , downsample_factor=None, plot_pref=False):
    ### resample temporalFactors_faceRhythm into ws and s2p times
    # this script assumes that the GPIO sync trace has identical indexing as the video that was used for the factor decomposition (hopefully it's the same video)

    tic = time.time()
    upsample_factor = num_camera_frames / camSignal.shape[0]

    camSignal_camFrameTimes = np.int64(np.round(np.linspace(upsample_factor , num_camera_frames , camSignal.shape[0] )))
    camSignal_Idx_withinWS = [(camSignal_camFrameTimes >= first_camPulse_camIdx) * 
                              (camSignal_camFrameTimes <= last_camPulse_camIdx)]
    camSignal_withinWS = camSignal[tuple(camSignal_Idx_withinWS)]
    camSignal_camFrameTimes_withinWS = camSignal_camFrameTimes[tuple(camSignal_Idx_withinWS)]
    camTimes_wsInd_withinWS = camTimes_wsInd[camSignal_camFrameTimes_withinWS]

    function_interp = scipy.interpolate.interp1d(camTimes_wsInd_withinWS , camSignal_withinWS , kind='cubic' , axis=0)
    first_s2pIdx_usable = np.min(np.where(ws_frameTimes_wsTime > np.min(camTimes_wsInd_withinWS)))

    camSignal_s2pInd = function_interp(ws_frameTimes_wsTime[first_s2pIdx_usable::downsample_factor])
    chunk_to_concatenate = np.zeros(tuple(np.concatenate((np.array([first_s2pIdx_usable]), camSignal_s2pInd.shape[1:]))))
    camSignal_s2pInd = np.concatenate((chunk_to_concatenate, camSignal_s2pInd) , axis=0)
    camSignal_s2pInd[camSignal_s2pInd < 0] = 0

    if plot_pref:
        plt.figure()
        plt.plot(camSignal[:,0])
        plt.plot(camSignal_s2pInd[:,0])
    print(f'Completed aligning camera signal to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return camSignal_s2pInd , first_s2pIdx_usable


# def align_faceTensor_toS2p_andToWS(camSignal , camTimes_wsInd , num_camera_frames , ws_frameTimes_wsTime , first_camPulse_camIdx , last_camPulse_camIdx , downsample_factor=None, plot_pref=False):
#     ### resample the big Sxx_all tensor from face-rhythm into (optionally downsampled) ws and s2p times
#     # this script assumes that the GPIO sync trace has similar indexing as the video and 'camSignal'
#     #  though 'camSignal' is allow to be downsampled relative to the original (num_camera_frames) framerate

#     tic = time.time()
#     upsample_factor = num_camera_frames / camSignal.shape[0]

#     camSignal_camFrameTimes = np.int64(np.round(np.linspace(upsample_factor , num_camera_frames , camSignal.shape[0] )))
#     camSignal_Idx_withinWS = [(camSignal_camFrameTimes >= first_camPulse_camIdx) * 
#                               (camSignal_camFrameTimes <= last_camPulse_camIdx)]
#     camSignal_withinWS = camSignal[tuple(camSignal_Idx_withinWS)]
#     camSignal_camFrameTimes_withinWS = camSignal_camFrameTimes[tuple(camSignal_Idx_withinWS)]
#     camTimes_wsInd_withinWS = camTimes_wsInd[camSignal_camFrameTimes_withinWS]

#     function_interp = scipy.interpolate.interp1d(camTimes_wsInd_withinWS , camSignal_withinWS , kind='cubic' , axis=0)
#     first_s2pIdx_usable = np.min(np.where(ws_frameTimes_wsTime > np.min(camTimes_wsInd_withinWS)))

#     camSignal_s2pInd = function_interp(ws_frameTimes_wsTime[first_s2pIdx_usable::downsample_factor])
#     chunk_to_concatenate = np.zeros(tuple(np.concatenate((np.array([first_s2pIdx_usable]), camSignal_s2pInd.shape[1:]))))
#     camSignal_s2pInd = np.concatenate((chunk_to_concatenate , camSignal_s2pInd) , axis=0)
#     camSignal_s2pInd[camSignal_s2pInd < 0] = 0

#     if plot_pref:
#         plt.figure()
#         plt.plot(camSignal_s2pInd[:,0])
#         plt.plot(camSignal[:,0])
#     print(f'Completed aligning camera signal to S2p. Total elapsed time: {round(time.time() - tic,2)} seconds')

#     return camSignal_s2pInd , first_s2pIdx_usable