from pathlib import Path

from tqdm import tqdm

import bnpm.path_helpers, bnpm.file_helpers

dir_s2p_outer = str(Path(r'/media/rich/bigSSD/downloads_tmp/tmp_data/mouse_0322R/statFiles/').resolve())
path_save     = str(Path(r'/media/rich/bigSSD/analysis_data/BMI/trace_quality_posthoc/mouse_0322R/').resolve() / 'trace_quality_posthoc.pkl')

paths_tmp = bnpm.path_helpers.find_paths(
    dir_outer=dir_s2p_outer,
    reMatch='stat.npy',
    depth=4,
)

dates_of_files = [Path(d).parent.name for d in paths_tmp]

dirs_s2p_all = {d: str(Path(p).resolve().parent) for d, p in zip(dates_of_files, paths_tmp)}

paths_s2p = {fn: {d: str(Path(p) / fn) for d, p in dirs_s2p_all.items()} for fn in ['stat.npy', 'ops.npy', 'F.npy', 'Fneu.npy']}

print(dirs_s2p_all)

## == IMPORT DATA ==
data_s2p_all = {d: bnpm.ca2p_preprocessing.import_s2p(p) for d, p in dirs_s2p_all.items()}

data_s2p_all = {d: {
    'F': s[0], 
    'Fneu': s[1], 
    'iscell': s[2], 
    'ops': s[3], 
    'spks': s[4], 
    'stat': s[5]
} for d, s in data_s2p_all.items()}


results = {}

for date, data in tqdm(data_s2p_all.items()):
    F, Fneu, ops, stat = data['F'], data['Fneu'], data['ops'], data['stat']
    
    n_frames, n_rois = F.shape[1], F.shape[0]
    Fs = ops['fs']

    # channelOffset_correction = 0
    percentile_baseline = 30
    neuropil_fraction=0.7

    dFoF , dF , F_neuSub , F_baseline = bnpm.ca2p_preprocessing.make_dFoF(
        F=F,
        Fneu=Fneu,
        neuropil_fraction=neuropil_fraction,
        percentile_baseline=percentile_baseline,
        rolling_percentile_window=None,
        multicore_pref=True,
        verbose=True
    )

    dFoF_params = {
        "channelOffset_correction": 0,
        "percentile_baseline": percentile_baseline,
        "neuropil_fraction": neuropil_fraction,
    }

    # dFoF with reduced percentile for baseline
    channelOffset_correction = 0
    percentile_baseline = 30
    neuropil_fraction = 0.7
    win_rolling_percentile = 15*60*30

    dFoF_rollingPtile, dF_rollingPtile, F_neuSub_rollingPtile, F_baseline_rollingPtile = bnpm.ca2p_preprocessing.make_dFoF(
        F=F,
        Fneu=Fneu,
        neuropil_fraction=neuropil_fraction,
        percentile_baseline=percentile_baseline,
        rolling_percentile_window=win_rolling_percentile,
        multicore_pref=True,
        verbose=True
    )
    # # Threshold for nonnegativity
    # dFoF_z = dFoF / np.std(dFoF,axis=1,keepdims=True)

    dFoF_params = {
        "channelOffset_correction": 0,
        "percentile_baseline": percentile_baseline,
        "neuropil_fraction": neuropil_fraction,
    }


    thresh = {
        'var_ratio__Fneu_over_F': (0, 0.75),
        'EV__F_by_Fneu': (0, 0.75),
        'base_FneuSub': (100, 2000),
        'base_F': (200, 3500),
        'nsr_autoregressive': (0, 10),
        'noise_derivMAD': (0, 0.04),
        'max_dFoF': (0.75, 15),
        'baseline_var': (0, 0.02),
    }

    tqm, iscell_tqm = bnpm.ca2p_preprocessing.trace_quality_metrics(
        F=F,
        Fneu=Fneu,
        dFoF=dFoF_rollingPtile,
        F_neuSub=F_neuSub,
        F_baseline_roll=F_baseline_rollingPtile,
        percentile_baseline=percentile_baseline,
        window_rolling_baseline=win_rolling_percentile,
        Fs=Fs,
        plot_pref=True,
        thresh=thresh,
    )

    results[date] = {
        'tqm': tqm,
        'iscell_tqm': iscell_tqm,
    }

bnpm.file_helpers.pickle_save(
    obj={
        "tqm": tqm,
        "iscell_tqm": iscell_tqm,
        "dFoF_params": dFoF_params
    },
    filepath=path_save,
    mkdir=True,
    allow_overwrite=False,
)

# np.save(
#     file= dir_save / 'iscell_NN_tqm.npy',
#     arr=iscell_new
# )