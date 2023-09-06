1. Download camera csv files to local
2. Eye_laser_masks
   a. Make eye_laser masks locally
   b. Run eye_laser_extraction on o2
4. Face-rhythm
   a. Download a single camera video to local
   b. Make ROI masks locally
   c. Run face-rythm on one video locally.
   d. Use server to run face-rhythm on the rest of the videos using the masks and FR parameters.
   e. Download all FR outputs to local
   f. Run big TCA on local
5. Download wavesurfer files
6. Run trace_quality_metrics and calculate dFoF
   a. Download one full s2p output folder to local
   b. Run tqm on it to find parameters locally
   c. Use server to calculate dFoF on all the sessions
   d. Download all dFoF and tqm files to local
7. Run ROICaT
   a. Download all stat_and_ops files to local
   b. Run ROICaT tracking and classification locally. If it is a BMI animal, probably use the day 0 iscell instead.
   c. Align dFoF across sessions using above tracking and classification outputs 
8. Download logger files to local
9. Download wavesurfer files to local
10. Temporally align wavesurfer and TCA (or other FR outputs) to ScanImage ('idxSI') locally