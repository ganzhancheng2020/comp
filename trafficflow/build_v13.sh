#!/bin/sh
# One-command rebuild of the V13 submission (public 0.88107) from the raw Kaggle release.
# Prereqs: data in data_tfb/kaggle_public (kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench),
# official toolkit at ../tfb_ref (git clone https://github.com/jacky850/trafficflowbench-public), pip: numpy pandas
# pyarrow lightgbm torch scipy. CPU only (4 cores, 15 GB RAM); never run torch next to LightGBM.
set -e
cd "$(dirname "$0")"
export TFB_CF=1 TFB_SCEN=split
python3 -m tfb.data                                      # dense caches
python3 -m tfb.t4 l2                                     # Task 4: L2 projection of the prior onto the counts
python3 -c "import pandas as pd; from tfb import t2_onset as on; from tfb.data import CACHE
pd.concat([on.train_frame(p)[0] for p in on.T2P], ignore_index=True).to_parquet(CACHE/'t2_onset.parquet')"
python3 -m tfb.t2_build_parts t2_ongoing_parts_vt 1 0.6  # ongoing frame with the published origin row
python3 -m tfb.t2_prod lgb                               # ongoing LightGBM
python3 -m tfb.t2_prod cnn 0 25 && python3 -m tfb.t2_prod cnn 1 25   # ongoing CNN seeds
python3 -m tfb.t2_prod assemble v12                      # Task 2 file (onset V10a recipe + ongoing blend)
python3 -m tfb.t1_all frame && python3 -m tfb.t1_all train speed && python3 -m tfb.t1_all train flow   # T1 main
python3 -m tfb.t1_prod 300000 4000 >/dev/null 2>&1 || true   # builds the 300k frame used by the gap specialist recipe
python3 -m tfb.t1_gap_prod state_cf.csv state_cfr_gap.csv 3000 3500  # ramp-aware gap specialist (t1p_gapr_*)
TFB_T1_MAIN=t1a python3 -m tfb.t1_predict v13            # Task 1 file (split-own plateaus and profiles)
python3 -m tfb.make_sub state_v13.csv queue_v12.csv odme_l2.csv v13   # -> out/tfb/sub_v13.zip
