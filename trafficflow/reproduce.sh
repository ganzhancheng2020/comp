#!/bin/sh
# TrafficFlowBench 2026, team Steins — single entry script reproducing the selected final submission (V13, public 0.88107).
#
#   ./reproduce.sh            # writes $TFB_OUT/sub_v13.zip (default out/tfb/sub_v13.zip at the repository root)
#
# Inputs : data_tfb/kaggle_public/  (kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench; unzip)
#          tfb_ref/                 (git clone https://github.com/jacky850/trafficflowbench-public tfb_ref), both at the
#                                    repository root; or set TFB_REL / TFB_CACHE / TFB_OUT.
# Machine: CPU only, 4 cores, 15 GB RAM, ~15 GB free disk; ~8-9 hours. Steps run strictly in sequence (never run torch
#          next to LightGBM: OpenMP spin-waiting slows torch ~30x). Every step caches its result and is skipped when
#          rerun, so the script can be restarted after an interruption.
set -e
cd "$(dirname "$0")"
export TFB_CF=1 TFB_SCEN=split
step() { echo "== $(date -u +%H:%M:%S) $*"; }

step "1/9 dense caches of the release";                python3 -m tfb.data
step "2/9 Task 4: L2 projection of the weak prior";    python3 -m tfb.t4 l2
step "3/9 Task 2 onset frame (train events)"
python3 -c "
import pandas as pd
from tfb import t2_onset as on
from tfb.data import CACHE
p = CACHE / 't2_onset.parquet'
if not p.exists():
    pd.concat([on.train_frame(x)[0] for x in on.T2P], ignore_index=True).to_parquet(p)"
step "4/9 Task 2 ongoing frame (history + published origin row)"; python3 -m tfb.t2_build_parts t2_ongoing_parts_vt 1 0.6
step "5/9 Task 2 ongoing LightGBM";                    [ -f "${TFB_CACHE:-../data_tfb/cache}/t2p_lgb_vt.txt" ] || python3 -m tfb.t2_prod lgb
step "6/9 Task 2 ongoing CNN seeds 0, 1"
for s in 0 1; do [ -f "${TFB_CACHE:-../data_tfb/cache}/t2p_cnn_vt_s$s.pt" ] || python3 -m tfb.t2_prod cnn $s 25; done
step "   Task 2 file";                                 python3 -m tfb.t2_prod assemble v13
step "7/9 Task 1 main models on all train targets"
python3 -m tfb.t1_all frame && python3 -m tfb.t1_all train speed && python3 -m tfb.t1_all train flow
step "8/9 Task 1 gap specialist (ramp congestion sensor)"
[ -f "${TFB_CACHE:-../data_tfb/cache}/t1p_gapr_flow.txt" ] || python3 -m tfb.t1_gap_prod - - 3000 3500
step "   Task 1 file (split-own plateaus and profiles)"; TFB_T1_MAIN=t1a python3 -m tfb.t1_predict v13
step "9/9 merge into the Kaggle upload";               python3 -m tfb.make_sub state_v13.csv queue_v13.csv odme_l2.csv v13
step "done"
