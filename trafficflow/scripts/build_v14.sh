#!/bin/sh
# V14 = V13 + Round 10 transductive Task 1 corrections (main rows and gap rows) + Round 11 onset prior correction.
# Runs after reproduce.sh (needs its caches: t1a_*, t1p_gapr_*, t2p_lgb_vt, t2p_cnn_vt_s0/1, t2_onset.parquet, odme_l2.csv).
set -e
cd "$(dirname "$0")/.."
export TFB_CF=1 TFB_SCEN=split
C="${TFB_CACHE:-../data_tfb/cache}"; O="${TFB_OUT:-../out/tfb}"
step() { echo "== $(date -u +%H:%M:%S) $*"; }
step "a Task 1 correction frames + fit (6 passes x 6%, 300 rounds)"
TFB_TX_PASSES=6 python3 -m tfb.t1_tx_prod frames && TFB_TX_PASSES=6 TFB_TX_ROUNDS=300 python3 -m tfb.t1_tx_prod fit
step "b gap correction frames + fit (4 passes, 200 rounds)"
python3 -m tfb.t1_txgap_prod frames && python3 -m tfb.t1_txgap_prod fit
step "c Task 1 file";   [ -f "$O/state_v14.csv" ] || TFB_T1_MAIN=t1a TFB_T1_TX=1 TFB_T1_TXG=1 python3 -m tfb.t1_predict v14
step "d Task 2 file (onset prior correction; CNN seeds 0, 1)"
[ -f "$O/queue_v14.csv" ] || TFB_ONSET_PRIOR=1 TFB_CNN_SEEDS=0,1 python3 -m tfb.t2_prod assemble v14
step "e merge";         python3 -m tfb.make_sub state_v14.csv queue_v14.csv odme_l2.csv v14
step "done"
