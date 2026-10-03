#!/bin/sh
# Verification of reproduce.sh steps 10-13 (V14a) in the separate reproduction cache built by verify_repro.sh.
set -e
export TFB_CACHE=/home/user/comp/data_tfb/cache_repro TFB_OUT=/home/user/comp/out/repro TFB_CF=1 TFB_SCEN=split
cd /home/user/comp/trafficflow
TFB_TX_PASSES=6 python3 -m tfb.t1_tx_prod frames && TFB_TX_PASSES=6 TFB_TX_ROUNDS=300 python3 -m tfb.t1_tx_prod fit
python3 -m tfb.t1_txgap_prod frames && python3 -m tfb.t1_txgap_prod fit
TFB_T1_MAIN=t1a TFB_T1_TX=1 TFB_T1_TXG=1 python3 -m tfb.t1_predict v14a
python3 -m tfb.make_sub state_v14a.csv queue_v13.csv odme_l2.csv v14a
echo "== done"
