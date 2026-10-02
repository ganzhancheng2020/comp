#!/bin/sh
# Verification run of reproduce.sh in a separate cache/output (raw-data npz caches symlinked from data_tfb/cache).
export TFB_CACHE=/home/user/comp/data_tfb/cache_repro TFB_OUT=/home/user/comp/out/repro
/home/user/comp/trafficflow/reproduce.sh >> /home/user/comp/repro.log 2>&1
