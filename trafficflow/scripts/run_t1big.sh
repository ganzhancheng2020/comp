#!/bin/sh
# restart-safe chain for round 3 (each step resumes from its cache)
cd /home/user/comp/trafficflow
export TFB_CF=1
python3 -m tfb.t1_big frame >> /home/user/comp/t1_big.log 2>&1 && \
python3 -m tfb.t1_big train >> /home/user/comp/t1_big.log 2>&1 && \
python3 -m tfb.t1_big eval >> /home/user/comp/t1_big.log 2>&1
