#!/bin/sh
cd /home/user/comp/trafficflow
export TFB_CF=1
python3 -m tfb.t1_all frame >> /home/user/comp/t1_all.log 2>&1 && \
python3 -m tfb.t1_all train speed >> /home/user/comp/t1_all.log 2>&1 && \
python3 -m tfb.t1_all train flow >> /home/user/comp/t1_all.log 2>&1 && \
python3 -m tfb.t1_all eval >> /home/user/comp/t1_all.log 2>&1
