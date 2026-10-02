#!/bin/sh
cd /home/user/comp/trafficflow
export TFB_CF=1
for step in frame train eval gframe gtrain geval; do
  python3 -m tfb.t1_cap $step >> /home/user/comp/t1_cap.log 2>&1 || exit 1
done
