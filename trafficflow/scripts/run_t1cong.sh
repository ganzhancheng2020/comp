#!/bin/sh
cd /home/user/comp/trafficflow
export TFB_CF=1 TFB_CONG=1
for step in frame train eval; do python3 -m tfb.t1_cong_ab $step >> /home/user/comp/t1_cong.log 2>&1 || exit 1; done
