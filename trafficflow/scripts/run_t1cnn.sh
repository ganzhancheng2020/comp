#!/bin/sh
cd /home/user/comp/trafficflow
export TFB_CF=1
python3 -m tfb.t1_cnn train 12 >> /home/user/comp/t1_cnn.log 2>&1
