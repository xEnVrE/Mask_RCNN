#!/bin/bash

source ../../../virtualenvs/mask-rcnn/bin/activate

python online_demo_bottles.py --width 320 --height 240 bottles_humanoids_training/mask_rcnn_ycb_video_training_0010.h5
