#!/bin/bash

source ../../../virtualenvs/mask-rcnn/bin/activate

python online_demo_yarp.py --width 640 --height 480 ycb_video_training20190328T0952/mask_rcnn_ycb_video_training_0060.h5
